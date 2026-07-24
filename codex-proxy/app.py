"""Local Codex CLI proxy with OpenAI and Anthropic-compatible endpoints.

The service deliberately exposes only text chat. Codex itself can use its own
tools; callers cannot inject arbitrary external tool calls through this proxy.
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


CODEX_BIN = os.getenv("CODEX_BIN", "codex")
CODEX_SANDBOX = os.getenv("CODEX_SANDBOX", "read-only").strip()
CODEX_TIMEOUT_SECONDS = float(os.getenv("CODEX_TIMEOUT_SECONDS", "900"))
CODEX_EPHEMERAL = env_bool("CODEX_EPHEMERAL", True)
CODEX_MAX_CONCURRENCY = max(1, int(os.getenv("CODEX_MAX_CONCURRENCY", "1")))
KEEPALIVE_SECONDS = float(os.getenv("KEEPALIVE_SECONDS", "10"))

app = FastAPI(title="Codex CLI OpenAI/Anthropic compatibility proxy", version="0.1.0")
_codex_slots = asyncio.Semaphore(CODEX_MAX_CONCURRENCY)
# Reuse Uvicorn's configured application logger so these messages are visible
# alongside its normal request logs.
logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class CodexResult:
    text: str
    usage: Usage
    thread_id: str | None = None


class CodexRunError(RuntimeError):
    """Raised when the local Codex CLI cannot complete a request."""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def error_type(status: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error", 429: "rate_limit_error"}.get(status, "api_error")


async def request_json(request: Request) -> Any:
    """Read a JSON request body and expose malformed JSON as a client error."""
    try:
        return await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON request") from exc


@app.exception_handler(StarletteHTTPException)
async def formatted_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = new_id("req")
    if request.url.path.rstrip("/").endswith("/messages"):
        return JSONResponse(status_code=exc.status_code, content={"type": "error", "error": {"type": error_type(exc.status_code), "message": str(exc.detail)}, "request_id": request_id})
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": str(exc.detail), "type": error_type(exc.status_code), "param": None, "code": None}}, headers={"x-request-id": request_id})


@app.exception_handler(Exception)
async def formatted_unhandled_error(request: Request, _: Exception) -> JSONResponse:
    request_id = new_id("req")
    if request.url.path.rstrip("/").endswith("/messages"):
        return JSONResponse(status_code=500, content={"type": "error", "error": {"type": "api_error", "message": "Internal proxy error"}, "request_id": request_id})
    return JSONResponse(status_code=500, content={"error": {"message": "Internal proxy error", "type": "server_error", "param": None, "code": None}}, headers={"x-request-id": request_id})


def extract_text(content: Any) -> str:
    """Normalize OpenAI and Anthropic text content blocks into a prompt string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part for item in content if (part := extract_text(item)))
    if isinstance(content, dict):
        kind = str(content.get("type", ""))
        if kind in {"image", "image_url", "input_image", "audio", "input_audio", "tool_use", "tool_result", "function_call", "function_call_output"}:
            raise HTTPException(status_code=400, detail="Only text input is supported by this proxy")
        if kind in {"text", "input_text", "output_text"}:
            text = content.get("text", "")
            return str(text.get("value", "") if isinstance(text, dict) else text)
        return extract_text(content["content"]) if "content" in content else str(content.get("text", ""))
    return str(content)


def validate_request(body: dict[str, Any]) -> None:
    if body.get("tools") or body.get("functions"):
        raise HTTPException(status_code=400, detail="External tool calling is not implemented by this proxy")
    if body.get("n", 1) not in (None, 1):
        raise HTTPException(status_code=400, detail="Only n=1 is supported")


def requested_model(body: dict[str, Any]) -> str | None:
    """Return an optional caller-selected Codex model.

    ``model`` is the standard OpenAI/Anthropic field. ``mode`` is accepted as
    a compatibility alias for clients using that spelling.
    """
    model = body.get("model") or body.get("mode")
    if model is None:
        return None
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model must be a non-empty string")
    return model.strip()


def build_prompt(messages: Any, system: Any = None) -> str:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    transcript: list[dict[str, str]] = []
    if system is not None and (text := extract_text(system).strip()):
        transcript.append({"role": "system", "content": text})
    for raw in messages:
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object")
        role = str(raw.get("role", ""))
        if role not in {"system", "developer", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Unsupported message role: {role}")
        if raw.get("tool_calls") or raw.get("function_call"):
            raise HTTPException(status_code=400, detail="External tool calling is not implemented by this proxy")
        transcript.append({"role": role, "content": extract_text(raw.get("content"))})
    if not any(message["role"] == "user" for message in transcript):
        raise HTTPException(status_code=400, detail="At least one user message is required")
    prompt = "You are serving a chat API request. Interpret each role literally and answer the final user message. Return only the assistant response.\n\nCONVERSATION_JSON:\n" + json.dumps(transcript, ensure_ascii=False)
    return prompt


async def run_codex(prompt: str, model: str | None = None, request_id: str | None = None) -> CodexResult:
    if shutil.which(CODEX_BIN) is None and not Path(CODEX_BIN).exists():
        raise CodexRunError(f"Codex executable not found: {CODEX_BIN}")
    async with _codex_slots:
        # Keep each chat turn separate from the proxy's launch directory.
        # Codex sees only this empty, per-request workspace rather than the
        # repository from which the server was started.
        with tempfile.TemporaryDirectory(prefix="codex-cli-proxy-request-") as workspace:
            output_path = Path(workspace) / "last-message.txt"
            process: asyncio.subprocess.Process | None = None
            try:
                started_at = time.monotonic()
                logger.info("Request %s: Codex execution started", request_id or "unknown")
                command = [CODEX_BIN, "exec", "--json", "--color", "never", "--output-last-message", str(output_path), "--sandbox", CODEX_SANDBOX, "--skip-git-repo-check"]
                if CODEX_EPHEMERAL:
                    command.append("--ephemeral")
                if model:
                    command.extend(("--model", model))
                try:
                    process = await asyncio.create_subprocess_exec(*command, cwd=workspace, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                except OSError as exc:
                    raise CodexRunError(f"Could not start Codex: {exc}") from exc
                stdout, stderr = await asyncio.wait_for(process.communicate(prompt.encode()), timeout=CODEX_TIMEOUT_SECONDS)
                usage, thread_id, fallback_text = Usage(), None, ""
                for line in stdout.splitlines():
                    with contextlib.suppress(json.JSONDecodeError):
                        event = json.loads(line)
                        if event.get("type") == "thread.started":
                            thread_id = event.get("thread_id")
                        elif event.get("type") == "turn.completed":
                            raw = event.get("usage") or {}
                            usage = Usage(int(raw.get("input_tokens") or 0), int(raw.get("cached_input_tokens") or 0), int(raw.get("output_tokens") or 0), int(raw.get("reasoning_output_tokens") or 0))
                        elif event.get("type") == "item.completed" and (item := event.get("item") or {}).get("type") == "agent_message":
                            fallback_text = str(item.get("text") or fallback_text)
                if process.returncode:
                    raise CodexRunError(f"Codex exited with status {process.returncode}: {stderr.decode(errors='replace')[-4000:].strip() or 'no stderr output'}")
                text = output_path.read_text(encoding="utf-8", errors="replace").strip() or fallback_text.strip()
                if not text:
                    raise CodexRunError("Codex completed without a final agent message")
                logger.info("Request %s: Codex execution completed in %.2fs", request_id or "unknown", time.monotonic() - started_at)
                return CodexResult(text, usage, thread_id)
            except TimeoutError as exc:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                raise CodexRunError(f"Codex timed out after {CODEX_TIMEOUT_SECONDS:g} seconds") from exc


def openai_usage(usage: Usage) -> dict[str, Any]:
    return {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens, "total_tokens": usage.total_tokens, "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens}, "completion_tokens_details": {"reasoning_tokens": usage.reasoning_output_tokens}}


def anthropic_usage(usage: Usage) -> dict[str, int]:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": usage.cached_input_tokens}


async def list_codex_models() -> list[dict[str, Any]]:
    """Read the signed-in CLI's currently available model catalog."""
    if shutil.which(CODEX_BIN) is None and not Path(CODEX_BIN).exists():
        raise CodexRunError(f"Codex executable not found: {CODEX_BIN}")
    try:
        process = await asyncio.create_subprocess_exec(
            CODEX_BIN, "app-server", "--stdio",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise CodexRunError(f"Could not start Codex app server: {exc}") from exc

    stdin, stdout, stderr_stream = process.stdin, process.stdout, process.stderr
    if stdin is None or stdout is None or stderr_stream is None:
        process.kill()
        await process.wait()
        raise CodexRunError("Could not open Codex app server standard streams")
    stderr_task = asyncio.create_task(stderr_stream.read())
    try:
        deadline = asyncio.get_running_loop().time() + CODEX_TIMEOUT_SECONDS

        async def rpc(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
            stdin.write((compact_json({"id": request_id, "method": method, "params": params}) + "\n").encode())
            await stdin.drain()
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                line = await asyncio.wait_for(stdout.readline(), timeout=remaining)
                if not line:
                    return {}
                with contextlib.suppress(json.JSONDecodeError):
                    response = json.loads(line)
                    if response.get("id") != request_id:
                        continue
                    if error := response.get("error"):
                        raise CodexRunError(f"Codex {method} failed: {error.get('message', error)}")
                    result = response.get("result")
                    return result if isinstance(result, dict) else {}

        await rpc(1, "initialize", {"clientInfo": {"name": "codex-cli-proxy", "version": "0.1.0"}})
        data = (await rpc(2, "model/list", {})).get("data")
        if isinstance(data, list):
            return [model for model in data if isinstance(model, dict)]
    except TimeoutError as exc:
        raise CodexRunError(f"Codex model listing timed out after {CODEX_TIMEOUT_SECONDS:g} seconds") from exc
    finally:
        if not stdin.is_closing():
            stdin.close()
            with contextlib.suppress(Exception):
                await stdin.wait_closed()
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode is None:
                process.kill()
                await process.wait()
        stderr = await stderr_task
    if process.returncode:
        raise CodexRunError(f"Codex model listing exited with status {process.returncode}: {stderr.decode(errors='replace')[-4000:].strip() or 'no stderr output'}")
    raise CodexRunError("Codex model listing returned no model catalog")


async def complete_with_keepalives(prompt: str, model: str | None, request_id: str) -> AsyncIterator[CodexResult | None]:
    task = asyncio.create_task(run_codex(prompt, model, request_id))
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
        if not done:
            yield None
    yield task.result()


async def openai_stream(prompt: str, model: str, selected_model: str | None, include_usage: bool, request_id: str) -> AsyncIterator[str]:
    completion_id, created = new_id("chatcmpl"), int(time.time())
    def event(delta: dict[str, Any], finish_reason: str | None = None, usage: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [] if usage else [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]}
        if usage:
            payload["usage"] = usage
        return f"data: {compact_json(payload)}\n\n"
    yield event({"role": "assistant", "content": ""})
    try:
        async for result in complete_with_keepalives(prompt, selected_model, request_id):
            if result is None:
                yield ": keep-alive\n\n"
            else:
                yield event({"content": result.text})
                yield event({}, "stop")
                if include_usage:
                    yield event({}, usage=openai_usage(result.usage))
    except Exception as exc:
        yield f"data: {compact_json({'error': {'message': str(exc), 'type': 'api_error'}})}\n\n"
    yield "data: [DONE]\n\n"


async def anthropic_stream(prompt: str, model: str, selected_model: str | None, request_id: str) -> AsyncIterator[str]:
    message_id = new_id("msg")
    yield f"event: message_start\ndata: {compact_json({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield 'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    try:
        async for result in complete_with_keepalives(prompt, selected_model, request_id):
            if result is None:
                yield 'event: ping\ndata: {"type":"ping"}\n\n'
            else:
                yield f"event: content_block_delta\ndata: {compact_json({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': result.text}})}\n\n"
                yield 'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
                yield f"event: message_delta\ndata: {compact_json({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': result.usage.output_tokens}})}\n\n"
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    except Exception as exc:
        yield f"event: error\ndata: {compact_json({'type': 'error', 'error': {'type': 'api_error', 'message': str(exc)}})}\n\n"


@app.get("/healthz")
@app.get("/readyz")
async def health() -> dict[str, Any]:
    executable_ok = shutil.which(CODEX_BIN) is not None or Path(CODEX_BIN).exists()
    return {"ok": executable_ok, "codex_executable": executable_ok, "sandbox": CODEX_SANDBOX}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    try:
        models = await list_codex_models()
    except CodexRunError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"object": "list", "data": [
        {"id": model["model"], "object": "model", "created": 0, "owned_by": "openai"}
        for model in models if isinstance(model.get("model"), str) and model["model"]
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    request_id = new_id("req")
    logger.info("Request %s started: %s %s", request_id, request.method, request.url.path)
    body = await request_json(request)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    validate_request(body)
    selected_model, prompt = requested_model(body), build_prompt(body.get("messages"))
    model = selected_model or "codex-default"
    if body.get("stream"):
        return StreamingResponse(openai_stream(prompt, model, selected_model, bool((body.get("stream_options") or {}).get("include_usage")), request_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        result = await run_codex(prompt, selected_model, request_id)
    except CodexRunError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"id": new_id("chatcmpl"), "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text, "refusal": None}, "logprobs": None, "finish_reason": "stop"}], "usage": openai_usage(result.usage)}


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Any:
    request_id = new_id("req")
    logger.info("Request %s started: %s %s", request_id, request.method, request.url.path)
    body = await request_json(request)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    validate_request(body)
    selected_model, prompt = requested_model(body), build_prompt(body.get("messages"), body.get("system"))
    model = selected_model or "codex-default"
    if body.get("stream"):
        return StreamingResponse(anthropic_stream(prompt, model, selected_model, request_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        result = await run_codex(prompt, selected_model, request_id)
    except CodexRunError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"id": new_id("msg"), "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": result.text}], "stop_reason": "end_turn", "stop_sequence": None, "usage": anthropic_usage(result.usage)}


def main() -> None:
    """Run the localhost-only API server."""
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "9000")))


if __name__ == "__main__":
    main()
