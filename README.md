# Codex CLI Proxy

Codex CLI Proxy turns a signed-in local Codex CLI into a localhost-only,
chat-only REST service. It accepts text conversations in OpenAI Chat
Completions and Anthropic Messages formats, then returns the matching response
shape without passing the server's project directory to Codex.

Every request runs in a new empty temporary directory that is deleted when the
turn finishes. Callers cannot submit images, audio, or tool/function calls.

## Run

```bash
uv run codex-cli-proxy
```

The server listens on `http://127.0.0.1:9000` by default.

## Endpoints

- `GET /healthz` and `GET /readyz`
- `GET /v1/models` — models currently available to the signed-in Codex CLI
- `POST /v1/chat/completions` — OpenAI Chat Completions shape
- `POST /v1/messages` — Anthropic Messages shape

Both POST endpoints accept `stream: true` and return server-sent events. A
Codex turn is completed before its response text is emitted, so streaming is
compatibility streaming rather than token-by-token streaming.

Send the standard `model` field to select a Codex model for that request. If
it is omitted, the proxy sends no `--model` flag and Codex uses its configured
default. `mode` is accepted as a compatibility alias.

## Request examples

See [test.rest](https://github.com/rahgadda/codex-cli-proxy/blob/main/test.rest)
for ready-to-run local OpenAI and Anthropic requests, plus a direct OpenAI
GPT-5.6 example that reads `OPENAI_API_KEY` from your REST client environment.

## Configuration

Set these environment variables before launch when needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_BIN` | `codex` | Path or command for the local Codex CLI. |
| `CODEX_SANDBOX` | `read-only` | Codex sandbox mode for each request. |
| `CODEX_TIMEOUT_SECONDS` | `900` | Maximum duration of one Codex turn. |
| `CODEX_MAX_CONCURRENCY` | `1` | Maximum concurrent Codex turns. |
| `HOST` / `PORT` | `127.0.0.1` / `9000` | Server bind address and port. |

Keep the service behind a trusted network boundary. The proxy rejects images,
audio, and caller-supplied tool/function calls.

## Examples

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

```bash
curl http://127.0.0.1:9000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"gpt-5.6-sol","max_tokens":512,"messages":[{"role":"user","content":"Write a one-sentence greeting."}]}'
```
