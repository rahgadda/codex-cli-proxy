# Codex CLI Proxy

## Overview

Codex CLI Proxy exposes a signed-in local Codex CLI as a localhost-only REST
service. It accepts OpenAI Chat Completions and Anthropic Messages request
formats and returns compatible response shapes.

Each request runs in its own empty temporary workspace, which is removed when
the Codex turn finishes. The proxy therefore does not provide its own project
directory as chat context. It supports text and base64-encoded image inputs.

## Installation

Prerequisites:

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- A signed-in local `codex` CLI

Install the project dependencies:

```bash
uv sync
```

## Run

```bash
uv run codex-cli-proxy
```

The server listens on `http://127.0.0.1:9000` by default.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/healthz`, `/readyz` | Proxy and Codex executable health status. |
| `GET` | `/v1/models` | Models available through the signed-in Codex CLI. |
| `GET` | `/api/tags` | Models available through the signed-in Codex CLI, in Ollama's tags response format. |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions-compatible chat endpoint. |
| `POST` | `/v1/messages` | Anthropic Messages-compatible chat endpoint. |

Both chat endpoints support `stream: true`; output is returned as
compatibility streaming after the Codex turn completes.

### Inputs

- Text content is supported.
- Images are supported as base64 payloads:
  - OpenAI: `image_url` or `input_image` with a `data:image/...;base64,...` URL.
  - Anthropic: `image` with a base64 `source` block.
  - JPEG, PNG, GIF, and WebP are accepted; up to 10 images and 20 MB per image.
- Audio and caller-supplied tools/functions are not supported by the local
  Codex CLI execution path and return a `400` error.

Use the standard `model` field to select a Codex model. If omitted, Codex uses
its configured default. `mode` is accepted as a compatibility alias.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_BIN` | `codex` | Path or command for the local Codex CLI. |
| `CODEX_SANDBOX` | `read-only` | Codex sandbox mode for each request. |
| `CODEX_TIMEOUT_SECONDS` | `900` | Maximum duration of one Codex turn. |
| `CODEX_MAX_CONCURRENCY` | `1` | Maximum concurrent Codex turns. |
| `HOST` / `PORT` | `127.0.0.1` / `9000` | Server bind address and port. |

Keep the service behind a trusted network boundary.

## Examples

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

## References

- [Request examples (`test.rest`)](https://github.com/rahgadda/codex-cli-proxy/blob/main/test.rest)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
