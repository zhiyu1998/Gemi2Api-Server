# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a FastAPI-based server that provides OpenAI-compatible API endpoints for Google's Gemini AI model via the `gemini-webapi` library. The server acts as a bridge, translating OpenAI API requests to Gemini API calls.

### Core Components

- **main.py**: Single-file application containing all API endpoints, authentication, and request handling
- **gemini_models.py**: Built-in model catalog (Gemini 4 Hydrogen/Carbon + 3.x) with name/alias/tier resolution and model-header builders. `ModelSpec` carries hex id, mode (header index 14 / `inner[79]`), think wire (1 normal / 2 extended / 4 medium) and capacity.
- **gemini_wire.py**: HTTP-layer request shim. It patches the underlying session's `stream()` so every StreamGenerate request gets a canonical `x-goog-ext-525001261-jspb` header and rewritten `f.req` slots (`inner[79]`=mode, `inner[80]`=think, plus the mode-8 slots `inner[49]=40`, `inner[68]=2`, `inner[83]/[91]/[96]/[98]=1`). Model choice travels through a `ContextVar`, so it is per-request and safe under concurrency. This keeps behaviour identical across gemini-webapi 2.0 (69-slot inner, header verbatim) and 2.1 (81-slot inner, mirrors header model number).
- **Authentication**: Uses Gemini cookies (`SECURE_1PSID`, `SECURE_1PSIDTS`) for Gemini API access and optional `API_KEY` for server authentication
- **API Endpoints**:
  - `GET /`: Health check endpoint
  - `GET /v1/models`: Lists available Gemini models in OpenAI format
  - `POST /v1/chat/completions`: Main chat completion endpoint (supports streaming)

### Gemini 4 notes

- Model selection is hex + mode + think wire, not the deprecated `Model` enum. `gemini-4-hydrogen` shares `56fdd199312815e2` with `gemini-3.8-flash`; `gemini-4-carbon` / `gemini-4-argon` is `95a4c0abcba24364` with mode 8.
- Thinking tier is a **request field**, not a model name: `reasoning_effort` (`low`=1, `medium`=4, `high`=2; `thinking_effort` / `think_effort` accepted as aliases). Do not add `-thinking` suffixes back to the catalog.
- Mode 8 needs `inner[49]=40` or paid accounts hang; that slot also disables the High thinking tier, which is why `high` on Argon answers at the normal tier.
- `custom_models.yaml` headers are parsed back into a spec (`parse_header_spec`), so legacy hand-written long headers keep working after the library stopped passing them through verbatim.

### Key Features

- OpenAI-compatible chat completions API
- Streaming response support
- Image processing (base64 encoded images via temporary files)
- Markdown link correction for Google search results
- CORS enabled for web clients
- Docker containerization with uv package manager

## Coding Conventions

When making changes to this codebase, please adhere to the following principles:

- **Keep It Simple, Stupid (KISS):** Write code that is simple, straightforward, and easy to understand. Avoid introducing unnecessary complexity.
- **Don't Repeat Yourself (DRY):** Instead of duplicating code for similar functionalities, create generic, reusable functions. A good example is the `initProviderFilter` function in `scripts/settings.js`, which handles filtering logic for multiple providers in a unified way.
- **Centralize Configuration:** Group related configurations together to make the code easier to maintain and extend. For instance, the `filterConfigurations` array in `scripts/settings.js` centralizes all the settings for the provider-specific filters, making it easy to add new ones in the future.

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install fastapi uvicorn gemini-webapi

# Set up environment variables (copy from example)
cp .env.example .env
# Edit .env with your Gemini credentials
```

### Running the Server
```bash
# Development server with auto-reload
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Production server
uvicorn main:app --host 0.0.0.0 --port 8000

# Using uv
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Code Quality
```bash
# Lint and format with ruff
ruff check .
ruff format .
```

### Docker Commands
```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs

# Rebuild and restart
docker-compose up -d --build

# Stop services
docker-compose down
```

## Configuration

### Required Environment Variables
- `SECURE_1PSID`: Gemini cookie for authentication (obtained from browser dev tools)
- `SECURE_1PSIDTS`: Gemini cookie timestamp for authentication
- `API_KEY`: Optional server authentication key
- `ENABLE_THINKING`: Optional boolean to enable thinking content in responses (default: false)

### Code Style
- Uses ruff for linting and formatting
- Line length: 150 characters
- Tab-based indentation
- Double quotes for strings
- Ignores E501 (line length warnings due to custom 150 char limit)

## Model Mapping

`resolve_model_spec()` in `main.py` turns a requested model name into a `ModelSpec`: custom models (`CUSTOM_MODELS_FILE`) win, then the built-in catalog in `gemini_models.py`, then a keyword fallback (flash/hydrogen → 4 Hydrogen, pro/carbon → 4 Carbon). Unknown names return 400 instead of silently falling back to another model. Thinking tier comes from `reasoning_effort`, not from the model name.

## Request Flow

1. Client sends OpenAI-compatible request to `/v1/chat/completions`
2. Server authenticates using optional API_KEY
3. Messages are converted from OpenAI format to conversation string
4. Images are decoded from base64 and saved to temporary files
5. Request is sent to Gemini via `gemini-webapi`
6. Response is processed, markdown corrected, and returned in OpenAI format
7. Temporary files are cleaned up