import asyncio
import base64
import hashlib
import hmac
import importlib.metadata
import io
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlparse

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")

import httpx
import numpy as np
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import MODEL_HEADER_KEY
from PIL import Image
from pydantic import BaseModel

from cookie_pool import (
	CookieLease,
	CookiePool,
	CookiePoolUnavailable,
	load_cookie_entries,
	load_extra_cookies,
)
from gemini_models import (
	BUILTIN_MODELS,
	FEATURE_TASK,
	TIER_SUFFIXES,
	ModelSpec,
	build_model_arg,
	fuzzy_builtin,
	normalize_model_name,
	parse_header_spec,
	resolve_builtin,
	think_from_effort,
)
from gemini_task import ensure_task_ready
from gemini_tools import (
	TOOL_FENCE_CLOSE,
	TOOL_FENCE_OPEN,
	ToolFenceGate,
	build_tools_instruction,
	format_tool_result,
	parse_tool_calls,
	tool_calls_for_stream,
)
from gemini_wire import active_spec as active_model_spec
from gemini_wire import install as install_wire_shim
from gemini_wire import pop_spec as gemini_wire_pop_spec
from gemini_wire import push_spec as gemini_wire_push_spec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_log_level("INFO")

gemini_client = None
gemini_client_lock = asyncio.Lock()

# 多 Cookie 轮询池：一个进程挂多个 Gemini 账号，请求轮询分配、失败自动换号
gemini_pool = None
gemini_pool_lock = asyncio.Lock()


def _env_number(name: str, default, cast=float):
	"""读一个数字型环境变量，写错就告警并回退默认值（别让一处拼写错误拖崩启动）。"""
	raw = (os.environ.get(name) or "").strip()
	if not raw:
		return default
	try:
		return cast(raw)
	except (TypeError, ValueError):
		logger.warning("环境变量 %s=%r 不是合法数字，回退到 %s", name, raw, default)
		return default


# 服务配置（从环境变量读取，修改后需重启服务生效）
HOST = os.environ.get("HOST", "0.0.0.0")
try:
	PORT = int(os.environ.get("PORT", "7860"))
except ValueError:
	raise ValueError(f"Invalid PORT environment variable: '{os.environ.get('PORT')}' must be an integer")


async def _init_gemini_client_background():
	"""Background task: initialize the Gemini client without blocking startup."""
	try:
		pool = await get_gemini_pool()
		await get_gemini_client()
		if COOKIE_EAGER_INIT:
			# 预热剩下的账号：启动时就把坏 Cookie 暴露出来，而不是等流量打上去
			await pool.warm_up()
		logger.info(
			"Gemini client initialized successfully in background (cookie accounts=%s: %s)",
			len(pool),
			", ".join(pool.labels),
		)
		# 顺手拉一次账号额度/套餐，管理面板一打开就有数据（失败不影响服务）
		try:
			await fetch_live_quota()
		except Exception:
			pass
	except Exception as e:
		logger.warning(f"Gemini client init failed (service running without API access): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Start Gemini client init in background; do not block FastAPI startup."""
	init_task = asyncio.create_task(_init_gemini_client_background())
	try:
		yield
	finally:
		# Wait for background init to finish before cleaning up (or cancel if still running)
		if not init_task.done():
			init_task.cancel()
			try:
				await init_task
			except asyncio.CancelledError:
				pass
		global gemini_client, gemini_pool
		pool = gemini_pool
		gemini_pool = None
		gemini_client = None
		if pool is not None:
			try:
				await pool.close()
			except Exception as e:
				logger.warning(f"Failed to close Gemini cookie pool during shutdown: {e}")


app = FastAPI(title="Gemini API FastAPI Server", lifespan=lifespan)

# 管理面板集成（必须在 app 初始化之后，admin.py 依赖 main 模块的配置和 app 实例）
from admin import router as admin_router, setup_middleware
app.include_router(admin_router)
setup_middleware(app)


def get_gemini_webapi_version() -> str:
	"""Return the installed gemini-webapi package version for runtime diagnostics."""
	try:
		return importlib.metadata.version("gemini-webapi")
	except importlib.metadata.PackageNotFoundError:
		return "unknown"


def get_cached_1psidts_path(psid: str) -> str:
	"""Return the cache path for a rotated 1PSIDTS value."""
	if not psid or not re.match("^[\\w\\-\\.]+$", psid):
		return ""
	return os.path.join(GEMINI_COOKIE_PATH, f".cached_1psidts_{psid}.txt")


def load_cached_1psidts(psid: str) -> str:
	"""Load a cached rotated 1PSIDTS value for the given 1PSID."""
	cached_file_path = get_cached_1psidts_path(psid)
	if not cached_file_path:
		return ""

	if os.path.exists(cached_file_path):
		try:
			content = Path(cached_file_path).read_text().strip()
			if content:
				return content
		except Exception as e:
			logger.warning(f"Error reading cache file {cached_file_path}: {e}")

	return ""


def get_cookie_value(cookies, name: str) -> str:
	"""Safely read a cookie value from an httpx cookie jar or mapping."""
	if not cookies:
		return ""

	for domain in (".google.com", ".googleusercontent.com", None):
		try:
			value = cookies.get(name, domain=domain) if domain is not None else cookies.get(name)
		except TypeError:
			value = cookies.get(name)
		except Exception:
			value = ""

		if value:
			return value

	return ""


# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Authentication credentials
SECURE_1PSID = os.environ.get("SECURE_1PSID", "")
SECURE_1PSIDTS = os.environ.get("SECURE_1PSIDTS", "")
API_KEY = os.environ.get("API_KEY", "")
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() == "true"
TEMPORARY_CHAT = os.environ.get("TEMPORARY_CHAT", "false").lower() == "true"
DISABLE_BUILTIN_MODELS = os.environ.get("DISABLE_BUILTIN_MODELS", "false").lower() == "true"
AUTO_DELETE_CHAT = os.environ.get("AUTO_DELETE_CHAT", "true").lower() == "true" and not TEMPORARY_CHAT
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# 多 Cookie 轮询调参：单账号并发上限（0=不限）、单次请求最多试几个号、失败冷却退避
COOKIE_MAX_CONCURRENCY = max(0, int(_env_number("COOKIE_MAX_CONCURRENCY", 0, int)))
COOKIE_MAX_RETRIES = max(0, int(_env_number("COOKIE_MAX_RETRIES", 0, int)))
COOKIE_COOLDOWN_BASE = max(1.0, float(_env_number("COOKIE_COOLDOWN_BASE", 15.0)))
COOKIE_COOLDOWN_MAX = max(COOKIE_COOLDOWN_BASE, float(_env_number("COOKIE_COOLDOWN_MAX", 300.0)))
COOKIE_VALIDATE_ON_INIT = os.environ.get("COOKIE_VALIDATE_ON_INIT", "true").lower() == "true"
COOKIE_EAGER_INIT = os.environ.get("COOKIE_EAGER_INIT", "false").lower() == "true"
SECRET_FILE_PATH = os.path.join(os.path.dirname(__file__), "secrets", "proxy_secret")
GEMINI_COOKIE_PATH = os.path.join(os.path.dirname(__file__), "secrets")
SESSION_VALIDATION_PROMPT = "Reply with exactly OK."
GEM_ID = os.environ.get("GEM_ID", "")
AUTH_FAILURE_TEXT_PATTERNS = (
	"are you signed in",
	"sign in",
	"signed in",
	"log in",
	"logged in",
)
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0"

os.environ.setdefault("GEMINI_COOKIE_PATH", GEMINI_COOKIE_PATH)


def normalize_custom_model_specs(raw_config) -> List[Dict[str, Dict[str, str]]]:
	"""Normalize custom model config from dict/list YAML or JSON."""
	if not raw_config:
		return []

	if isinstance(raw_config, dict):
		raw_config = raw_config.get("models", [raw_config])

	if not isinstance(raw_config, list):
		raise ValueError("Custom model config must be a list or a dict containing a 'models' list")

	normalized_models = []
	for idx, item in enumerate(raw_config, start=1):
		if not isinstance(item, dict):
			raise ValueError(f"Custom model entry #{idx} must be a mapping")

		model_name = str(item.get("model_name", "")).strip()
		model_header = item.get("model_header")
		if not model_name:
			raise ValueError(f"Custom model entry #{idx} is missing model_name")
		if not isinstance(model_header, dict):
			raise ValueError(f"Custom model '{model_name}' must define model_header as a mapping")

		normalized_models.append(
			{
				"model_name": model_name,
				"model_header": {str(key): str(value) for key, value in model_header.items()},
			}
		)

	return normalized_models


def load_custom_models() -> Dict[str, Dict[str, Dict[str, str]]]:
	"""Load custom model overrides from file/env on demand."""
	registry = {}
	custom_models_file = os.environ.get("CUSTOM_MODELS_FILE", "").strip()
	custom_models = os.environ.get("CUSTOM_MODELS", "").strip()

	if custom_models_file:
		config_path = Path(custom_models_file)
		if not config_path.is_absolute():
			config_path = Path(__file__).parent / config_path
		if not config_path.exists():
			raise ValueError(f"CUSTOM_MODELS_FILE does not exist: {config_path}")
		file_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
		for model in normalize_custom_model_specs(file_config):
			registry[model["model_name"].lower()] = model

	if custom_models:
		inline_config = yaml.safe_load(custom_models)
		for model in normalize_custom_model_specs(inline_config):
			registry[model["model_name"].lower()] = model

	return registry


def get_custom_model_registry() -> Dict[str, Dict[str, Dict[str, str]]]:
	"""Read the latest custom model config for each request."""
	return load_custom_models()


async def background_delete_chat(client: GeminiClient, cid: str):
	"""Deletes a chat conversation in the background to avoid blocking the main thread."""
	if not cid:
		return
	try:
		await client.delete_chat(cid)
	except Exception as e:
		logger.error(f"Failed to auto-delete chat {cid}: {e}")


def response_indicates_auth_failure(text: str) -> bool:
	"""Return True if the response text looks like a signed-out or degraded session."""
	normalized = (text or "").strip().lower()
	if not normalized:
		return True
	return any(pattern in normalized for pattern in AUTH_FAILURE_TEXT_PATTERNS)


async def fetch_readable_chat_response(client: GeminiClient, cid: str, retry_delays: List[int]) -> Optional[object]:
	"""Poll Gemini history until the chat becomes readable or retries are exhausted."""
	for attempt, delay in enumerate(retry_delays, start=1):
		try:
			if delay:
				await asyncio.sleep(delay)

			recovered = await client.fetch_latest_chat_response(cid)
			if recovered and getattr(recovered, "text", ""):
				return recovered
		except Exception as e:
			logger.exception(
				"Gemini history read failed for cid=%s on retry %s/%s after %ss delay: %s",
				cid,
				attempt,
				len(retry_delays),
				delay,
				e,
			)
			continue

	return None


async def background_verify_chat_persistence(client: GeminiClient, cid: str, source: str):
	"""Best-effort verification that a returned cid is readable from Gemini history."""
	if not cid:
		return

	retry_delays = [1, 3, 8]
	recovered = await fetch_readable_chat_response(client, cid, retry_delays)
	if recovered:
		logger.debug(
			"Gemini history verification succeeded: source=%s cid=%s text_len=%s metadata=%s",
			source,
			cid,
			len(recovered.text),
			getattr(recovered, "metadata", None),
		)
		return

	logger.warning(
		"Gemini history verification exhausted retries for cid=%s source=%s",
		cid,
		source,
	)


async def validate_gemini_client_session(client: GeminiClient, source: str):
	"""Verify that an initialized client can create and read back a normal persistent Gemini chat."""
	validation_cid = None
	try:
		response = await client.generate_content(SESSION_VALIDATION_PROMPT, temporary=False)
		response_text = getattr(response, "text", "") or ""
		metadata = getattr(response, "metadata", None) or []
		validation_cid = metadata[0] if metadata else None

		if response_indicates_auth_failure(response_text):
			raise ValueError("validation probe returned signed-out or empty content")

		if not validation_cid:
			raise ValueError("validation probe returned no persistent chat metadata")

		recovered = await fetch_readable_chat_response(client, validation_cid, [1, 3, 8])
		if not recovered or response_indicates_auth_failure(getattr(recovered, "text", "") or ""):
			raise ValueError("validation probe chat was not readable from Gemini history")

		logger.info("Gemini session validation succeeded using %s credentials", source)
	finally:
		if validation_cid:
			try:
				await client.delete_chat(validation_cid)
			except Exception:
				logger.debug("Failed to delete Gemini validation chat %s", validation_cid)


def load_or_generate_secret() -> str:
	"""
	Load the signature secret from file, or generate a new one if not found.
	"""
	if os.path.exists(SECRET_FILE_PATH):
		try:
			with open(SECRET_FILE_PATH, "r") as f:
				secret = f.read().strip()
				if secret:
					logger.info(f"Loaded proxy secret from {SECRET_FILE_PATH}")
					return secret
		except Exception as e:
			logger.warning(f"Failed to read secret file, trying to generate a new one: {e}")

	# Generate new secret if not found or error occurred
	new_secret = secrets.token_hex(32)
	try:
		# Ensure directory exists
		os.makedirs(os.path.dirname(SECRET_FILE_PATH), exist_ok=True)
		with open(SECRET_FILE_PATH, "w") as f:
			f.write(new_secret)

		# Set restrictive permissions (user-only readable/writable)
		try:
			os.chmod(SECRET_FILE_PATH, 0o600)
		except Exception as e:
			logger.warning(f"Failed to set restrictive permissions on {SECRET_FILE_PATH}: {e}")

		logger.info(f"Generated new proxy secret and saved to {SECRET_FILE_PATH}")
		return new_secret
	except Exception as e:
		logger.error(f"Error writing secret file: {e}")
		# if unable to save, return an in-memory ephemeral secret instead of using API_KEY or SECURE_1PSID
		ephemeral_secret = secrets.token_urlsafe(32)
		logger.warning("Using an in-memory secret to proxy images for this session.")
		return ephemeral_secret


SIGNATURE_SECRET = load_or_generate_secret()

# Watermark removal constants
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ALPHA_MAP_CACHE = {}


def get_alpha_map(size: int) -> np.ndarray:
	"""Load and cache the alpha map from the background capture image."""
	if size in ALPHA_MAP_CACHE:
		return ALPHA_MAP_CACHE[size]

	bg_path = os.path.join(ASSETS_DIR, f"bg_{size}.png")
	if not os.path.exists(bg_path):
		logger.warning(f"Watermark asset not found: {bg_path}")
		return None

	try:
		with Image.open(bg_path) as img:
			img_data = np.array(img.convert("RGB"))
			alpha_map = np.max(img_data, axis=2) / 255.0
			ALPHA_MAP_CACHE[size] = alpha_map
			return alpha_map
	except Exception as e:
		logger.error(f"Error loading alpha map {size}: {e}")
		return None


def remove_gemini_watermark(image_bytes: bytes) -> bytes:
	"""Remove Gemini watermark using Reverse Alpha Blending."""
	try:
		with Image.open(io.BytesIO(image_bytes)) as img:
			width, height = img.size
			orig_format = img.format

			if width > 1024 and height > 1024:
				logo_size, margin = 96, 64
			else:
				logo_size, margin = 48, 32

			alpha_map = get_alpha_map(logo_size)
			if alpha_map is None:
				return image_bytes

			x = width - margin - logo_size
			y = height - margin - logo_size
			if x < 0 or y < 0:
				logger.warning(f"Image too small for watermark removal: {width}x{height}")
				return image_bytes

			# Reverse Alpha Blending: original = (watermarked - α × 255) / (1 - α)
			img_array = np.array(img.convert("RGB")).astype(np.float64)
			roi = img_array[y : y + logo_size, x : x + logo_size].copy()

			alpha = np.clip(alpha_map, 0.002, 0.99)
			alpha_expanded = np.expand_dims(alpha, axis=2)
			cleaned_roi = (roi - alpha_expanded * 255.0) / (1.0 - alpha_expanded)
			cleaned_roi = np.clip(np.round(cleaned_roi), 0, 255).astype(np.uint8)

			img_array_uint8 = np.array(img.convert("RGB"))
			img_array_uint8[y : y + logo_size, x : x + logo_size] = cleaned_roi

			out_io = io.BytesIO()
			save_format = orig_format or "PNG"
			if save_format.upper() == "JPEG":
				Image.fromarray(img_array_uint8).save(out_io, format="JPEG", quality=95)
			else:
				Image.fromarray(img_array_uint8).save(out_io, format=save_format)
			return out_io.getvalue()

	except Exception as e:
		logger.error(f"Error removing watermark: {e}")
		return image_bytes


if not SECURE_1PSID or not SECURE_1PSIDTS:
	logger.warning("Gemini credentials are missing; set SECURE_1PSID and SECURE_1PSIDTS before serving requests.")
else:
	logger.info(
		"Startup config: thinking=%s temporary_chat=%s auto_delete_chat=%s public_base_url=%s gemini_webapi=%s",
		ENABLE_THINKING,
		TEMPORARY_CHAT,
		AUTO_DELETE_CHAT,
		bool(PUBLIC_BASE_URL),
		get_gemini_webapi_version(),
	)
	if not re.match("^[\\w\\-\\.]+$", SECURE_1PSID):
		logger.warning(
			"SECURE_1PSID contains characters outside the safe cache filename pattern. This may be valid for auth, but cached 1PSIDTS lookup will fall back to the env value."
		)

if not API_KEY:
	logger.info("API key authentication is disabled.")
else:
	logger.info("API key authentication is enabled.")


def correct_markdown(md_text: str) -> str:
	"""
	修正Markdown文本，移除Google搜索链接包装器，并根据显示文本简化目标URL。
	"""

	def simplify_link_target(text_content: str) -> str:
		match_colon_num = re.match(r"([^:]+:\d+)", text_content)
		if match_colon_num:
			return match_colon_num.group(1)
		return text_content

	def replacer(match: re.Match) -> str:
		outer_open_paren = match.group(1)
		display_text = match.group(2)

		new_target_url = simplify_link_target(display_text)
		new_link_segment = f"[`{display_text}`]({new_target_url})"

		if outer_open_paren:
			return f"{outer_open_paren}{new_link_segment})"
		else:
			return new_link_segment

	pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"

	fixed_google_links = re.sub(pattern, replacer, md_text)
	# fix wrapped markdownlink
	pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
	return re.sub(pattern, r"\1", fixed_google_links)


# Pydantic models for API requests and responses
class ContentItem(BaseModel):
	type: str
	text: Optional[str] = None
	image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
	role: str
	content: Union[str, List[ContentItem], None] = None
	name: Optional[str] = None
	# tools 回放用：assistant 的调用记录、tool 角色对应的调用 id
	tool_calls: Optional[List[Dict[str, Any]]] = None
	tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Message]
	temperature: Optional[float] = 0.7
	top_p: Optional[float] = 1.0
	n: Optional[int] = 1
	stream: Optional[bool] = False
	max_tokens: Optional[int] = None
	presence_penalty: Optional[float] = 0
	frequency_penalty: Optional[float] = 0
	user: Optional[str] = None
	# OpenAI tools / function calling（提示词模拟，见 gemini_tools.py）
	tools: Optional[List[Dict[str, Any]]] = None
	tool_choice: Optional[Union[str, Dict[str, Any]]] = None
	# 思考档走 OpenAI 规范字段 reasoning_effort（low / medium / high），
	# 另外兼容几个常见写法，免得客户端要改代码。
	reasoning_effort: Optional[str] = None
	thinking_effort: Optional[str] = None
	think_effort: Optional[str] = None


class Choice(BaseModel):
	index: int
	message: Message
	finish_reason: str


class Usage(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int


class ChatCompletionResponse(BaseModel):
	id: str
	object: str = "chat.completion"
	created: int
	model: str
	choices: List[Choice]
	usage: Usage


class ModelData(BaseModel):
	id: str
	object: str = "model"
	created: int
	owned_by: str = "google"


class ModelList(BaseModel):
	object: str = "list"
	data: List[ModelData]


# Authentication dependency
async def verify_api_key(authorization: str = Header(None)):
	"""
	Verify the API key extracted from the Authorization header.
	Also accepts valid admin session tokens for the quick test chat.
	"""
	if not API_KEY:
		# If API_KEY is not set in environment, skip validation (for development)
		return

	if not authorization:
		raise HTTPException(status_code=401, detail="Missing Authorization header")

	# Accept valid admin session tokens (for admin panel quick test)
	if authorization.startswith("Bearer "):
		token = authorization[7:]
		from admin import _admin_sessions, _clean_expired_sessions
		_clean_expired_sessions()
		if token in _admin_sessions:
			return

	try:
		scheme, token = authorization.split()
		if scheme.lower() != "bearer":
			raise HTTPException(
				status_code=401,
				detail="Invalid authentication scheme. Use Bearer token",
			)

		if token != API_KEY:
			raise HTTPException(status_code=401, detail="Invalid API key")
	except ValueError:
		raise HTTPException(
			status_code=401,
			detail="Invalid authorization format. Use 'Bearer YOUR_API_KEY'",
		)

	return token


# Simple error handler middleware
@app.middleware("http")
async def error_handling(request: Request, call_next):
	"""
	Global middleware to catch unhandled exceptions, log the error,
	and return a standardized HTTP 500 response.
	"""
	try:
		return await call_next(request)
	except Exception:
		logger.exception("Request failed")
		return JSONResponse(
			status_code=500,
			content={
				"error": {
					"message": "Internal server error",
					"type": "internal_server_error",
				}
			},
		)


# Get list of available models
@app.get("/v1/models")
async def list_models():
	"""返回内置模型目录（含 Gemini 4 系列）与自定义模型。

	只列路由名；思考档由请求里的 reasoning_effort（low/medium/high）决定。
	"""
	now = int(datetime.now(tz=timezone.utc).timestamp())
	data = []
	seen = set()

	def add(model_id: str):
		key = model_id.lower()
		if key in seen:
			return
		seen.add(key)
		data.append(
			{
				"id": model_id,
				"object": "model",
				"created": now,
				"owned_by": "google-gemini-web",
			}
		)

	for custom_model in get_custom_model_registry().values():
		add(custom_model["model_name"])

	# 禁用内置模型时，仅返回自定义模型
	if not DISABLE_BUILTIN_MODELS:
		# 只暴露路由名，思考档交给 reasoning_effort 参数
		for spec in BUILTIN_MODELS.values():
			if spec.retired:
				continue
			add(spec.name)

	return {"object": "list", "data": data}


def resolve_model_spec(requested_model: str) -> Optional[ModelSpec]:
	"""把请求里的模型名解析成模型规格（自定义模型优先，其次内置目录）。

	认不出确切名字时按关键词兜底（flash/hydrogen → 4 Hydrogen，neon → 4 Neon，pro/carbon → 4 Carbon），
	保证老客户端写的 ``gemini-3.0-flash`` 这类名字也不会整条链路 500。
	"""
	name = normalize_model_name(requested_model)
	if not name:
		return None

	custom_model_registry = get_custom_model_registry()
	for key, custom_model in custom_model_registry.items():
		if key != name:
			# 只接受「自定义名 + 合法档位后缀」，避免 gemini-3.5-flash 误吃掉 -lite
			suffix = name[len(key) :] if name.startswith(key) else None
			if not suffix or suffix not in TIER_SUFFIXES:
				continue
		header_value = custom_model["model_header"].get(MODEL_HEADER_KEY, "")
		spec = parse_header_spec(header_value, custom_model["model_name"])
		if spec is not None:
			# 自定义模型名带档位后缀时，用后缀覆盖思考档
			tier = _tier_from_name(name, key)
			if tier:
				spec = spec.with_tier(tier, "")
			if name.endswith("-task"):
				# 自定义模型走 parse_header_spec 时拿不到 feature，名字以 -task 结尾就补上
				spec = spec.with_feature(FEATURE_TASK)
			return spec

	return resolve_builtin(name) or fuzzy_builtin(name)


def _tier_from_name(full_name: str, base_name: str) -> int:
	"""从 ``<base>-thinking-medium`` 这类名字里取思考档。"""
	if not full_name.startswith(base_name):
		return 0
	return TIER_SUFFIXES.get(full_name[len(base_name) :], 0)


def _supports_extended_thinking(client) -> bool:
	"""gemini-webapi 2.1+ 才有 extended_thinking 参数；2.0.0 只认模型头，交给改写层处理。"""
	import inspect

	try:
		params = inspect.signature(client.generate_content).parameters
	except (TypeError, ValueError):
		return False
	return "extended_thinking" in params


def get_effective_model_debug_info(spec: ModelSpec) -> Dict[str, str]:
	"""请求日志用：解析出来的模型名与线上选择参数。"""
	return {
		"model_name": spec.name,
		"hex_id": spec.hex_id,
		"mode": spec.mode,
		"think": spec.think_wire,
		"capacity": spec.capacity,
	}


# Prepare conversation history from OpenAI messages format
def prepare_conversation(
	messages: List[Message],
	skip_system: bool = False,
	tools: Optional[List[Dict[str, Any]]] = None,
	tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> tuple:
	"""
	Convert a list of OpenAI-formatted message objects into a
	flat string conversation format suitable for the Gemini API.
	Also extracts and saves base64 images to temporary files.

	Args:
		skip_system: 当启用 Gemini Gem 时传 True，跳过 system 消息拼接，
			避免与 Gem 的 system prompt 重复冲突。
		tools / tool_choice: OpenAI tools 定义；Gemini 网页端没有协议层工具调用，
			这里把定义拼成指令让模型吐 ```tool_call``` 围栏（见 gemini_tools.py）。

	Returns:
		A tuple containing the constructed conversation string and a list of paths to temporary image files.
	"""
	parts: List[str] = []
	temp_files: List[str] = []

	# 工具格式指令放最前，末尾还会再锚一次（agentic 客户端的长提示会把顶部指令冲掉）
	tools_top, tools_bottom = build_tools_instruction(tools, tool_choice)
	if tools_top:
		parts.append(tools_top)

	for msg in messages:
		# 启用 Gem 时跳过 system 消息（Gem 本身就是 system prompt 通道）
		if skip_system and msg.role == "system":
			continue

		# tool 角色：工具执行结果回放，给模型一个清爽的成功/失败信号
		if msg.role == "tool":
			parts.append(format_tool_result(msg.name or "", _content_to_text(msg.content)))
			continue

		text, images = _message_content_to_text(msg.content, temp_files)
		temp_files.extend(images)

		if msg.role == "system":
			parts.append(f"System: {text}")
		elif msg.role == "assistant":
			block = f"Assistant: {text}"
			# assistant 的调用记录渲染回围栏，让模型看到自己调过什么
			fences = _tool_calls_to_fences(msg.tool_calls)
			if fences:
				block += "\n" + fences
			parts.append(block)
		else:
			parts.append(f"Human: {text}")

	if tools_bottom:
		parts.append(tools_bottom)

	conversation = "\n\n".join(part for part in parts if part)

	# Add a final prompt for the assistant to respond to
	if conversation:
		conversation += "\n\n"
	conversation += "Assistant: "

	return conversation, temp_files


def _content_to_text(content: Union[str, List[ContentItem], None]) -> str:
	"""把 OpenAI content（字符串或内容数组）压成纯文本，忽略图片部分。"""
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	bits = []
	for item in content:
		if item.type == "text" and item.text:
			bits.append(item.text)
	return "\n".join(bits)


def _message_content_to_text(content: Union[str, List[ContentItem], None], temp_files: List[str]) -> tuple:
	"""文本 + 图片分开处理：文本进 prompt，base64 图片落成临时文件交给上游上传。"""
	if content is None:
		return "", []
	if isinstance(content, str):
		return content, []

	text_bits: List[str] = []
	images: List[str] = []
	for item in content:
		if item.type == "text":
			text_bits.append(item.text or "")
		elif item.type == "image_url" and item.image_url:
			image_url = item.image_url.get("url", "")
			if image_url.startswith("data:image/"):
				try:
					image_data = base64.b64decode(image_url.split(",")[1])
					with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
						tmp.write(image_data)
						images.append(tmp.name)
				except Exception as e:
					logger.error(f"Error processing base64 image: {str(e)}")
	return "\n".join(text_bits), images


def _tool_calls_to_fences(tool_calls: Optional[List[Dict[str, Any]]]) -> str:
	"""把 assistant 的历史 tool_calls 渲染成 ```tool_call``` 围栏文本。"""
	if not tool_calls:
		return ""
	fences = []
	for call in tool_calls:
		if not isinstance(call, dict):
			continue
		function = call.get("function") if isinstance(call.get("function"), dict) else {}
		name = (function.get("name") or "").strip()
		if not name:
			continue
		arguments = function.get("arguments") or "{}"
		fences.append(f'{TOOL_FENCE_OPEN}\n{{"name": "{name}", "arguments": {arguments}}}\n{TOOL_FENCE_CLOSE}')
	return "\n".join(fences)


async def get_gemini_pool() -> CookiePool:
	"""懒初始化多 Cookie 轮询池（首次用到才解析配置）。"""
	global gemini_pool
	if gemini_pool is not None:
		return gemini_pool

	async with gemini_pool_lock:
		if gemini_pool is not None:
			return gemini_pool

		# 传运行时变量而不是只读 os.environ：管理面板保存 Cookie 后能立刻生效
		entries = load_cookie_entries(single=(SECURE_1PSID, SECURE_1PSIDTS))
		if not entries:
			raise HTTPException(
				status_code=500,
				detail="Missing SECURE_1PSID/SECURE_1PSIDTS. 单账号填这两个，多账号用 GEMINI_COOKIES 或 COOKIES_FILE。",
			)

		gemini_pool = CookiePool(
			entries,
			install_shim=install_wire_shim,
			validator=validate_gemini_client_session if COOKIE_VALIDATE_ON_INIT else None,
			psidts_cache=load_cached_1psidts,
			extra_cookies=load_extra_cookies(),
			max_concurrency=COOKIE_MAX_CONCURRENCY,
			cooldown_base=COOKIE_COOLDOWN_BASE,
			cooldown_max=COOKIE_COOLDOWN_MAX,
		)
		logger.info(
			"Cookie 池已装载 %s 个账号：%s",
			len(entries),
			", ".join(entry.name for entry in entries),
		)
		return gemini_pool


async def acquire_lease(pool: CookiePool, tried: Optional[set] = None) -> CookieLease:
	"""从池里借一个账号；池子彻底没号时回 503，让客户端稍后重试。"""
	try:
		return await pool.acquire(exclude=tried)
	except CookiePoolUnavailable as e:
		raise HTTPException(status_code=503, detail=f"Gemini 账号全部不可用：{e}")


async def generate_with_cookie_pool(pool: CookiePool, conversation: str, gen_kwargs: Dict[str, Any], model_spec: ModelSpec) -> tuple:
	"""非流式生成：借账号 → 失败换号重试 → 返回 (response, client)。"""
	budget = COOKIE_MAX_RETRIES or pool.retry_budget
	attempts = max(1, min(budget, pool.retry_budget))
	tried: set = set()
	last_error: Optional[BaseException] = None

	for attempt in range(1, attempts + 1):
		lease = await acquire_lease(pool, tried)
		tried.add(lease.psid)
		try:
			if model_spec.feature:
				await ensure_task_ready(lease.client)
			with active_model_spec(model_spec):
				response = await lease.client.generate_content(conversation, **gen_kwargs)
			lease.succeed()
			return response, lease.client
		except Exception as e:
			last_error = e
			lease.fail(e)
			logger.warning(
				"账号 %s 生成失败（%s/%s），换号重试：%s",
				lease.label,
				attempt,
				attempts,
				e,
			)
		finally:
			lease.release()

	if last_error is not None:
		raise last_error
	raise RuntimeError("cookie pool exhausted without response")


LAST_QUOTA: Optional[Dict[str, Any]] = None


async def fetch_account_email(client) -> str:
	"""从 Gemini 网页引导页里抠出登录账号邮箱（拿不到就返回空串）。"""
	try:
		response = await client.client.get("https://gemini.google.com/app")
		text = (getattr(response, "text", "") or "")
	except Exception:
		return ""

	for candidate in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
		lowered = candidate.lower()
		if lowered.endswith("@google.com"):
			continue
		return candidate
	return ""


async def fetch_live_quota() -> Optional[Dict[str, Any]]:
	"""用临时客户端拉一次账号额度/套餐（池里的客户端状态可能不干净，单独建一个最稳）。"""
	global LAST_QUOTA

	if not SECURE_1PSID or not SECURE_1PSIDTS:
		return None

	from cookie_pool import load_extra_cookies, mask_email, quota_view

	client = GeminiClient(SECURE_1PSID, SECURE_1PSIDTS)
	try:
		extras = load_extra_cookies()
		if extras:
			client.cookies = extras
		await client.init(timeout=30, auto_close=False, auto_refresh=False)
		for method_name in ("_fetch_quota", "_fetch_usage_info"):
			method = getattr(client, method_name, None)
			if method is None:
				continue
			try:
				await method()
			except Exception:
				pass
		view = quota_view(client) or {}
		email = await fetch_account_email(client)
		if email:
			view["email"] = email
			view["email_masked"] = mask_email(email)
		if view:
			LAST_QUOTA = view
			return view
		return None
	except Exception as e:
		logger.warning("拉取账号额度失败：%s", e)
		return None
	finally:
		try:
			await client.close()
		except Exception:
			pass


async def reset_gemini_pool() -> None:
	"""关掉池里所有客户端并丢弃，下次请求按最新配置重建。"""
	global gemini_pool, gemini_client
	pool = gemini_pool
	gemini_pool = None
	gemini_client = None
	if pool is not None:
		await pool.close()
	logger.info("Cookie 池已重置")


async def get_gemini_client():
	"""
	Get an initialized GeminiClient (whichever cookie pool account is idlest).

	这个入口保留给管理面板 / Gem 管理 / 图片代理这类旁路调用；对话请求走
	acquire_lease()，借出的账号会真正计入负载并参与轮询。

	Raises:
		HTTPException: If initialization fails due to invalid parameters or connection issues.
	"""
	global gemini_client
	if gemini_client is not None:
		return gemini_client

	pool = await get_gemini_pool()
	lease = await acquire_lease(pool)
	try:
		gemini_client = lease.client
		return gemini_client
	finally:
		lease.release()


def get_image_signature(url: str) -> str:
	"""
	Generate a HMAC-SHA256 signature for the image URL using the persistent SIGNATURE_SECRET.
	"""
	secret = SIGNATURE_SECRET.encode()
	return hmac.new(secret, url.encode(), hashlib.sha256).hexdigest()


def postprocess_text(text: str) -> str:
	"""Apply text cleanup and markdown corrections to response text."""
	text = text.replace("&lt;", "<").replace("\\<", "<").replace("\\_", "_").replace("\\>", ">")
	return correct_markdown(text)


def extract_image_markdown(response, base_url: str) -> str:
	"""Extract images from a response and return markdown image links."""
	result = ""
	if hasattr(response, "images") and response.images:
		for img in response.images:
			img_url = getattr(img, "url", None)
			if img_url:
				sig = get_image_signature(img_url)
				proxy_url = f"{base_url}/gemini-proxy/image?url={quote(img_url)}&sig={sig}"
				result += f"\n\n![🎨 Loading image...]({proxy_url})"
	return result


@app.post("/v1/chat/completions")
async def create_chat_completion(
	request: ChatCompletionRequest,
	raw_request: Request,
	api_key: str = Depends(verify_api_key),
):
	"""
	Handle chat completion requests, translating from OpenAI API format to Gemini API format.
	Supports both streaming and non-streaming responses, caching, thinking features,
	and background conversation cleanup based on configuration.
	"""
	try:
		# 多 Cookie 轮询：账号按请求从池里借，不再全局共用一份

		# 转换消息为对话格式（启用 Gem 时跳过 system 消息）
		conversation, temp_files = prepare_conversation(
			request.messages,
			skip_system=bool(GEM_ID),
			tools=request.tools,
			tool_choice=request.tool_choice,
		)
		tools_active = bool(build_tools_instruction(request.tools, request.tool_choice)[0])
		logger.info(
			"Chat completion request: stream=%s requested_model=%s messages=%s temp_files=%s tools=%s",
			request.stream,
			request.model,
			len(request.messages),
			len(temp_files),
			len(request.tools or []) if tools_active else 0,
		)

		# 解析模型（自定义模型优先，其次内置目录；Gemini 4 系列在这里落到 hex/档位/思考档）
		model_spec = resolve_model_spec(request.model)
		if model_spec is None:
			raise HTTPException(
				status_code=400,
				detail=f"Unknown model: {request.model}. Try gemini-4-hydrogen, gemini-4-neon, gemini-4-carbon or gemini-3.5-flash-lite.",
			)
		if model_spec.retired:
			raise HTTPException(status_code=400, detail=model_spec.retired)

		# 思考档由客户端参数决定（reasoning_effort: low/medium/high），不再靠模型名后缀
		effort_think = think_from_effort(request.reasoning_effort, request.thinking_effort, request.think_effort)
		if effort_think:
			model_spec = model_spec.with_tier(effort_think, "")

		model_debug_info = get_effective_model_debug_info(model_spec)
		logger.info(
			"Resolved Gemini model: requested=%s effective=%s hex=%s mode=%s think=%s",
			request.model,
			model_debug_info["model_name"],
			model_debug_info["hex_id"],
			model_debug_info["mode"],
			model_debug_info["think"],
		)

		# 创建响应对象
		completion_id = f"chatcmpl-{uuid.uuid4()}"
		created_time = int(time.time())
		base_url = PUBLIC_BASE_URL or str(raw_request.base_url).rstrip("/")

		# Prepare generate_content arguments
		# 模型以 dict 形式交给 gemini_webapi：hex / 档位写进模型头，
		# 思考档与 4 系列专用槽位由 gemini_wire 在发请求前统一改写。
		gen_kwargs = {"model": build_model_arg(model_spec)}
		if TEMPORARY_CHAT and not model_spec.feature:
			# Task 只能用普通会话；临时会话会拿不到任务结果
			gen_kwargs["temporary"] = True
		if temp_files:
			gen_kwargs["files"] = temp_files
		if GEM_ID:
			gen_kwargs["gem"] = GEM_ID
		if model_spec.extended_thinking and _supports_extended_thinking(GeminiClient):
			gen_kwargs["extended_thinking"] = True

		# 多 Cookie 轮询：流式要占住一个账号，非流式要能失败换号，所以借号分别放在各自分支里
		pool = await get_gemini_pool()

		if request.stream:
			lease = await acquire_lease(pool)
			gemini_client = lease.client
			# Real streaming using upstream generate_content_stream
			async def generate_stream():
				# 流式生成在 StreamingResponse 自己的任务里跑，模型规格必须在这里挂上
				spec_token = gemini_wire_push_spec(model_spec)
				stream_failed = False
				if model_spec.feature:
					await ensure_task_ready(gemini_client)
				try:

					def make_chunk(delta: dict, finish_reason=None):
						return (
							"data: "
							+ json.dumps(
								{
									"id": completion_id,
									"object": "chat.completion.chunk",
									"created": created_time,
									"model": request.model,
									"choices": [
										{
											"index": 0,
											"delta": delta,
											"finish_reason": finish_reason,
										}
									],
								}
							)
							+ "\n\n"
						)

					# Send initial role chunk
					yield make_chunk({"role": "assistant"})

					thinking_started = False
					thinking_ended = False
					yielded_images = 0
					text_buffer = ""
					full_text = ""
					# 带 tools 时用围栏闸门：围栏内容全部扣住，收尾统一转成 tool_calls
					gate_output: List[str] = []
					gate = ToolFenceGate(gate_output.append) if tools_active else None
					captured_cid = None
					chunk_count = 0
					last_metadata = None

					async for chunk in gemini_client.generate_content_stream(conversation, **gen_kwargs):
						chunk_count += 1
						if hasattr(chunk, "metadata") and chunk.metadata:
							last_metadata = chunk.metadata
						# Capture conversation ID for auto-deletion
						if AUTO_DELETE_CHAT and captured_cid is None and hasattr(chunk, "metadata") and chunk.metadata and len(chunk.metadata) > 0:
							captured_cid = chunk.metadata[0]

						# Handle thinking/thoughts delta
						if ENABLE_THINKING and hasattr(chunk, "thoughts_delta") and chunk.thoughts_delta:
							if not thinking_started:
								yield make_chunk({"content": "<think>\n"})
								thinking_started = True

							# Also include reasoning_content for full Open WebUI native compatibility
							yield make_chunk(
								{
									"content": chunk.thoughts_delta,
									"reasoning_content": chunk.thoughts_delta,
								}
							)

						# Handle text delta
						if hasattr(chunk, "text_delta") and chunk.text_delta:
							# Close thinking tag before first text content
							if thinking_started and not thinking_ended:
								thinking_ended = True
								yield make_chunk({"content": "\n</think>\n\n"})

							full_text += chunk.text_delta

							if gate is not None:
								# 只放行确定不在围栏里的部分，围栏等收尾统一解析
								gate.push(chunk.text_delta)
								for piece in gate_output:
									yield make_chunk({"content": postprocess_text(piece)})
								gate_output.clear()
							else:
								text_buffer += chunk.text_delta
								safe_to_yield = False

								# Yield if buffer ends with whitespace and looks like it's outside a markdown link
								if (
									text_buffer[-1].isspace()
									and text_buffer.count("[") == text_buffer.count("]")
									and text_buffer.count("(") == text_buffer.count(")")
								):
									safe_to_yield = True
								elif len(text_buffer) > 500:
									safe_to_yield = True

								if safe_to_yield:
									yield make_chunk({"content": postprocess_text(text_buffer)})
									text_buffer = ""

						# Handle inline images as they arrive
						if hasattr(chunk, "images") and chunk.images and len(chunk.images) > yielded_images:
							# Close thinking tag if an image arrives before any text
							if thinking_started and not thinking_ended:
								thinking_ended = True
								yield make_chunk({"content": "\n</think>\n\n"})

							new_images = chunk.images[yielded_images:]
							for img in new_images:
								img_url = getattr(img, "url", None)
								if img_url:
									sig = get_image_signature(img_url)
									proxy_url = f"{base_url}/gemini-proxy/image?url={quote(img_url)}&sig={sig}"
									img_md = f"\n\n![🎨 Loading image...]({proxy_url})\n\n"
									yield make_chunk({"content": img_md})
							yielded_images = len(chunk.images)

					# Flush any remaining text
					stream_tool_calls: List[Dict[str, Any]] = []
					if gate is not None:
						# 围栏剥掉后的正文才是最终文本；尾巴按「已发内容」对齐补发
						clean_text, stream_tool_calls = parse_tool_calls(full_text)
						tail = gate.remaining(clean_text)
						if tail:
							yield make_chunk({"content": postprocess_text(tail)})
						if not clean_text.strip() and not stream_tool_calls:
							# 只有一个解析不了的围栏：不能静默回空，客户端会看不出失败
							logger.warning("Streaming reply had only an unparseable tool_call block")
							yield make_chunk({"content": "\n\n[上游只返回了一个解析不了的 tool_call 块，请重试]"})
					elif text_buffer:
						yield make_chunk({"content": postprocess_text(text_buffer)})

					# Close thinking tag if it was never closed
					if thinking_started and not thinking_ended:
						yield make_chunk({"content": "\n</think>\n\n"})

					# Task 模式：流可能以「运行中」结束、正文稍后才写进会话，这里补读一次
					if model_spec.feature and not full_text.strip() and last_metadata:
						recovered = await fetch_readable_chat_response(gemini_client, last_metadata[0], [2, 5, 10])
						if recovered and getattr(recovered, "text", ""):
							yield make_chunk({"content": postprocess_text(recovered.text)})

					# Send finish chunk（有工具调用时 finish_reason 必须是 tool_calls）
					if stream_tool_calls:
						yield make_chunk({"tool_calls": tool_calls_for_stream(stream_tool_calls)}, finish_reason="tool_calls")
					else:
						yield make_chunk({}, finish_reason="stop")
					yield "data: [DONE]\n\n"

					logger.info(
						"Streaming response completed: chunks=%s images=%s",
						chunk_count,
						yielded_images,
					)
					if last_metadata and len(last_metadata) > 0 and not AUTO_DELETE_CHAT:
						asyncio.create_task(background_verify_chat_persistence(gemini_client, last_metadata[0], "stream"))
				except Exception as e:
					logger.error(f"Error during streaming: {str(e)}", exc_info=True)
					# 这个账号这次失败了：进冷却，下次请求换号
					stream_failed = True
					lease.fail(e)
					# Send error as a content chunk so the client sees it
					error_msg = "\n\n[An internal error occurred while streaming]"
					yield make_chunk({"content": error_msg})
					yield make_chunk({}, finish_reason="stop")
					yield "data: [DONE]\n\n"
				finally:
					gemini_wire_pop_spec(spec_token)

					# 账号用完即归还：占位归零；这一轮没报错就记一次成功（清零失败退避）
					if not stream_failed:
						lease.succeed()
					lease.release()

					# Create background task to delete the chat if AUTO_DELETE_CHAT is enabled
					if AUTO_DELETE_CHAT and captured_cid:
						asyncio.create_task(background_delete_chat(gemini_client, captured_cid))

					# 清理临时文件
					for temp_file in temp_files:
						try:
							os.unlink(temp_file)
						except Exception as e:
							logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")

			return StreamingResponse(generate_stream(), media_type="text/event-stream")
		else:
			# Non-streaming response：借号 / 换号重试 / 归还都在 generate_with_cookie_pool 里
			try:
				response, gemini_client = await generate_with_cookie_pool(pool, conversation, gen_kwargs, model_spec)

				if AUTO_DELETE_CHAT and hasattr(response, "metadata") and response.metadata and len(response.metadata) > 0:
					cid = response.metadata[0]
					asyncio.create_task(background_delete_chat(gemini_client, cid))
				elif hasattr(response, "metadata") and response.metadata and len(response.metadata) > 0:
					asyncio.create_task(background_verify_chat_persistence(gemini_client, response.metadata[0], "non-stream"))
				elif not getattr(response, "metadata", None):
					logger.warning("Non-stream response returned no Gemini metadata. This request may not map to a persistent Gemini chat.")

			finally:
				# 清理临时文件
				for temp_file in temp_files:
					try:
						os.unlink(temp_file)
					except Exception as e:
						logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")

			# 提取文本响应
			thoughts_text = ""
			if ENABLE_THINKING and hasattr(response, "thoughts") and response.thoughts:
				thoughts_text = f"<think>\n{response.thoughts}\n</think>\n\n"
			raw_text = getattr(response, "text", None) or str(response)

			# Task 模式：首轮可能只回「运行中」，正文稍后才会话里可见，补读一次
			if model_spec.feature and not raw_text.strip():
				task_cid = response.metadata[0] if getattr(response, "metadata", None) else None
				if task_cid:
					recovered = await fetch_readable_chat_response(gemini_client, task_cid, [2, 5, 10])
					if recovered and getattr(recovered, "text", ""):
						raw_text = recovered.text

			# tools：把 ```tool_call``` 围栏解析成 OpenAI 的 tool_calls
			tool_calls: List[Dict[str, Any]] = []
			if tools_active:
				clean_text, tool_calls = parse_tool_calls(raw_text)
				if not clean_text.strip() and not tool_calls:
					# 只有一个解析不了的围栏（JSON 写坏 / 没有工具名）：必须报错，
					# 不能回 200 + content:null 且不带 tool_calls，客户端看不出失败
					logger.warning("Non-stream reply had only an unparseable tool_call block")
					raise HTTPException(
						status_code=502,
						detail="upstream reply had only an unparseable tool_call block; please retry",
					)
				raw_text = clean_text

			reply_text = thoughts_text + raw_text
			# 提取并追加图片响应
			reply_text += extract_image_markdown(response, base_url)
			reply_text = postprocess_text(reply_text)

			if not tool_calls and (not reply_text or reply_text.strip() == ""):
				logger.warning("Empty response received from Gemini")
				reply_text = "Server returned an empty response. Please check that Gemini API credentials are valid."

			message: Dict[str, Any] = {
				"role": "assistant",
				"content": reply_text if reply_text.strip() else None,
			}
			if tool_calls:
				message["tool_calls"] = tool_calls

			result = {
				"id": completion_id,
				"object": "chat.completion",
				"created": created_time,
				"model": request.model,
				"choices": [
					{
						"index": 0,
						"message": message,
						"finish_reason": "tool_calls" if tool_calls else "stop",
					}
				],
				"usage": {
					"prompt_tokens": len(conversation.split()),
					"completion_tokens": len(reply_text.split()),
					"total_tokens": len(conversation.split()) + len(reply_text.split()),
				},
			}

			logger.info("Non-streaming response completed: tool_calls=%s", len(tool_calls))
			return result

	except HTTPException:
		# 模型名不认识 / 已下线这类 4xx 直接透出，不要裹成 500
		raise
	except Exception as e:
		logger.error(f"Error generating completion: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")


@app.get("/gemini-proxy/image")
async def proxy_image(url: str, sig: str):
	"""
	Proxy images from Google domains to bypass browser security policies.
	Requires a valid HMAC signature.
	"""
	# Verify signature
	expected_sig = get_image_signature(url)
	if not hmac.compare_digest(sig, expected_sig):
		logger.warning(f"Invalid signature for proxy request: {url}")
		raise HTTPException(status_code=403, detail="Invalid signature")

	# Prevent open proxying
	allowed_domains = ["google.com", "googleusercontent.com", "gstatic.com"]

	try:
		parsed = urlparse(url)
		if parsed.scheme not in ["http", "https"]:
			logger.warning(f"Invalid scheme in proxy request: {parsed.scheme}")
			raise HTTPException(status_code=400, detail="Invalid URL scheme")

		hostname = parsed.hostname
		if not hostname:
			logger.warning(f"No hostname in proxy request: {url}")
			raise HTTPException(status_code=400, detail="Invalid URL")

		hostname = hostname.lower()
		is_allowed = any(hostname == d or hostname.endswith("." + d) for d in allowed_domains)

		if not is_allowed:
			logger.warning(f"Blocked proxy request for domain: {hostname}")
			raise HTTPException(status_code=403, detail="Domain not allowed")
	except ValueError:
		logger.warning(f"Malformed URL in proxy request: {url}")
		raise HTTPException(status_code=400, detail="Invalid URL")

	# Minimal browser-like headers
	headers = {
		"User-Agent": DEFAULT_USER_AGENT,
		"Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.9",
		"Referer": "https://gemini.google.com/",
	}

	# 10MB limit
	MAX_BYTES = 10 * 1024 * 1024

	# Use scoped cookies to prevent leakage during redirects
	jar = httpx.Cookies()

	# 多 Cookie：优先用池里现成的账号（主账号可能已经掉线），拿不到再退回环境变量
	cookie_client = gemini_pool.primary_client() if gemini_pool is not None else None
	cookie_client = cookie_client or gemini_client
	psid = get_cookie_value(getattr(cookie_client, "cookies", None), "__Secure-1PSID") or SECURE_1PSID
	psidts = get_cookie_value(getattr(cookie_client, "cookies", None), "__Secure-1PSIDTS") or load_cached_1psidts(psid) or SECURE_1PSIDTS

	jar.set("__Secure-1PSID", psid, domain=".google.com")
	jar.set("__Secure-1PSIDTS", psidts, domain=".google.com")
	jar.set("__Secure-1PSID", psid, domain=".googleusercontent.com")
	jar.set("__Secure-1PSIDTS", psidts, domain=".googleusercontent.com")

	async with httpx.AsyncClient(http2=True, cookies=jar, follow_redirects=True) as client:
		try:
			# Fetch original resolution to keep watermark at expected size/position
			fetch_url = re.sub(r"=s\d+$", "=s0", url) if re.search(r"=s\d+$", url) else url + "=s0"

			async with client.stream("GET", fetch_url, timeout=15.0, headers=headers) as resp:
				if resp.status_code != 200:
					logger.error(f"Google returned {resp.status_code} for image: {url}")

				resp.raise_for_status()

				content = bytearray()
				async for chunk in resp.aiter_bytes():
					content.extend(chunk)
					if len(content) > MAX_BYTES:
						logger.warning(f"Image too large: {url} (exceeded {MAX_BYTES} bytes)")
						raise HTTPException(status_code=413, detail="Image too large")
				# Validate Content-Type to prevent XSS/MIME sniffing
				upstream_content_type = resp.headers.get("content-type", "image/png").lower()
				if not upstream_content_type.startswith("image/"):
					logger.warning(f"Rejected non-image Content-Type: {upstream_content_type} for {url}")
					media_type = "image/png"
				else:
					media_type = upstream_content_type

				# Process watermark removal
				if media_type in ["image/png", "image/jpeg", "image/webp"]:
					processed_content = remove_gemini_watermark(bytes(content))
				else:
					processed_content = bytes(content)

				return Response(
					content=processed_content,
					media_type=media_type,
					headers={
						"Cross-Origin-Resource-Policy": "cross-origin",
						"Access-Control-Allow-Origin": "*",
						"Cache-Control": "public, max-age=86400",  # Cache for 24 hours
						"X-Content-Type-Options": "nosniff",
					},
				)
		except httpx.HTTPStatusError as e:
			logger.error(f"Failed to fetch image: {e.response.status_code} for {url}")
			raise HTTPException(
				status_code=e.response.status_code,
				detail=f"Failed to fetch image: Google returned {e.response.status_code}",
			)
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Proxy error: {str(e)}")
			raise HTTPException(status_code=500, detail="Internal proxy error")


@app.get("/")
async def root():
	"""根路径直接跳转到管理面板。"""
	return RedirectResponse(url="/admin", status_code=307)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
