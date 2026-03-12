import asyncio
import base64
import hmac
import hashlib
import io
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from urllib.parse import quote, urlparse

import httpx
import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import GeminiClient, set_log_level
from gemini_webapi.constants import Model
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_log_level("INFO")

app = FastAPI(title="Gemini API FastAPI Server")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Multi-account client pool & configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "")
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() == "true"
TEMPORARY_CHAT = os.environ.get("TEMPORARY_CHAT", "false").lower() == "true"
AUTO_DELETE_CHAT = os.environ.get("AUTO_DELETE_CHAT", "true").lower() == "true" and not TEMPORARY_CHAT
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
SECRET_FILE_PATH = os.path.join(os.path.dirname(__file__), "secrets", "proxy_secret")
MAX_CONCURRENT_PER_ACCOUNT = int(os.environ.get("MAX_CONCURRENT_PER_ACCOUNT", "3"))

# --- Cookie pool parsing (backward-compatible) ---
_COOKIES_POOL_RAW = os.environ.get("GEMINI_COOKIES_POOL", "")
_SECURE_1PSID_LEGACY = os.environ.get("SECURE_1PSID", "")
_SECURE_1PSIDTS_LEGACY = os.environ.get("SECURE_1PSIDTS", "")

cookie_entries: List[Dict[str, str]] = []

if _COOKIES_POOL_RAW:
	try:
		_parsed = json.loads(_COOKIES_POOL_RAW)
		if isinstance(_parsed, list) and len(_parsed) > 0:
			cookie_entries = _parsed
			logger.info(f"🏊 Loaded {len(cookie_entries)} account(s) from GEMINI_COOKIES_POOL")
		else:
			logger.warning("⚠️ GEMINI_COOKIES_POOL is not a non-empty JSON array, falling back to single-account mode")
	except json.JSONDecodeError as e:
		logger.error(f"❌ Failed to parse GEMINI_COOKIES_POOL: {e}")

if not cookie_entries and _SECURE_1PSID_LEGACY:
	cookie_entries = [{"__Secure-1PSID": _SECURE_1PSID_LEGACY, "__Secure-1PSIDTS": _SECURE_1PSIDTS_LEGACY}]
	logger.info("📌 Using legacy single-account mode (SECURE_1PSID / SECURE_1PSIDTS)")

if not cookie_entries:
	logger.warning("⚠️ No Gemini credentials configured! Set GEMINI_COOKIES_POOL or SECURE_1PSID/SECURE_1PSIDTS.")

# Global client pool (populated at startup)
client_pool: List[GeminiClient] = []
account_semaphores: List[asyncio.Semaphore] = []


@app.on_event("startup")
async def _init_client_pool():
	"""Initialize one GeminiClient per cookie entry at application startup."""
	for i, entry in enumerate(cookie_entries):
		psid = entry.get("__Secure-1PSID", "")
		psidts = entry.get("__Secure-1PSIDTS", "")
		if not psid:
			logger.warning(f"⚠️ Account #{i} has no __Secure-1PSID, skipping")
			continue
		try:
			client = GeminiClient(psid, psidts)
			await client.init(timeout=300)
			client_pool.append(client)
			logger.info(f"✅ Client #{len(client_pool) - 1} initialized (PSID={psid[:8]}...)")
		except Exception as e:
			logger.error(f"❌ Failed to initialize client #{i}: {e}")
	if client_pool:
		# Create per-account semaphores for concurrency limiting
		for _ in client_pool:
			account_semaphores.append(asyncio.Semaphore(MAX_CONCURRENT_PER_ACCOUNT))
		logger.info(f"🚀 Client pool ready: {len(client_pool)} active client(s), max {MAX_CONCURRENT_PER_ACCOUNT} concurrent per account")
	else:
		logger.error("❌ Client pool is EMPTY — all requests will fail with 503")


def get_sticky_client(authorization: Optional[str]) -> tuple:
	"""
	Select a client from the pool using hash-based sticky session.
	The same Authorization token always maps to the same client,
	ensuring context/session continuity.
	"""
	if not client_pool:
		return None, -1
	if len(client_pool) == 1:
		return client_pool[0], 0
	key = authorization or "__default__"
	digest = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
	index = int(digest, 16) % len(client_pool)
	logger.info(f"🔀 Sticky dispatch: token_hash={digest[:8]}... → client #{index} (pool_size={len(client_pool)})")
	return client_pool[index], index

async def background_delete_chat(client: GeminiClient, cid: str):
	"""Deletes a chat conversation in the background to avoid blocking the main thread."""
	if not cid:
		return
	try:
		logger.info(f"Auto-deleting chat {cid} in background...")
		await client.delete_chat(cid)
		logger.info(f"Successfully auto-deleted chat {cid}")
	except Exception as e:
		logger.error(f"Failed to auto-delete chat {cid}: {e}")


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
			roi = img_array[y:y+logo_size, x:x+logo_size].copy()

			alpha = np.clip(alpha_map, 0.002, 0.99)
			alpha_expanded = np.expand_dims(alpha, axis=2)
			cleaned_roi = (roi - alpha_expanded * 255.0) / (1.0 - alpha_expanded)
			cleaned_roi = np.clip(np.round(cleaned_roi), 0, 255).astype(np.uint8)

			img_array_uint8 = np.array(img.convert("RGB"))
			img_array_uint8[y:y+logo_size, x:x+logo_size] = cleaned_roi

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


# Print debug info at startup
if not cookie_entries:
	logger.warning("⚠️ No Gemini credentials configured!")
	logger.warning("Set GEMINI_COOKIES_POOL (JSON array) or SECURE_1PSID/SECURE_1PSIDTS in .env")
else:
	for i, entry in enumerate(cookie_entries):
		psid_preview = entry.get("__Secure-1PSID", "")[:8]
		logger.info(f"Account #{i}: PSID={psid_preview}...")

if not API_KEY:
	logger.warning("⚠️ API_KEY is not set or empty! API authentication will not work.")
	logger.warning("Make sure API_KEY is correctly set in your .env file or environment.")
else:
	logger.info(f"API_KEY found. API_KEY starts with: {API_KEY[:5]}...")


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
	content: Union[str, List[ContentItem]]
	name: Optional[str] = None


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
	
	Raises:
		HTTPException: If the authorization header is missing, incorrectly formatted, or the token is invalid.
	"""
	if not API_KEY:
		# If API_KEY is not set in environment, skip validation (for development)
		logger.warning("API key validation skipped - no API_KEY set in environment")
		return

	if not authorization:
		raise HTTPException(status_code=401, detail="Missing Authorization header")

	try:
		scheme, token = authorization.split()
		if scheme.lower() != "bearer":
			raise HTTPException(status_code=401, detail="Invalid authentication scheme. Use Bearer token")

		if token != API_KEY:
			raise HTTPException(status_code=401, detail="Invalid API key")
	except ValueError:
		raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer YOUR_API_KEY'")

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
	except Exception as e:
		logger.error(f"Request failed: {str(e)}")
		return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "internal_server_error"}})


# Get list of available models
@app.get("/v1/models")
async def list_models():
	"""返回 gemini_webapi 中声明的模型列表"""
	now = int(datetime.now(tz=timezone.utc).timestamp())
	data = [
		{
			"id": m.model_name,  # 如 "gemini-2.0-flash"
			"object": "model",
			"created": now,
			"owned_by": "google-gemini-web",
		}
		for m in Model
	]
	print(data)
	return {"object": "list", "data": data}


# Helper to convert between Gemini and OpenAI model names
def map_model_name(openai_model_name: str) -> Model:
	"""根据模型名称字符串查找匹配的 Model 枚举值"""
	# 打印所有可用模型以便调试
	all_models = [m.model_name if hasattr(m, "model_name") else str(m) for m in Model]
	logger.info(f"Available models: {all_models}")

	# 首先尝试直接查找匹配的模型名称
	for m in Model:
		model_name = m.model_name if hasattr(m, "model_name") else str(m)
		if openai_model_name.lower() in model_name.lower():
			return m

	# 如果找不到匹配项，使用默认映射
	model_keywords = {
		"gemini-pro": ["pro", "2.0"],
		"gemini-pro-vision": ["vision", "pro"],
		"gemini-flash": ["flash", "2.0"],
		"gemini-1.5-pro": ["1.5", "pro"],
		"gemini-1.5-flash": ["1.5", "flash"],
	}

	# 根据关键词匹配
	keywords = model_keywords.get(openai_model_name, ["pro"])  # 默认使用pro模型

	for m in Model:
		model_name = m.model_name if hasattr(m, "model_name") else str(m)
		if all(kw.lower() in model_name.lower() for kw in keywords):
			return m

	# 如果还是找不到，返回第一个模型
	return next(iter(Model))


# Prepare conversation history from OpenAI messages format
def prepare_conversation(messages: List[Message]) -> tuple:
	"""
	Convert a list of OpenAI-formatted message objects into a 
	flat string conversation format suitable for the Gemini API.
	Also extracts and saves base64 images to temporary files.
	
	Returns:
		A tuple containing the constructed conversation string and a list of paths to temporary image files.
	"""
	conversation = ""
	temp_files = []

	for msg in messages:
		if isinstance(msg.content, str):
			# String content handling
			if msg.role == "system":
				conversation += f"System: {msg.content}\n\n"
			elif msg.role == "user":
				conversation += f"Human: {msg.content}\n\n"
			elif msg.role == "assistant":
				conversation += f"Assistant: {msg.content}\n\n"
		else:
			# Mixed content handling
			if msg.role == "user":
				conversation += "Human: "
			elif msg.role == "system":
				conversation += "System: "
			elif msg.role == "assistant":
				conversation += "Assistant: "

			for item in msg.content:
				if item.type == "text":
					conversation += item.text or ""
				elif item.type == "image_url" and item.image_url:
					# Handle image
					image_url = item.image_url.get("url", "")
					if image_url.startswith("data:image/"):
						# Process base64 encoded image
						try:
							# Extract the base64 part
							base64_data = image_url.split(",")[1]
							image_data = base64.b64decode(base64_data)

							# Create temporary file to hold the image
							with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
								tmp.write(image_data)
								temp_files.append(tmp.name)
						except Exception as e:
							logger.error(f"Error processing base64 image: {str(e)}")

			conversation += "\n\n"

	# Add a final prompt for the assistant to respond to
	conversation += "Assistant: "

	return conversation, temp_files




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


def extract_image_markdown(response, base_url: str, account_idx: int = 0) -> str:
	"""Extract images from a response and return markdown image links."""
	result = ""
	if hasattr(response, "images") and response.images:
		for img in response.images:
			img_url = getattr(img, "url", None)
			if img_url:
				sig = get_image_signature(img_url)
				proxy_url = f"{base_url}/gemini-proxy/image?url={quote(img_url)}&sig={sig}&account={account_idx}"
				result += f"\n\n![🎨 Loading image...]({proxy_url})"
	return result


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request, api_key: str = Depends(verify_api_key)):
	"""
	Handle chat completion requests, translating from OpenAI API format to Gemini API format.
	Supports both streaming and non-streaming responses, caching, thinking features,
	and background conversation cleanup based on configuration.
	"""
	try:
		# Sticky session: select client based on Authorization header hash
		auth_header = raw_request.headers.get("Authorization")
		client, client_idx = get_sticky_client(auth_header)
		if client is None:
			raise HTTPException(status_code=503, detail="No Gemini clients available. Check GEMINI_COOKIES_POOL config.")

		# Acquire per-account semaphore to limit concurrency
		semaphore = account_semaphores[client_idx] if client_idx < len(account_semaphores) else None
		if semaphore and semaphore.locked():
			logger.info(f"⏳ Client #{client_idx} at concurrency limit, request queued...")

		# 转换消息为对话格式
		conversation, temp_files = prepare_conversation(request.messages)
		logger.info(f"Prepared conversation: {conversation}")
		logger.info(f"Temp files: {temp_files}")

		# 获取适当的模型
		model = map_model_name(request.model)
		logger.info(f"Using model: {model}")

		# 创建响应对象
		completion_id = f"chatcmpl-{uuid.uuid4()}"
		created_time = int(time.time())
		base_url = PUBLIC_BASE_URL or str(raw_request.base_url).rstrip("/")

		# Prepare generate_content arguments
		gen_kwargs = {"model": model}
		if TEMPORARY_CHAT:
			gen_kwargs["temporary"] = True
		if temp_files:
			gen_kwargs["files"] = temp_files

		if request.stream:
			# Real streaming using upstream generate_content_stream
			async def generate_stream():
				async with semaphore if semaphore else asyncio.Semaphore(999):
					try:
						logger.info("Starting streaming response from Gemini...")

						def make_chunk(delta: dict, finish_reason=None):
							return "data: " + json.dumps({
								"id": completion_id,
								"object": "chat.completion.chunk",
								"created": created_time,
								"model": request.model,
								"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
							}) + "\n\n"

						# Send initial role chunk
						yield make_chunk({"role": "assistant"})

						thinking_started = False
						thinking_ended = False
						yielded_images = 0
						text_buffer = ""
						captured_cid = None

						async for chunk in client.generate_content_stream(conversation, **gen_kwargs):
							# Capture conversation ID for auto-deletion
							if AUTO_DELETE_CHAT and captured_cid is None and hasattr(chunk, "metadata") and chunk.metadata and len(chunk.metadata) > 0:
								captured_cid = chunk.metadata[0]

							# Handle thinking/thoughts delta
							if ENABLE_THINKING and hasattr(chunk, "thoughts_delta") and chunk.thoughts_delta:
								if not thinking_started:
									yield make_chunk({"content": "<think>\n"})
									thinking_started = True
								
								# Also include reasoning_content for full Open WebUI native compatibility
								yield make_chunk({
									"content": chunk.thoughts_delta,
									"reasoning_content": chunk.thoughts_delta
								})

							# Handle text delta
							if hasattr(chunk, "text_delta") and chunk.text_delta:
								# Close thinking tag before first text content
								if thinking_started and not thinking_ended:
									thinking_ended = True
									yield make_chunk({"content": "\n</think>\n\n"})
								
								text_buffer += chunk.text_delta
								safe_to_yield = False
								
								# Yield if buffer ends with whitespace and looks like it's outside a markdown link
								if text_buffer[-1].isspace() and text_buffer.count('[') == text_buffer.count(']') and text_buffer.count('(') == text_buffer.count(')'):
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
										proxy_url = f"{base_url}/gemini-proxy/image?url={quote(img_url)}&sig={sig}&account={client_idx}"
										img_md = f"\n\n![🎨 Loading image...]({proxy_url})\n\n"
										yield make_chunk({"content": img_md})
								yielded_images = len(chunk.images)

						# Flush any remaining text
						if text_buffer:
							yield make_chunk({"content": postprocess_text(text_buffer)})

						# Close thinking tag if it was never closed
						if thinking_started and not thinking_ended:
							yield make_chunk({"content": "\n</think>\n\n"})

						# Send finish chunk
						yield make_chunk({}, finish_reason="stop")
						yield "data: [DONE]\n\n"

						logger.info("Streaming response completed")
					except Exception as e:
						logger.error(f"Error during streaming: {str(e)}", exc_info=True)
						# Send error as a content chunk so the client sees it
						error_msg = "\n\n[An internal error occurred while streaming]"
						yield make_chunk({"content": error_msg})
						yield make_chunk({}, finish_reason="stop")
						yield "data: [DONE]\n\n"
					finally:
						# Create background task to delete the chat if AUTO_DELETE_CHAT is enabled
						if AUTO_DELETE_CHAT and captured_cid:
							asyncio.create_task(background_delete_chat(client, captured_cid))

						# 清理临时文件
						for temp_file in temp_files:
							try:
								os.unlink(temp_file)
							except Exception as e:
								logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")

			return StreamingResponse(generate_stream(), media_type="text/event-stream")
		else:
			# Non-streaming response with semaphore guard
			async with semaphore if semaphore else asyncio.Semaphore(999):
				logger.info("Sending request to Gemini...")
				try:
					response = await client.generate_content(conversation, **gen_kwargs)
					
					if AUTO_DELETE_CHAT and hasattr(response, "metadata") and response.metadata and len(response.metadata) > 0:
						cid = response.metadata[0]
						asyncio.create_task(background_delete_chat(client, cid))

				finally:
					# 清理临时文件
					for temp_file in temp_files:
						try:
							os.unlink(temp_file)
						except Exception as e:
							logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")

			# 提取文本响应
			reply_text = ""
			if ENABLE_THINKING and hasattr(response, "thoughts") and response.thoughts:
				reply_text += f"<think>\n{response.thoughts}\n</think>\n\n"
			if hasattr(response, "text"):
				reply_text += response.text
			else:
				reply_text += str(response)

			# 提取并追加图片响应
			reply_text += extract_image_markdown(response, base_url, account_idx=client_idx)
			reply_text = postprocess_text(reply_text)

			logger.info(f"Response: {reply_text}")

			if not reply_text or reply_text.strip() == "":
				logger.warning("Empty response received from Gemini")
				reply_text = "Server returned an empty response. Please check that Gemini API credentials are valid."

			result = {
				"id": completion_id,
				"object": "chat.completion",
				"created": created_time,
				"model": request.model,
				"choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
				"usage": {
					"prompt_tokens": len(conversation.split()),
					"completion_tokens": len(reply_text.split()),
					"total_tokens": len(conversation.split()) + len(reply_text.split()),
				},
			}

			logger.info(f"Returning response: {result}")
			return result

	except Exception as e:
		logger.error(f"Error generating completion: {str(e)}", exc_info=True)
		raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")


@app.get("/gemini-proxy/image")
async def proxy_image(url: str, sig: str, account: int = 0):
	"""
	Proxy images from Google domains to bypass browser security policies.
	Requires a valid HMAC signature. The `account` parameter selects which
	account's cookies to use for fetching (defaults to first account).
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
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0",
		"Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.9",
		"Referer": "https://gemini.google.com/",
	}

	# 10MB limit
	MAX_BYTES = 10 * 1024 * 1024

	# Use cookies from the specified account (clamp to valid range)
	account_idx = max(0, min(account, len(cookie_entries) - 1)) if cookie_entries else 0
	entry = cookie_entries[account_idx] if cookie_entries else {}
	psid = entry.get("__Secure-1PSID", "")
	psidts = entry.get("__Secure-1PSIDTS", "")

	jar = httpx.Cookies()
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
					logger.info(f"Checking for Gemini watermark on {url}")
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
			raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch image: Google returned {e.response.status_code}")
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Proxy error: {str(e)}")
			raise HTTPException(status_code=500, detail="Internal proxy error")


@app.get("/")
async def root():
	"""
	Health check endpoint to verify the API server is currently running.
	"""
	return {"status": "online", "message": "Gemini API FastAPI Server is running"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
