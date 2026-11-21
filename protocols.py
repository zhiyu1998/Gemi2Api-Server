"""额外协议适配层：OpenAI Responses、Anthropic Messages、Gemini 原生 generateContent。

现有 /v1/chat/completions 只覆盖 OpenAI Chat Completions。这里把参考实现
（Gemini-Web2API 的 internal/adapter/{responses,claude,gemini}_handler.go）里另外三种
协议的请求/响应形状搬过来，底层仍然复用同一套模型解析 + Cookie 池 + gemini_wire。

实现范围：文本对话、system/instructions、temperature/top_p、stop_sequences、tools（以
提示词围栏模拟，与 chat/completions 一致）。图片、音频、视频、Deep Research 这些富媒体
不在本适配层内，继续走 /v1/chat/completions 的 gemini-* 模型。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gemini_tools import build_tools_instruction, parse_tool_calls


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------
def _check_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
	import main

	if not main.API_KEY:
		return
	token = ""
	if authorization and authorization.startswith("Bearer "):
		token = authorization[7:].strip()
	elif x_api_key:
		token = x_api_key.strip()
	if token != main.API_KEY:
		raise HTTPException(status_code=401, detail="invalid api key")


def _content_to_text(content: Any) -> str:
	if content is None:
		return ""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		bits: List[str] = []
		for block in content:
			if not isinstance(block, dict):
				bits.append(str(block))
				continue
			block_type = block.get("type")
			if block_type in (None, "text", "input_text", "output_text"):
				bits.append(str(block.get("text") or ""))
			elif block_type == "tool_result":
				inner = block.get("content")
				bits.append(f"Tool result ({block.get('tool_use_id') or ''}): {_content_to_text(inner)}")
			elif block_type in ("image_url", "image"):
				bits.append("[image omitted]")
		return "\n".join(bits)
	return str(content)


def _system_text(system: Any) -> str:
	return _content_to_text(system).strip()


def _build_prompt(system: str, messages: List[Dict[str, Any]], tools_instruction: str = "") -> str:
	parts: List[str] = []
	if tools_instruction:
		parts.append(tools_instruction)
	if system:
		parts.append(f"System: {system}")
	for message in messages:
		role = (message.get("role") or "user").lower()
		text = _content_to_text(message.get("content"))
		if role == "assistant":
			parts.append(f"Assistant: {text}")
		elif role == "system":
			parts.append(f"System: {text}")
		else:
			parts.append(f"Human: {text}")
	return "\n\n".join(part for part in parts if part) + "\n\nAssistant: "


def _tokens(text: str) -> int:
	# 粗估：网页端没有 tokenizer，按 4 字符 ≈ 1 token
	return max(1, len(text) // 4)


def _openai_tools(tools: Any) -> List[Dict[str, Any]]:
	"""把 Anthropic/Gemini 风格的工具定义统一成 OpenAI 形式，交给提示词围栏。"""
	result: List[Dict[str, Any]] = []
	if not isinstance(tools, list):
		return result
	for tool in tools:
		if not isinstance(tool, dict):
			continue
		if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
			result.append(tool)
			continue
		name = tool.get("name")
		if not name:
			continue
		schema = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
		result.append({"type": "function", "function": {"name": name, "description": tool.get("description", ""), "parameters": schema}})
	return result


def _tool_calls(clean_text: str, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	normalized: List[Dict[str, Any]] = []
	for call in calls or []:
		function = call.get("function") if isinstance(call, dict) else None
		if not isinstance(function, dict):
			continue
		arguments = function.get("arguments")
		try:
			parsed = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
		except Exception:
			parsed = {"_raw": arguments}
		normalized.append(
			{
				"id": call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
				"name": function.get("name") or "",
				"input": parsed,
			}
		)
	return normalized


def _stop_truncate(text: str, stops: List[str]) -> tuple:
	if not stops:
		return text, None
	for stop in stops:
		if stop and stop in text:
			return text.split(stop, 1)[0], stop
	return text, None


def _resolve(model_name: str):
	import main

	spec = main.resolve_model_spec(model_name or "")
	if spec is None:
		raise HTTPException(status_code=404, detail=f"model not found: {model_name}")
	return spec


def _gen_kwargs(spec) -> Dict[str, Any]:
	import main

	kwargs: Dict[str, Any] = {"model": main.build_model_arg(spec)}
	if main.TEMPORARY_CHAT and not spec.feature:
		kwargs["temporary"] = True
	if spec.extended_thinking and main._supports_extended_thinking(main.GeminiClient):
		kwargs["extended_thinking"] = True
	return kwargs


async def _generate_text(model_name: str, prompt: str) -> tuple:
	import main

	spec = _resolve(model_name)
	pool = await main.get_gemini_pool()
	response, _client = await main.generate_with_cookie_pool(pool, prompt, _gen_kwargs(spec), spec)
	text = getattr(response, "text", None) or ""
	return spec, response, text


async def _stream_chunks(model_name: str, prompt: str) -> AsyncIterator[tuple]:
	"""逐块产出 (kind, payload)：kind 为 text / thinking / metadata。"""
	import main

	spec = _resolve(model_name)
	pool = await main.get_gemini_pool()
	lease = await main.acquire_lease(pool)
	token = main.gemini_wire_push_spec(spec)
	try:
		if spec.feature:
			await main.ensure_task_ready(lease.client)
		async for chunk in lease.client.generate_content_stream(prompt, **_gen_kwargs(spec)):
			metadata = getattr(chunk, "metadata", None)
			if metadata:
				yield ("metadata", metadata)
			thoughts = getattr(chunk, "thoughts_delta", None)
			if thoughts:
				yield ("thinking", thoughts)
			delta = getattr(chunk, "text_delta", None)
			if delta:
				yield ("text", delta)
		lease.succeed()
	except Exception as exc:
		lease.fail(exc)
		raise
	finally:
		main.gemini_wire_pop_spec(token)
		lease.release()


def _sse(event: str, payload: Dict[str, Any]) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------
def _register_anthropic(app: FastAPI) -> None:
	@app.post("/v1/messages")
	async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
		_check_auth(authorization, x_api_key)
		body = await request.json()
		model_name = body.get("model") or ""
		messages = body.get("messages") or []
		system = _system_text(body.get("system"))
		stream = bool(body.get("stream"))
		stops = [s for s in (body.get("stop_sequences") or []) if isinstance(s, str)]
		openai_tools = _openai_tools(body.get("tools"))
		tools_top, tools_bottom = build_tools_instruction(openai_tools, body.get("tool_choice")) if openai_tools else ("", "")
		prompt = _build_prompt(system, messages, tools_top)
		if tools_bottom:
			prompt += "\n" + tools_bottom
		message_id = f"msg_{uuid.uuid4().hex[:24]}"

		if not stream:
			spec, response, text = await _generate_text(model_name, prompt)
			text, stop_hit = _stop_truncate(text, stops)
			calls: List[Dict[str, Any]] = []
			if openai_tools:
				text, raw_calls = parse_tool_calls(text)
				calls = _tool_calls(text, raw_calls)
			blocks: List[Dict[str, Any]] = []
			if text.strip():
				blocks.append({"type": "text", "text": text})
			for call in calls:
				blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["input"]})
			stop_reason = "tool_use" if calls else ("stop_sequence" if stop_hit else "end_turn")
			return {
				"id": message_id,
				"type": "message",
				"role": "assistant",
				"model": model_name,
				"content": blocks or [{"type": "text", "text": ""}],
				"stop_reason": stop_reason,
				"stop_sequence": stop_hit,
				"usage": {"input_tokens": _tokens(prompt), "output_tokens": _tokens(text)},
			}

		async def event_stream():
			yield _sse("message_start", {"type": "message_start", "message": {"id": message_id, "type": "message", "role": "assistant", "model": model_name, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": _tokens(prompt), "output_tokens": 0}}})
			yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
			collected = ""
			try:
				async for kind, payload in _stream_chunks(model_name, prompt):
					if kind == "text":
						collected += payload
						yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": payload}})
			except Exception as exc:
				yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
				return
			text, stop_hit = _stop_truncate(collected, stops)
			yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
			yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "stop_sequence" if stop_hit else "end_turn", "stop_sequence": stop_hit}, "usage": {"output_tokens": _tokens(text)}})
			yield _sse("message_stop", {"type": "message_stop"})

		return StreamingResponse(event_stream(), media_type="text/event-stream")

	@app.post("/v1/messages/count_tokens")
	async def anthropic_count_tokens(request: Request, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
		_check_auth(authorization, x_api_key)
		body = await request.json()
		system = _system_text(body.get("system"))
		messages = body.get("messages") or []
		prompt = _build_prompt(system, messages)
		return {"input_tokens": _tokens(prompt)}


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------
def _response_shell(response_id: str, model: str, created: int, status: str) -> Dict[str, Any]:
	return {
		"id": response_id,
		"object": "response",
		"created_at": created,
		"completed_at": None,
		"status": status,
		"model": model,
		"output": [],
		"error": None,
		"incomplete_details": None,
		"instructions": None,
		"metadata": {},
		"parallel_tool_calls": True,
		"previous_response_id": None,
		"reasoning": None,
		"temperature": 1,
		"text": {"format": {"type": "text"}},
		"tool_choice": "auto",
		"tools": [],
		"top_p": 1,
		"truncation": "disabled",
		"usage": None,
	}


def _register_responses(app: FastAPI) -> None:
	@app.post("/v1/responses")
	async def openai_responses(request: Request, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
		_check_auth(authorization, x_api_key)
		body = await request.json()
		model_name = body.get("model") or ""
		instructions = _system_text(body.get("instructions"))
		raw_input = body.get("input")
		messages: List[Dict[str, Any]] = []
		if isinstance(raw_input, str):
			messages.append({"role": "user", "content": raw_input})
		elif isinstance(raw_input, list):
			for item in raw_input:
				if isinstance(item, str):
					messages.append({"role": "user", "content": item})
				elif isinstance(item, dict):
					role = item.get("role") or "user"
					content = item.get("content")
					if isinstance(content, str):
						messages.append({"role": role, "content": content})
					elif isinstance(content, list):
						messages.append({"role": role, "content": content})
		stream = bool(body.get("stream"))
		prompt = _build_prompt(instructions, messages)
		response_id = f"resp_{uuid.uuid4().hex[:24]}"
		item_id = f"msg_{uuid.uuid4().hex[:24]}"
		created = int(time.time())

		if not stream:
			spec, response, text = await _generate_text(model_name, prompt)
			output_item = {
				"id": item_id,
				"type": "message",
				"status": "completed",
				"role": "assistant",
				"content": [{"type": "output_text", "text": text, "annotations": []}],
			}
			payload = _response_shell(response_id, model_name, created, "completed")
			payload["instructions"] = instructions or None
			payload["output"] = [output_item]
			payload["usage"] = {"input_tokens": _tokens(prompt), "output_tokens": _tokens(text), "total_tokens": _tokens(prompt) + _tokens(text)}
			return payload

		async def event_stream():
			sequence = 0

			def emit(event_type: str, payload_obj: Dict[str, Any]) -> str:
				nonlocal sequence
				payload_obj["type"] = event_type
				payload_obj["sequence_number"] = sequence
				sequence += 1
				return _sse(event_type, payload_obj)

			yield emit("response.created", {"response": _response_shell(response_id, model_name, created, "in_progress")})
			yield emit("response.in_progress", {"response": _response_shell(response_id, model_name, created, "in_progress")})
			yield emit("response.output_item.added", {"output_index": 0, "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}})
			yield emit("response.content_part.added", {"item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
			collected = ""
			try:
				async for kind, payload in _stream_chunks(model_name, prompt):
					if kind == "text":
						collected += payload
						yield emit("response.output_text.delta", {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": payload, "logprobs": []})
			except Exception as exc:
				yield emit("response.failed", {"response": {**_response_shell(response_id, model_name, created, "failed"), "error": {"code": "server_error", "message": str(exc)}}})
				return
			part = {"type": "output_text", "text": collected, "annotations": []}
			yield emit("response.output_text.done", {"item_id": item_id, "output_index": 0, "content_index": 0, "text": collected, "logprobs": []})
			yield emit("response.content_part.done", {"item_id": item_id, "output_index": 0, "content_index": 0, "part": part})
			item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]}
			yield emit("response.output_item.done", {"output_index": 0, "item": item})
			final = _response_shell(response_id, model_name, created, "completed")
			final["instructions"] = instructions or None
			final["output"] = [item]
			final["usage"] = {"input_tokens": _tokens(prompt), "output_tokens": _tokens(collected), "total_tokens": _tokens(prompt) + _tokens(collected)}
			yield emit("response.completed", {"response": final})

		return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Gemini 原生 generateContent / streamGenerateContent
# ---------------------------------------------------------------------------
def _gemini_prompt(body: Dict[str, Any]) -> str:
	system_parts = ((body.get("systemInstruction") or {}).get("parts") or [])
	system = "\n".join(str(p.get("text") or "") for p in system_parts if isinstance(p, dict))
	messages: List[Dict[str, Any]] = []
	for content in body.get("contents") or []:
		if not isinstance(content, dict):
			continue
		role = "assistant" if (content.get("role") or "").lower() in ("model", "assistant") else "user"
		text = "\n".join(str(p.get("text") or "") for p in (content.get("parts") or []) if isinstance(p, dict))
		messages.append({"role": role, "content": text})
	return _build_prompt(system, messages)


def _gemini_response(model_name: str, text: str, prompt: str) -> Dict[str, Any]:
	return {
		"candidates": [
			{
				"content": {"role": "model", "parts": [{"text": text}]},
				"finishReason": "STOP",
				"index": 0,
			}
		],
		"usageMetadata": {
			"promptTokenCount": _tokens(prompt),
			"candidatesTokenCount": _tokens(text),
			"totalTokenCount": _tokens(prompt) + _tokens(text),
		},
		"modelVersion": model_name,
	}


async def _gemini_action(model_name: str, action: str, request: Request):
	body = await request.json()
	if action == "countTokens":
		prompt = _gemini_prompt(body)
		return JSONResponse({"totalTokens": _tokens(prompt)})
	if action not in ("generateContent", "streamGenerateContent"):
		raise HTTPException(status_code=400, detail=f"unsupported method: {action}")

	prompt = _gemini_prompt(body)
	config = body.get("generationConfig") or {}
	stops = [s for s in (config.get("stopSequences") or []) if isinstance(s, str)]

	if action == "generateContent":
		spec, response, text = await _generate_text(model_name, prompt)
		text, _ = _stop_truncate(text, stops)
		return JSONResponse(_gemini_response(model_name, text, prompt))

	alt_sse = (request.query_params.get("alt") or "").lower() == "sse"

	async def json_array_stream():
		yield "["
		first = True
		collected = ""
		try:
			async for kind, payload in _stream_chunks(model_name, prompt):
				if kind == "text":
					collected += payload
					chunk = {"candidates": [{"content": {"role": "model", "parts": [{"text": payload}]}, "index": 0}]}
					separator = "" if first else ","
					first = False
					yield separator + json.dumps(chunk, ensure_ascii=False)
		except Exception as exc:
			yield ("," if not first else "") + json.dumps({"error": {"code": 500, "message": str(exc), "status": "INTERNAL"}}, ensure_ascii=False)
			yield "]"
			return
		collected, _ = _stop_truncate(collected, stops)
		final = {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": _tokens(prompt), "candidatesTokenCount": _tokens(collected), "totalTokenCount": _tokens(prompt) + _tokens(collected)}, "modelVersion": model_name}
		yield ("," if not first else "") + json.dumps(final, ensure_ascii=False)
		yield "]"

	async def sse_stream():
		collected = ""
		try:
			async for kind, payload in _stream_chunks(model_name, prompt):
				if kind == "text":
					collected += payload
					chunk = {"candidates": [{"content": {"role": "model", "parts": [{"text": payload}]}, "index": 0}]}
					yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
		except Exception as exc:
			yield f"data: {json.dumps({'error': {'code': 500, 'message': str(exc), 'status': 'INTERNAL'}}, ensure_ascii=False)}\n\n"
			return
		collected, _ = _stop_truncate(collected, stops)
		final = {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": _tokens(prompt), "candidatesTokenCount": _tokens(collected), "totalTokenCount": _tokens(prompt) + _tokens(collected)}, "modelVersion": model_name}
		yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

	if alt_sse:
		return StreamingResponse(sse_stream(), media_type="text/event-stream")
	return StreamingResponse(json_array_stream(), media_type="application/json; charset=UTF-8")


def _register_gemini(app: FastAPI) -> None:
	for prefix in ("/v1beta", "/v1"):
		@app.post(prefix + "/models/{model_action:path}")
		async def gemini_router(model_action: str, request: Request, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
			_check_auth(authorization, x_api_key)
			if ":" not in model_action:
				raise HTTPException(status_code=404, detail="unknown method")
			model_name, action = model_action.split(":", 1)
			return await _gemini_action(model_name, action, request)


def register_protocol_routes(app: FastAPI) -> None:
	_register_anthropic(app)
	_register_responses(app)
	_register_gemini(app)
