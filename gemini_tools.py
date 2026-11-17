"""OpenAI tools / function calling 的提示词模拟层。

Gemini 网页端没有协议层的工具调用（StreamGenerate 里没有 functionCall 字段），
所以只能让模型按约定吐 ```tool_call``` 围栏，再把围栏解析回 OpenAI 的 tool_calls。
这套做法与 gemini-web2api-go 的 messages.go 对齐，几个踩过的坑也一并照搬：

1. **格式指令被冲掉**（issue #15）：agentic 客户端的开发者提示动辄几十 KB，还自带
   "emit function calls" 这类原生工具框架措辞，压在中间会把顶部的格式指令冲没，
   模型于是答"我没有工具"或答非所问。所以末尾再锚一次，并给一个行为化示例。
2. **围栏闭合不能只看第一个 ``` **：写代码类工具的参数里常带 markdown 代码块，
   JSON 字符串里的 ``` 是原样出现的。先按 JSON 解析出一个完整值，它后面的第一个
   ``` 才是闭合围栏；JSON 语法错才退回"第一个 ```"。
3. **流式不能把围栏原文推给客户端**：围栏要完整文本才能解析，边出边发会让客户端
   看到一段 markdown 代码块而不是 tool_calls。只发确定不在围栏里的部分，尾巴上
   压着半个开围栏（`` `` ``、`` ```tool_c ``）也得扣住等下一帧。
4. **流式 tool_call 必须带 index**（issue #10）：OpenAI 流式规范要求客户端靠它把
   分片的 tool_call 拼起来。
5. **工具结果要给出清爽的成功信号**：Codex 风格结果是 "Chunk ID… / Wall time… /
   Process exited with code 0 / Output:" 的包装，写文件类命令 stdout 为空，弱模型
   会把"无输出"当成"没干成"，于是换个写法反复重试。这里压成一行，并说明无输出正常。
"""

from __future__ import annotations

import json
import re
import secrets
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

TOOL_FENCE_OPEN = "```tool_call"
TOOL_FENCE_CLOSE = "```"

#: 模型写工具名时用过的键
TOOL_CALL_NAME_KEYS = ("name", "tool", "tool_name", "function_name")
#: 模型写参数时用过的容器键，按优先级排：有多个时只取第一个能用的，不合并
TOOL_CALL_ARG_KEYS = ("arguments", "parameters", "input", "args")
#: 平铺兜底时要跳过的非参数键
TOOL_CALL_META_KEYS = ("function", "type", "id", "thought")

#: 从工具结果里抠退出码（Codex 的 shell 结果是一段带 "Process exited with code N" 的包装）
_EXIT_CODE_RE = re.compile(r"exit(?:ed with|\s*code)?\s*(?:code\s*)?(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# tool_choice
# ---------------------------------------------------------------------------
def parse_tool_choice(tool_choice: Any) -> Tuple[str, str]:
	"""解析 OpenAI 的 tool_choice，返回 (mode, forced_name)。

	mode ∈ {"auto","none","required"}；forced_name 非空表示客户端点名了某个函数。
	"""
	if isinstance(tool_choice, str):
		if tool_choice in ("none", "required", "auto"):
			return tool_choice, ""
	elif isinstance(tool_choice, dict):
		# {"type":"function","function":{"name":"..."}}
		function = tool_choice.get("function")
		if isinstance(function, dict):
			name = function.get("name")
			if isinstance(name, str) and name.strip():
				return "required", name.strip()
	return "auto", ""


def _iter_tool_definitions(tools: Iterable[Any]) -> Iterable[Dict[str, Any]]:
	"""把 tools[] 里 OpenAI 的 {type:function, function:{...}} 拆成函数定义。"""
	for tool in tools or ():
		if not isinstance(tool, dict):
			continue
		definition = tool
		if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
			definition = tool["function"]
		if not isinstance(definition, dict):
			continue
		name = definition.get("name")
		if not isinstance(name, str) or not name.strip():
			continue
		yield {
			"name": name.strip(),
			"description": definition.get("description", "") or "",
			"parameters": definition.get("parameters") or {"type": "object", "properties": {}},
		}


def build_tools_instruction(tools: Optional[List[Any]], tool_choice: Any = None) -> Tuple[str, str]:
	"""生成 (顶部格式指令, 末尾再锚一次的指令)；没有可用工具时返回 ("", "")。"""
	definitions = list(_iter_tool_definitions(tools))
	if not definitions:
		return "", ""

	mode, forced = parse_tool_choice(tool_choice)
	if mode == "none":
		return "", ""

	if forced:
		definitions = [d for d in definitions if d["name"] == forced]
		if not definitions:
			return "", ""

	if forced:
		rule = (
			f'You MUST call the tool "{forced}". Reply with the tool_call block and nothing '
			"else — do not answer the question yourself, even if you know the answer."
		)
	elif mode == "required":
		rule = (
			"You MUST call one of the tools above. Reply with the tool_call block and nothing "
			"else — do not answer the question yourself, even if you know the answer."
		)
	else:
		rule = "Only use tool_call blocks when needed."

	top = (
		"[System instruction]: You have access to tools. To call a tool, respond with:\n"
		f"{TOOL_FENCE_OPEN}\n"
		'{"name": "func_name", "arguments": {...}}\n'
		f"{TOOL_FENCE_CLOSE}\n"
		f"{rule}\n\n"
		"Available tools:\n"
		+ json.dumps(definitions, ensure_ascii=False, indent=2)
	)

	# 末尾再锚一次：压过客户端自带的「原生工具框架」措辞，并给行为化示例。
	# 实测只放格式指令、不带示例时弱模型 0/4 命中，全退化成写 ```powershell 给用户看；
	# 带一个「用户问 X → 正确回复是这个 tool_call 块」的示例则 3-4/4 命中。
	bottom = (
		"[System instruction — highest priority]: To run a command or read a file you MUST "
		f"output a {TOOL_FENCE_OPEN}{TOOL_FENCE_CLOSE} block — this is the ONLY execution channel. "
		"A ```powershell/```bash/```cmd/```python code block is NOT executed, it is only shown "
		"to the user; never use one to run anything, and never claim you lack tools.\n"
		"Act, do not explain: when the request needs a tool, your reply is ONE "
		f"{TOOL_FENCE_OPEN}{TOOL_FENCE_CLOSE} block and nothing else — do not list the commands "
		"you 'would' run, do not output more than one tool_call, do not add prose. If several "
		"steps are needed, do the FIRST one now; you will be called again with its result to "
		"continue.\n"
		"Know when to STOP: once a tool result shows a step succeeded (for example it exited "
		"with code 0, even with no printed output), that step is DONE — do not run it again and "
		"do not retry the same thing a different way. When the whole request is accomplished, "
		"stop calling tools and reply with a short final answer instead of another tool_call.\n"
		'Example of a correct turn — user asks "list the files here and tell me the size of '
		'README.md", you reply with exactly one block and nothing else:\n'
		f'{TOOL_FENCE_OPEN}\n'
		'{"name": "run_command", "arguments": {"command": "ls -la"}}\n'
		f"{TOOL_FENCE_CLOSE}\n"
		"(In your block use one of the real tool names listed above with its own arguments.) "
		"When the request needs a tool, emit the tool_call block now."
	)
	return top, bottom


# ---------------------------------------------------------------------------
# 工具结果渲染
# ---------------------------------------------------------------------------
def format_tool_result(name: str, raw: str) -> str:
	"""把工具结果渲染进 prompt，关键是让弱模型认得出「成功」。"""
	label = f"Tool result for {name}" if name else "Tool result"
	trimmed = (raw or "").strip()
	if not trimmed:
		return f"[{label}]: ✅ 完成，无输出（正常，动作已生效，不要重复执行）。"

	# Codex 风格包装：末尾 "Output:" 后是真正的 stdout
	out = trimmed
	idx = trimmed.rfind("Output:")
	if idx >= 0:
		out = trimmed[idx + len("Output:") :].strip()

	match = _EXIT_CODE_RE.search(trimmed)
	if match:
		code = match.group(1)
		if code == "0":
			if not out:
				return (
					f"[{label}]: ✅ 命令成功（exit 0），无文本输出——这对写文件/设值类命令是正常的，"
					"动作已完成，不要再跑同一条命令。"
				)
			return f"[{label}]: ✅ 命令成功（exit 0）。输出：\n{out}"
		return f"[{label}]: ❌ 命令失败（exit {code}）。输出：\n{out}"
	return f"[{label}]: {raw}"


# ---------------------------------------------------------------------------
# 围栏闭合判定
# ---------------------------------------------------------------------------
def partial_prefix_len(text: str, marker: str) -> int:
	"""返回 text 末尾有多少字符是 marker 的前缀（不含完整匹配）。"""
	limit = min(len(marker) - 1, len(text))
	for k in range(limit, 0, -1):
		if marker.startswith(text[-k:]):
			return k
	return 0


def tool_fence_end(text: str, final: bool) -> Tuple[int, int, bool]:
	"""在开围栏之后的文本里找闭合围栏。

	返回 (body, next, ok)：body 是围栏体的长度，next 是闭合围栏之后的位置。
	ok=False 表示还判定不了（流式时等下一帧）或者没闭合。

	闭合判据：先按 JSON 解析出围栏体里的一个完整值，它后面的第一个 ``` 才是闭合围栏。
	不能直接找第一个 ```：写代码类工具的参数里常带 markdown 代码块，JSON 字符串里的
	``` 是原样出现的，按第一个算会把调用截成半截 JSON。
	JSON 写坏（语法错）退回「第一个 ```」；JSON 没写完时流式等下一帧、final 同样退回。
	"""
	lead = len(text) - len(text.lstrip(" \t\r\n"))
	start = 0
	try:
		_value, end = json.JSONDecoder().raw_decode(text[lead:])
	except json.JSONDecodeError as exc:
		# 错误落在末尾 = 输入还没写完（Expecting ',' delimiter / Unterminated string 等）
		if not final and lead + exc.pos >= len(text) - 1:
			return 0, 0, False
	except ValueError:
		pass
	else:
		end += lead
		rest = text[end:].lstrip(" \t\r\n")
		if rest.startswith(TOOL_FENCE_CLOSE):
			body = len(text) - len(rest)
			return body, body + len(TOOL_FENCE_CLOSE), True
		if not final and TOOL_FENCE_CLOSE.startswith(rest):
			return 0, 0, False
		start = end

	idx = text.find(TOOL_FENCE_CLOSE, start)
	if idx < 0:
		return 0, 0, False
	return idx, idx + len(TOOL_FENCE_CLOSE), True


# ---------------------------------------------------------------------------
# tool_calls 解析
# ---------------------------------------------------------------------------
def _args_object(value: Any) -> Optional[str]:
	"""接受对象，或内容是 JSON 对象的字符串；缺失、null、数组、标量都返回 None。"""
	if isinstance(value, dict):
		return json.dumps(value, ensure_ascii=False)
	if isinstance(value, str):
		try:
			parsed = json.loads(value.strip())
		except (json.JSONDecodeError, ValueError):
			return None
		if isinstance(parsed, dict):
			return json.dumps(parsed, ensure_ascii=False)
	return None


def tool_call_name(raw: Dict[str, Any]) -> str:
	"""取围栏里的工具名，也认 {"function":{"name":…}} 嵌套。"""
	for key in TOOL_CALL_NAME_KEYS:
		value = raw.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	nested = raw.get("function")
	if isinstance(nested, dict):
		value = nested.get("name")
		if isinstance(value, str) and value.strip():
			return value.strip()
	return ""


def tool_call_args(raw: Dict[str, Any]) -> str:
	"""把围栏里的参数整理成 JSON 对象字符串，保证不是 "null"（OpenAI 客户端只收对象）。"""
	nested = raw.get("function")
	if isinstance(nested, dict):
		args = _args_object(nested.get("arguments"))
		if args:
			return args
	for key in TOOL_CALL_ARG_KEYS:
		args = _args_object(raw.get(key))
		if args:
			return args

	# 平铺兜底：其余键当参数
	flat = {
		key: value
		for key, value in raw.items()
		if key not in TOOL_CALL_NAME_KEYS and key not in TOOL_CALL_ARG_KEYS and key not in TOOL_CALL_META_KEYS
	}
	return json.dumps(flat, ensure_ascii=False)


def new_tool_call_id() -> str:
	"""生成 OpenAI 风格的调用 id。"""
	return "call_" + secrets.token_hex(4)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
	"""抽出所有 ```tool_call``` 围栏，返回 (剥掉围栏的正文, tool_calls)。

	围栏内不强求前后换行——模型有时吐 ```tool_call\\n{...}\\n```、有时一行内；
	内容交给 JSON 解析。解析不出的围栏也从正文里去掉，未闭合的原样留下。
	"""
	tool_calls: List[Dict[str, Any]] = []
	clean: List[str] = []
	rest = text or ""
	while True:
		start = rest.find(TOOL_FENCE_OPEN)
		if start < 0:
			break
		after = rest[start + len(TOOL_FENCE_OPEN) :]
		body, nxt, ok = tool_fence_end(after, True)
		if not ok:
			break
		clean.append(rest[:start])
		rest = after[nxt:]
		try:
			raw = json.loads(after[:body].strip())
		except (json.JSONDecodeError, ValueError):
			continue
		if not isinstance(raw, dict):
			continue
		name = tool_call_name(raw)
		if not name:
			continue
		tool_calls.append(
			{
				"id": new_tool_call_id(),
				"type": "function",
				"function": {"name": name, "arguments": tool_call_args(raw)},
			}
		)
	clean.append(rest)
	return "".join(clean).strip(), tool_calls


def tool_calls_for_stream(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""流式 delta 里每个 tool_call 必须带 index，客户端靠它把分片拼起来（issue #10）。"""
	out = []
	for index, call in enumerate(tool_calls):
		out.append(
			{
				"index": index,
				"id": call.get("id") or new_tool_call_id(),
				"type": call.get("type") or "function",
				"function": {
					"name": call.get("function", {}).get("name", ""),
					"arguments": call.get("function", {}).get("arguments", "{}"),
				},
			}
		)
	return out


# ---------------------------------------------------------------------------
# 流式闸门
# ---------------------------------------------------------------------------
class ToolFenceGate:
	"""只放行「确定不属于围栏」的文本，围栏内容全部扣住交给 parse_tool_calls。

	尾巴上可能压着半个开围栏（只到两个反引号，或到 tool_c 就断了），那部分也得扣住
	等下一帧——否则先发出去，下一帧才发现它是围栏开头，已发出的内容收不回来。
	"""

	def __init__(self, emit: Callable[[str], None]):
		self._emit = emit
		self._buffer = ""
		self._sent: List[str] = []
		self._in_fence = False

	@property
	def sent(self) -> str:
		"""到目前为止实际发给客户端的全部文本。"""
		return "".join(self._sent)

	def _send(self, text: str) -> None:
		if not text:
			return
		self._sent.append(text)
		self._emit(text)

	def push(self, delta: str) -> None:
		"""吃进一段增量文本，把确定不在围栏里的部分立刻发出去。"""
		if not delta:
			return
		self._buffer += delta
		while True:
			if self._in_fence:
				_body, nxt, ok = tool_fence_end(self._buffer, False)
				if not ok:
					return  # 围栏还没闭合，整段扣住
				self._buffer = self._buffer[nxt:]
				self._in_fence = False
				continue

			start = self._buffer.find(TOOL_FENCE_OPEN)
			if start >= 0:
				self._send(self._buffer[:start])
				self._buffer = self._buffer[start + len(TOOL_FENCE_OPEN) :]
				self._in_fence = True
				continue

			keep = partial_prefix_len(self._buffer, TOOL_FENCE_OPEN)
			cut = len(self._buffer) - keep
			self._send(self._buffer[:cut])
			self._buffer = self._buffer[cut:]
			return

	def remaining(self, final_text: str) -> str:
		"""收尾要补发的尾巴：final_text 里还没发给客户端的那部分。

		final_text 必须是 parse_tool_calls 处理过的正文（围栏已剥掉），否则前缀对不上。
		"""
		sent = self.sent
		if sent and final_text.startswith(sent):
			return final_text[len(sent) :]
		return final_text if not sent else ""
