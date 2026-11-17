"""把 Gemini 4 系列真正送进 StreamGenerate 的线上改写层。

gemini-webapi 只把模型 hex 写进 ``x-goog-ext-525001261-jspb``，4 系列还需要三样东西才能被服务端
认下来（对齐 gemini-web2api-go 的抓包结论）：

1. header 下标 14 的**档位**：4 Carbon/Argon 必须写 8，光有 hex 不生效；
2. header 下标 15 / ``inner[80]`` 的**思考档**：1 普通、2 扩展(High)、4 中档；
3. 档位 8 的请求要补 ``inner[49]=40`` 等槽位，否则付费号上会挂到超时。

这些槽位在不同版本的库里位置不一样（2.0.0 的 inner 只有 69 槽，2.1.1 是 81 槽），
所以不改编库，而是在 HTTP 层做一次改写：拿到真正要发出去的 header 和 ``f.req``，
按模型规格重写，再交给原来的 stream 方法。
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import uuid
from typing import Any, Dict, Optional

from gemini_webapi.constants import MODEL_HEADER_KEY

from gemini_models import THINK_NORMAL, ModelSpec, build_header_string

logger = logging.getLogger(__name__)

#: 当前请求要用的模型规格（每请求一份，asyncio 下用 ContextVar 隔离）
_active_spec: contextvars.ContextVar[Optional[ModelSpec]] = contextvars.ContextVar("gemini_active_spec", default=None)

#: 需要补槽位时数组要够长（Go 版固定发 99 槽）
_WIRE_MIN_SLOTS = 99


@contextlib.contextmanager
def active_spec(spec: Optional[ModelSpec]):
	"""在上下文里挂上本次请求的模型规格，出口自动还原。"""
	token = _active_spec.set(spec)
	try:
		yield spec
	finally:
		_active_spec.reset(token)


def push_spec(spec: Optional[ModelSpec]):
	"""流式生成器里用：挂上规格并返回 token（配 :func:`pop_spec`）。"""
	return _active_spec.set(spec)


def pop_spec(token) -> None:
	"""还原上一份规格。"""
	_active_spec.reset(token)


def get_session(client):
	"""取库底层真正发 HTTP 的对象（2.0.0 叫 client，2.1.1 叫 _live_client）。"""
	for attr in ("_live_client", "client"):
		session = getattr(client, attr, None)
		if session is not None and hasattr(session, "stream"):
			return session
	return None


def install(client) -> bool:
	"""给一个 GeminiClient 装上线上改写层（幂等）。"""
	session = get_session(client)
	if session is None:
		logger.warning("Gemini wire: 没找到底层 HTTP session，跳过改写层安装")
		return False

	if getattr(session, "_gemini_wire_installed", False):
		return True

	original_stream = session.stream

	def stream(method, url, *args, **kwargs):
		if "StreamGenerate" in str(url):
			try:
				_rewrite_request(kwargs)
			except Exception:
				logger.exception("Gemini wire: 改写 StreamGenerate 请求失败，按原样发送")
		return original_stream(method, url, *args, **kwargs)

	try:
		session.stream = stream
		session._gemini_wire_installed = True
		logger.info("Gemini wire: 已挂上模型改写层 (%s)", type(session).__name__)
		return True
	except Exception as exc:  # 少数实现不允许实例属性赋值
		logger.warning("Gemini wire: 无法直接改写 session.stream (%s)，改用代理对象", exc)
		proxy = _SessionProxy(session, stream)
		for attr in ("_live_client", "client"):
			if getattr(client, attr, None) is session:
				try:
					setattr(client, attr, proxy)
					return True
				except Exception:
					logger.exception("Gemini wire: 代理替换 %s 失败", attr)
		return False


class _SessionProxy:
	"""兜底：包一层，只接管 stream，其余属性转发。"""

	def __init__(self, session, stream):
		self._session = session
		self.stream = stream

	def __getattr__(self, name):
		return getattr(self._session, name)


def _rewrite_request(kwargs: Dict[str, Any]) -> None:
	spec = _active_spec.get()
	if spec is None:
		return

	headers = kwargs.get("headers")
	if headers is not None:
		rewritten = dict(headers)
		rewritten[MODEL_HEADER_KEY] = build_header_string(
			spec.hex_id,
			spec.capacity,
			spec.mode,
			spec.think_wire,
			str(uuid.uuid4()).upper(),
		)
		kwargs["headers"] = rewritten

	data = kwargs.get("data")
	if isinstance(data, dict) and isinstance(data.get("f.req"), str):
		data["f.req"] = _rewrite_freq(data["f.req"], spec)


def _rewrite_freq(freq: str, spec: ModelSpec) -> str:
	"""重写 f.req 里的 inner 数组槽位。解析失败就原样返回。"""
	try:
		outer = json.loads(freq)
		inner = json.loads(outer[1])
	except Exception:
		logger.debug("Gemini wire: f.req 解析失败，跳过槽位改写")
		return freq

	if not isinstance(inner, list):
		return freq

	_apply_slots(inner, spec)

	outer[1] = json.dumps(inner)
	return json.dumps(outer)


def _apply_slots(inner: list, spec: ModelSpec) -> None:
	"""按规格补/改槽位。

	只动模型选择相关的槽，提示词、附件、Gem、临时对话这些库已经填好的槽保持原样。
	"""
	think = spec.think_wire
	needs_extra = spec.mode == 8 or think != THINK_NORMAL
	if needs_extra and len(inner) < _WIRE_MIN_SLOTS:
		inner.extend([None] * (_WIRE_MIN_SLOTS - len(inner)))

	if len(inner) > 79:
		inner[79] = spec.mode
	if len(inner) > 80:
		inner[80] = think
	if len(inner) > 96:
		inner[96] = 1 if think != THINK_NORMAL else 0

	if spec.mode != 8:
		return

	# 档位 8 的必需槽位。inner[49]=40 是让付费号不挂的那一个，
	# 但它会关掉思考档——深度研究（inner[3] 为长 token）自己要用 inner[49]，
	# 那种请求不覆盖。
	deep_research = isinstance(inner[3], str) and inner[3].startswith("!")
	inner[6] = [1]
	inner[30] = []
	if not deep_research:
		inner[49] = 40
	inner[68] = 2
	inner[83] = 1
	inner[91] = 1
	inner[96] = 1
	inner[98] = 1
