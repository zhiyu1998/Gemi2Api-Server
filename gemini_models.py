"""Gemini 模型目录 / 解析（含 Gemini 4 系列）。

gemini-webapi 的 ``Model`` 枚举是历史遗留：里面写死的 hex 一旦 Google 换名或提权就过期，
而且根本没有 4 系列。这里改成一张显式的模型表：

* ``hex_id``   —— ``x-goog-ext-525001261-jspb`` 里第 4 位的模型开关，服务端只认它；
* ``mode``     —— 请求头下标 14 / ``inner[79]`` 的档位，8 才真正选 4 Carbon/Argon，3 选 4 Neon；
* ``think``    —— 请求头下标 15 / ``inner[80]`` 的思考档：1 普通、2 扩展(High)、4 中档；
* ``capacity`` —— 请求头下标 11 的账号档位尾值。

命名对齐 gemini-web2api-go：4 Hydrogen 与 3.8 Flash 同一个 hex（上游原地升级），
4 Argon 被上游改名成 4 Carbon，两条名字都认；4 Neon 接替 3.1 Pro 的位置（同档位 3），
``gemini-3.1-pro`` 是它的旧名字。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from gemini_webapi.constants import MODEL_HEADER_KEY

# ---------------------------------------------------------------------------
# 上游模型 hex（服务端唯一认的模型开关）
# ---------------------------------------------------------------------------
HEX_FLASH_36 = "fbb127bbb056c959"  # 3.6 Flash
HEX_FLASH_LITE_35 = "cf41b0e0dd7d53e5"  # 3.5 Flash-Lite
HEX_PRO_31 = "9d8ca3786ebdfbea"  # 3.1 Pro 的旧位置，现由 4 Neon 接替（同档位 3）
HEX_FLASH_38 = "56fdd199312815e2"  # 3.8 Flash，原地升级后显示为 4 Hydrogen
HEX_HYDROGEN_4 = HEX_FLASH_38
HEX_ARGON_4 = "95a4c0abcba24364"  # 4 Argon → 上游改名为 4 Carbon
# 4 Neon：otAQ7b 清单里的主 hex，短名 Neon、代号 Neon4p0、网页描述「Deeper analysis」。
# 真正选中它的是**档位 3**（不是 hex）：2026-11-19 Ultra 号实测 mode 3 回报 "4 Neon"，
# 高思考档回报 "4 Neon High" 并带思考链；免费号发 mode 3 会挂到超时。
# 3.1 Pro 的位置就是被它接替的，所以 HEX_PRO_31 和它落到同一条路由（同 mode 3）。
HEX_NEON_4 = "e6fa609c3fa255c0"

# ---------------------------------------------------------------------------
# 思考档线值（请求头下标 15 / inner[80]）
# ---------------------------------------------------------------------------
THINK_NORMAL = 1
THINK_EXTENDED = 2
THINK_MEDIUM = 4

#: 模型名后缀 → 思考档。空后缀取模型自身默认档。
TIER_SUFFIXES: Dict[str, int] = {
	"": 0,  # 用模型默认档
	"-low": THINK_NORMAL,
	"-medium": THINK_MEDIUM,
	"-high": THINK_EXTENDED,
	"-thinking": THINK_EXTENDED,
	"-thinking-medium": THINK_MEDIUM,
	"-thinking-high": THINK_EXTENDED,
	"-thinking-low": THINK_NORMAL,
	"-extended": THINK_EXTENDED,
}

#: 客户端传的 reasoning_effort → 思考档线值（对齐网页 UI 的低/中/高）。
#: OpenAI 规范只定义了 minimal/low/medium/high，这里把常见的同义词一并认下。
EFFORT_TO_THINK: Dict[str, int] = {
	"minimal": THINK_NORMAL,
	"low": THINK_NORMAL,
	"fast": THINK_NORMAL,
	"medium": THINK_MEDIUM,
	"mid": THINK_MEDIUM,
	"deep": THINK_MEDIUM,
	"high": THINK_EXTENDED,
	"deeper": THINK_EXTENDED,
	# 直接给线值也认：1 普通、4 中档、2 扩展
	"1": THINK_NORMAL,
	"2": THINK_EXTENDED,
	"4": THINK_MEDIUM,
}


def think_from_effort(*values) -> int:
	"""把客户端提交的思考档参数翻译成线值；都没给返回 0（用模型默认档）。"""
	for value in values:
		if value is None:
			continue
		key = str(value).strip().lower()
		if key in EFFORT_TO_THINK:
			return EFFORT_TO_THINK[key]
	return 0


# 3.1 Pro 的位置 2026-11-19 起换成了 4 Neon（同档位 3），旧名字不再算下线，
# 否则老配置请求 3.1 Pro 会被 400 拦掉，等于把 Neon 一起拦在门外。


@dataclass(frozen=True)
class ModelSpec:
	"""一条模型路由的完整描述。"""

	name: str
	hex_id: str
	mode: int = 1
	think: int = THINK_NORMAL
	capacity: int = 1
	display: str = ""
	description: str = ""
	aliases: tuple = field(default_factory=tuple)
	needs_login: bool = False
	retired: str = ""

	@property
	def think_wire(self) -> int:
		"""思考档线值：显式指定优先，其余按普通/扩展推。"""
		return self.think or THINK_NORMAL

	@property
	def extended_thinking(self) -> bool:
		"""是否为扩展思考档（档值 2）。中档 4 服务端内部思考、不吐思考链。"""
		return self.think_wire == THINK_EXTENDED

	def with_tier(self, think: int, suffix: str) -> "ModelSpec":
		"""返回同一路由换了思考档的副本。"""
		return ModelSpec(
			name=self.name + suffix,
			hex_id=self.hex_id,
			mode=self.mode,
			think=think or self.think,
			capacity=self.capacity,
			display=self.display,
			description=self.description,
			aliases=self.aliases,
			needs_login=self.needs_login,
			retired=self.retired,
		)


# ---------------------------------------------------------------------------
# 内置模型表
# ---------------------------------------------------------------------------
BUILTIN_MODELS: Dict[str, ModelSpec] = {
	# --- Gemini 4 -----------------------------------------------------------
	"gemini-4-hydrogen": ModelSpec(
		name="gemini-4-hydrogen",
		hex_id=HEX_HYDROGEN_4,
		mode=1,
		capacity=2,
		display="4 Hydrogen",
		description="4 Hydrogen：Flash 路由，登录号（含免费号）可用",
		aliases=("gemini-4", "gemini-4-flash", "4-hydrogen", "hydrogen"),
		needs_login=True,
	),
	"gemini-4-carbon": ModelSpec(
		name="gemini-4-carbon",
		hex_id=HEX_ARGON_4,
		mode=8,
		capacity=2,
		display="4 Carbon",
		description="4 Carbon（原 4 Argon）：最复杂的活，付费/Ultra 号专属，免费号会被降级",
		aliases=(
			"gemini-4-argon",
			"gemini-4-pro",
			"gemini-4-ultra",
			"4-carbon",
			"4-argon",
			"carbon",
			"argon",
		),
		needs_login=True,
	),
	"gemini-4-neon": ModelSpec(
		name="gemini-4-neon",
		hex_id=HEX_NEON_4,
		mode=3,
		capacity=2,
		display="4 Neon",
		description="4 Neon（Deeper analysis）：接替 3.1 Pro 的位置，付费/Ultra 号专属，免费号会挂到超时",
		aliases=("neon", "4-neon", "gemini-4-deep"),
		needs_login=True,
	),
	# --- 3.x 保留（老配置还在用） -------------------------------------------
	"gemini-3.8-flash": ModelSpec(
		name="gemini-3.8-flash",
		hex_id=HEX_FLASH_38,
		mode=1,
		capacity=2,
		display="3.8 Flash",
		description="3.8 Flash：与 4 Hydrogen 同一条路由（上游原地升级）",
		aliases=("gemini-3.7-flash", "3.8-flash", "3.7-flash"),
		needs_login=True,
	),
	"gemini-3.6-flash": ModelSpec(
		name="gemini-3.6-flash",
		hex_id=HEX_FLASH_36,
		mode=1,
		capacity=1,
		display="3.6 Flash",
		description="3.6 Flash：匿名时是 3.5 Flash-Lite，登录后是 4 Hydrogen",
		aliases=("3.6-flash",),
	),
	"gemini-3.5-flash-lite": ModelSpec(
		name="gemini-3.5-flash-lite",
		hex_id=HEX_FLASH_LITE_35,
		mode=6,
		capacity=1,
		display="3.5 Flash-Lite",
		description="3.5 Flash-Lite：最快最轻，匿名可用",
		aliases=("gemini-flash-lite", "3.5-flash-lite", "flash-lite", "lite"),
	),
	"gemini-3.1-pro": ModelSpec(
		name="gemini-3.1-pro",
		hex_id=HEX_PRO_31,
		mode=3,
		capacity=1,
		display="3.1 Pro",
		description="3.1 Pro 的旧名称：上游已把这一档换成 4 Neon，跟 gemini-4-neon 同一条路由",
		aliases=("3.1-pro",),
		needs_login=True,
	),
}

#: 别名 → 规范模型名（大小写不敏感）
ALIASES: Dict[str, str] = {}
for _key, _spec in BUILTIN_MODELS.items():
	ALIASES[_key.lower()] = _key
	for _alias in _spec.aliases:
		ALIASES.setdefault(_alias.lower(), _key)

#: 关键词兜底：客户端乱传模型名时按关键词挑一个，避免整条链路 500
_KEYWORD_ROUTES: tuple = (
	(("hydrogen", "flash", "fast"), "gemini-4-hydrogen"),
	(("neon", "deeper"), "gemini-4-neon"),
	(("carbon", "argon", "pro", "ultra", "reason"), "gemini-4-carbon"),
	(("lite", "light", "mini"), "gemini-3.5-flash-lite"),
)

_MODEL_PREFIX_RE = re.compile(r"^(?:models/)?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 请求头构造
# ---------------------------------------------------------------------------
def build_header_string(hex_id: str, capacity: int, mode: int, think: int, session_id: str) -> str:
	"""构造 ``x-goog-ext-525001261-jspb`` 的线值。

	与 gemini-webapi 的 ``build_model_header`` 前 15 位一致（下标 14 = 档位），
	再补下标 15 思考档、16 会话 uuid，也就是浏览器抓包里的最终形状。
	"""
	return (
		f'[1,null,null,null,"{hex_id}",null,null,0,[4,5,6,8],null,null,{capacity},'
		f'null,null,{mode},{think},"{session_id}"]'
	)


def build_selection_header(hex_id: str, capacity: int, mode: int) -> str:
	"""构造选择档位用的 15 位 header（下标 14 = 档位）。

	库自己会往后追加思考档和会话 id，追加出来的位置正好是 15/16；
	真正发出去的线值由 :mod:`gemini_wire` 统一改写，不受库版本影响。
	"""
	return f'[1,null,null,null,"{hex_id}",null,null,0,[4,5,6,8],null,null,{capacity},null,null,{mode}]'


def build_model_arg(spec: ModelSpec) -> dict:
	"""把模型规格转成 gemini-webapi 接受的 dict 模型参数。"""
	header = build_selection_header(spec.hex_id, spec.capacity, spec.mode)
	return {
		"model_name": spec.name,
		"model_header": {
			MODEL_HEADER_KEY: header,
			"x-goog-ext-73010989-jspb": "[0]",
			"x-goog-ext-73010990-jspb": "[0,0,0]",
		},
		"capacity": spec.capacity,
		"capacity_field": 12,
		"display_name": spec.display or spec.name,
		"description": spec.description or spec.name,
	}


def parse_header_spec(header_value: str, name: str = "") -> Optional[ModelSpec]:
	"""从一条已有 header 里反推模型规格（自定义模型兼容用）。

	兼容老格式：下标 4 是 hex，11 是账号档位，14 是档位，15 是思考档，
	后面还可能有 uuid / ``["boq_..."]`` / 时间戳这类浏览器抓包残留，一律忽略。
	"""
	if not header_value:
		return None
	try:
		parsed = json.loads(header_value)
	except Exception:
		return None
	if not isinstance(parsed, list) or len(parsed) < 5 or not isinstance(parsed[4], str):
		return None

	hex_id = parsed[4]

	def _int_at(index: int, default: int) -> int:
		if len(parsed) > index and isinstance(parsed[index], int):
			return parsed[index]
		return default

	return ModelSpec(
		name=name or hex_id,
		hex_id=hex_id,
		mode=_int_at(14, 1),
		think=_int_at(15, THINK_NORMAL),
		capacity=_int_at(11, 1),
		display=name or hex_id,
	)


# ---------------------------------------------------------------------------
# 名称解析
# ---------------------------------------------------------------------------
def normalize_model_name(raw: str) -> str:
	"""去掉 ``models/`` 前缀、``@think=N`` 尾巴和首尾空白。"""
	name = (raw or "").strip()
	name = _MODEL_PREFIX_RE.sub("", name)
	if "@think=" in name:
		name = name.split("@think=", 1)[0]
	return name.strip().lower().replace("_", "-")


def split_tier(name: str) -> tuple:
	"""把 ``gemini-4-carbon-thinking-medium`` 拆成 (基础名, 思考档, 后缀)。"""
	for suffix in sorted((s for s in TIER_SUFFIXES if s), key=len, reverse=True):
		if name.endswith(suffix):
			return name[: -len(suffix)], TIER_SUFFIXES[suffix], suffix
	return name, 0, ""


def resolve_builtin(name: str) -> Optional[ModelSpec]:
	"""在内置模型表里解析模型名（支持档位后缀与别名）。"""
	if not name:
		return None

	base, tier, suffix = split_tier(name)
	canonical = ALIASES.get(name) or ALIASES.get(base)
	if canonical:
		spec = BUILTIN_MODELS[canonical]
		if tier:
			return spec.with_tier(tier, suffix)
		return spec

	# 别名本身带档位，例如 gemini-4-argon-high
	if tier:
		canonical = ALIASES.get(base)
		if canonical:
			return BUILTIN_MODELS[canonical].with_tier(tier, suffix)
	return None


def fuzzy_builtin(name: str) -> Optional[ModelSpec]:
	"""关键词兜底：认不出确切名字时按关键词挑一条路由。"""
	lowered = name.lower()
	for keywords, target in _KEYWORD_ROUTES:
		if any(keyword in lowered for keyword in keywords):
			return BUILTIN_MODELS[target]
	return None


def list_builtin_specs(include_retired: bool = False) -> List[ModelSpec]:
	"""列出内置模型（默认去掉已下线的）。"""
	specs = []
	for spec in BUILTIN_MODELS.values():
		if spec.retired and not include_retired:
			continue
		specs.append(spec)
	return specs


def tier_names(spec: ModelSpec) -> Iterable[str]:
	"""给一个路由生成对外暴露的档位名（普通 / High / Medium）。"""
	yield spec.name
	yield spec.name + "-thinking"
	yield spec.name + "-thinking-medium"
