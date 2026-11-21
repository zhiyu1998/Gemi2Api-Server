"""多 Gemini Cookie 轮询池。

一份服务同时挂多个 Gemini 账号：请求按「在途请求少的优先」分配到某个账号，
某个账号失败就进冷却（指数退避）并自动换下一个号重试，冷却结束后自动放回池子。

配置来源（按顺序合并，按 1PSID 去重，先出现的优先）：

1. ``SECURE_1PSID`` / ``SECURE_1PSIDTS`` —— 单账号，兼容旧部署，永远排第一位
2. ``GEMINI_COOKIES`` —— JSON 数组，或每行 ``psid,psidts[,label]``
3. ``COOKIES_FILE`` 指向的文件（默认 ``secrets/cookies.txt``，存在才读），格式同 2

最小可用配置示例::

	# 老配置继续可用（单账号）
	SECURE_1PSID = xxx
	SECURE_1PSIDTS = yyy

	# 再多挂两个号
	GEMINI_COOKIES=[{"label":"号2","psid":"xxx","psidts":"yyy"},{"label":"号3","psid":"xxx","psidts":"yyy"}]

	# 或者用文件（每行 psid,psidts[,label]，支持 # 注释）
	COOKIES_FILE=secrets/cookies.txt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from gemini_webapi import GeminiClient

logger = logging.getLogger(__name__)

_PSID_SAFE = re.compile(r"^[\w\-.]+$")
_DELIMITERS = re.compile(r"[,;\t|]")
_COOKIE_PAIR = re.compile(r"([^=;\s]+)=([^;]*)")
_PSID_KEYS = ("__Secure-1PSID", "__Secure-3PSID")
_PSIDTS_KEYS = ("__Secure-1PSIDTS", "__Secure-3PSIDTS")


def split_cookie_bundle(*values: str) -> tuple:
	"""把「浏览器里复制的一整段 Cookie」或分开的 1PSID/1PSIDTS 归一成 (psid, psidts, extras)。

	Google 多账号会话除了 1PSID/1PSIDTS，还要 OSID / __Secure-OSID / SID / SAPISID 这些
	cookie 才能选中正确的登录态；只发前两个会被重定向到 accounts.google.com 登录页，
	表现为「Cookie 保存成功但请求一直挂」。这里把整段都收下来，额外的放进 extras。
	"""
	psid = psidts = ""
	extras: dict = {}
	for raw in values:
		text = (raw or "").strip()
		if not text:
			continue
		if "=" in text:
			for name, value in _COOKIE_PAIR.findall(text):
				name, value = name.strip(), value.strip()
				if name in _PSID_KEYS:
					psid = psid or value
				elif name in _PSIDTS_KEYS:
					psidts = psidts or value
				elif name:
					extras[name] = value
		elif not psid:
			psid = text
		elif not psidts:
			psidts = text
	return psid, psidts, extras


def serialize_extra_cookies(extras: dict) -> str:
	"""把额外 cookie 拼回 ``k=v; k=v`` 存进 .env。"""
	return "; ".join(f"{k}={v}" for k, v in extras.items() if k and v)


def load_extra_cookies() -> dict:
	"""从 ``GEMINI_EXTRA_COOKIES`` 读额外 cookie（整段 Cookie 里除 1PSID/1PSIDTS 之外的部分）。"""
	raw = (os.environ.get("GEMINI_EXTRA_COOKIES") or "").strip()
	if not raw:
		return {}
	extras = {}
	for name, value in _COOKIE_PAIR.findall(raw):
		name, value = name.strip(), value.strip()
		if name and name not in _PSID_KEYS and name not in _PSIDTS_KEYS:
			extras[name] = value
	return extras


def quota_view(client: Any) -> Optional[Dict[str, Any]]:
	"""把 gemini-webapi 客户端的额度/套餐信息压成管理面板能直接渲染的结构。

	``usage_info`` 是新的算力用量口径：tier(FREE/PRO/ULTRA) + 5 小时/每周窗口 +
	AI credits；``quotas`` 是老口径的按模型桶（x/y credits）。两者都收，前端择优显示。
	"""
	if client is None:
		return None

	info = getattr(client, "usage_info", None)
	info = info if isinstance(info, dict) else {}
	tier = info.get("tier") if isinstance(info.get("tier"), dict) else {}

	view: Dict[str, Any] = {
		"tier": tier.get("label"),
		"subscription": info.get("use_overage_ai_credits"),
		"ai_credits_remaining": info.get("ai_credits_remaining"),
		"windows": [],
		"buckets": [],
	}

	for key in ("current_5h", "weekly"):
		window = info.get(key)
		if isinstance(window, dict):
			view["windows"].append(
				{
					"window": window.get("window") or key,
					"remaining": window.get("remaining_credits"),
					"usage_percentage": window.get("usage_percentage"),
					"reset_at": window.get("reset_at"),
				}
			)

	quotas = getattr(client, "quotas", None)
	quotas = quotas if isinstance(quotas, dict) else {}
	for quota_id, item in quotas.items():
		if quota_id == "usage_info" or not isinstance(item, dict):
			continue
		if item.get("remaining") is None and item.get("total") is None:
			continue
		view["buckets"].append(
			{
				"label": item.get("label") or quota_id,
				"remaining": item.get("remaining"),
				"total": item.get("total"),
				"usage_percentage": item.get("usage_percentage"),
				"window": item.get("window"),
			}
		)

	if not view["tier"] and not view["windows"] and not view["buckets"]:
		return None
	return view

# 命中这些字样就认为会话本身失效了，得丢掉客户端重建（而不是只冷却一下）
AUTH_FAILURE_HINTS = (
	"401",
	"403",
	"unauthorized",
	"forbidden",
	"are you signed in",
	"signed in",
	"log in",
	"logged in",
	"login required",
	"session expired",
	"invalid cookie",
)


class CookiePoolUnavailable(RuntimeError):
	"""池里没有任何账号可借（配置为空、全部冷却或全部初始化失败）。"""


def mask_secret(value: str, keep: int = 4) -> str:
	"""把 cookie 之类的东西打码后用于日志和面板展示。"""
	if not value:
		return ""
	if len(value) <= keep * 2:
		return "***"
	return f"{value[:keep]}***{value[-keep:]}"


def mask_email(value: str) -> str:
	"""邮箱脱敏：保留首字符与域名，例如 h***@gmail.com。"""
	email = (value or "").strip()
	if "@" not in email:
		return ""
	local, _, domain = email.partition("@")
	if not local or not domain:
		return ""
	head = local[0]
	return f"{head}***@{domain}"


@dataclass(frozen=True)
class CookieEntry:
	"""一个 Gemini 账号（1PSID + 1PSIDTS）。"""

	psid: str
	psidts: str
	label: str = ""

	@property
	def name(self) -> str:
		"""日志 / 面板里的账号名字。"""
		return self.label or mask_secret(self.psid)


def _entry_from_delimited(line: str) -> CookieEntry:
	"""解析 ``psid,psidts[,label]`` 一行。"""
	parts = [part.strip() for part in _DELIMITERS.split(line) if part.strip()]
	if len(parts) < 2:
		raise ValueError(f"无法解析 Cookie 行（需要 psid,psidts[,label]）：{mask_secret(line)}")
	return CookieEntry(parts[0], parts[1], parts[2] if len(parts) > 2 else "")


def parse_cookie_entries(raw: str) -> List[CookieEntry]:
	"""解析多 Cookie 配置：JSON 数组（对象或字符串）或每行 ``psid,psidts[,label]``。"""
	text = (raw or "").strip()
	if not text:
		return []

	if text[0] in "{[":
		try:
			data = json.loads(text)
		except json.JSONDecodeError as e:
			raise ValueError(f"Cookie 配置不是合法 JSON：{e}") from e

		if isinstance(data, dict):
			# 允许 {"cookies": [...]} / {"accounts": [...]} / 直接一个账号对象
			for key in ("cookies", "accounts", "data"):
				if isinstance(data.get(key), list):
					data = data[key]
					break
			else:
				data = [data]
		if not isinstance(data, list):
			raise ValueError("Cookie 配置的 JSON 根节点必须是数组或含 cookies/accounts 数组的对象")

		entries: List[CookieEntry] = []
		for item in data:
			if isinstance(item, str):
				entries.append(_entry_from_delimited(item))
			elif isinstance(item, dict):
				psid = str(item.get("psid") or item.get("secure_1psid") or item.get("__Secure-1PSID") or "").strip()
				psidts = str(item.get("psidts") or item.get("secure_1psidts") or item.get("__Secure-1PSIDTS") or "").strip()
				label = str(item.get("label") or item.get("name") or "").strip()
				if psid and psidts:
					entries.append(CookieEntry(psid, psidts, label))
				else:
					logger.warning("跳过一条缺少 psid/psidts 的 Cookie 配置：%s", mask_secret(psid))
			else:
				logger.warning("跳过一条类型不支持的 Cookie 配置：%r", type(item).__name__)
		return entries

	entries = []
	for line in text.splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		entries.append(_entry_from_delimited(line))
	return entries


def default_cookie_file() -> str:
	"""默认的 cookie 文件路径（``GEMINI_COOKIE_PATH`` 或项目下 secrets 目录）。"""
	base = os.environ.get("GEMINI_COOKIE_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets")
	return os.path.join(base, "cookies.txt")


def dedupe_entries(entries: Sequence[CookieEntry]) -> List[CookieEntry]:
	"""按 1PSID 去重，先出现的优先（单账号环境变量永远排第一）。"""
	seen: Set[str] = set()
	unique: List[CookieEntry] = []
	for entry in entries:
		if not entry.psid or not entry.psidts or entry.psid in seen:
			continue
		seen.add(entry.psid)
		unique.append(entry)
	return unique


def load_cookie_entries(single: Optional[tuple] = None) -> List[CookieEntry]:
	"""合并环境变量与 cookie 文件，得到本次进程要轮询的账号列表。

	``single`` 传入运行时单账号 (1PSID, 1PSIDTS) 时优先用它，而不是读 os.environ：
	管理面板保存 Cookie 只更新了 main 的模块变量（没写 os.environ），池子按环境变量
	重建会拿到旧值，这条参数就是为了让「面板改完立刻生效」成立。
	"""
	entries: List[CookieEntry] = []

	if single is None:
		psid = (os.environ.get("SECURE_1PSID") or "").strip()
		psidts = (os.environ.get("SECURE_1PSIDTS") or "").strip()
	else:
		psid = (single[0] or "").strip()
		psidts = (single[1] or "").strip()
	if psid and psidts:
		entries.append(CookieEntry(psid, psidts, "environment"))

	try:
		entries.extend(parse_cookie_entries(os.environ.get("GEMINI_COOKIES", "")))
	except ValueError as e:
		logger.error("GEMINI_COOKIES 解析失败，已跳过：%s", e)

	cookie_file = (os.environ.get("COOKIES_FILE") or "").strip() or default_cookie_file()
	if os.path.exists(cookie_file):
		try:
			entries.extend(parse_cookie_entries(Path(cookie_file).read_text(encoding="utf-8")))
		except Exception as e:
			logger.error("读取 COOKIES_FILE=%s 失败，已跳过：%s", cookie_file, e)

	return dedupe_entries(entries)


def looks_like_auth_failure(message: str) -> bool:
	"""返回这句报错是否意味着会话失效（需要丢客户端重建）。"""
	lowered = (message or "").lower()
	return any(hint in lowered for hint in AUTH_FAILURE_HINTS)


async def _safe_close(client: Any) -> None:
	try:
		await client.close()
	except Exception:
		pass


def _schedule_close(client: Any) -> None:
	"""在事件循环里异步关掉客户端，别让失败路径变成串行阻塞。"""
	try:
		loop = asyncio.get_running_loop()
	except RuntimeError:
		return
	loop.create_task(_safe_close(client))


class ClientSlot:
	"""一个账号的运行时状态：客户端 + 在途数 + 冷却计时。"""

	def __init__(self, entry: CookieEntry, index: int):
		self.entry = entry
		self.index = index
		self.client: Optional[GeminiClient] = None
		self.init_lock = asyncio.Lock()
		self.inflight = 0
		self.successes = 0
		self.failures = 0
		self.cooldown_until = 0.0
		self.last_used = 0.0
		self.last_error = ""

	def available(self, now: Optional[float] = None) -> bool:
		return (now if now is not None else time.time()) >= self.cooldown_until


class CookieLease:
	"""一次账号占用。用完 ``release()``；也可以当上下文管理器自动上报成功/失败。"""

	def __init__(self, pool: "CookiePool", slot: ClientSlot, client: GeminiClient):
		self._pool = pool
		self.slot = slot
		self.client = client
		self._released = False

	@property
	def psid(self) -> str:
		return self.slot.entry.psid

	@property
	def label(self) -> str:
		return self.slot.entry.name

	def succeed(self) -> None:
		self._pool.report_success(self.slot)

	def fail(self, error: BaseException) -> None:
		self._pool.report_failure(self.slot, error)

	def release(self) -> None:
		"""归还账号（幂等）。"""
		if self._released:
			return
		self._released = True
		self.slot.inflight = max(0, self.slot.inflight - 1)

	async def __aenter__(self) -> "CookieLease":
		return self

	async def __aexit__(self, exc_type, exc, tb) -> bool:
		if exc is not None:
			self.fail(exc)
		else:
			self.succeed()
		self.release()
		return False


class CookiePool:
	"""多账号轮询池。

	分配策略：冷却中的账号跳过；其余按（在途请求数, 上次使用时间）排序取最小，
	所以空闲账号天然被轮着用，慢账号也不会被压满。
	"""

	def __init__(
		self,
		entries: Sequence[CookieEntry],
		*,
		install_shim: Optional[Callable[[GeminiClient], Any]] = None,
		validator: Optional[Callable[[GeminiClient, str], Any]] = None,
		psidts_cache: Optional[Callable[[str], str]] = None,
		extra_cookies: Optional[Dict[str, str]] = None,
		max_concurrency: int = 0,
		cooldown_base: float = 15.0,
		cooldown_max: float = 300.0,
		init_timeout: float = 300.0,
		degrade_after: int = 3,
	):
		self.slots: List[ClientSlot] = [ClientSlot(entry, index) for index, entry in enumerate(entries)]
		self.install_shim = install_shim
		self.validator = validator
		self.psidts_cache = psidts_cache
		self.extra_cookies = {k: v for k, v in (extra_cookies or {}).items() if k and v}
		self.max_concurrency = max(0, int(max_concurrency))
		self.cooldown_base = max(1.0, float(cooldown_base))
		self.cooldown_max = max(self.cooldown_base, float(cooldown_max))
		self.init_timeout = float(init_timeout)
		# 连续失败这么多轮的账号降级为「兜底号」：只有别的号都不可用才轮到它
		self.degrade_after = max(1, int(degrade_after))

	def __len__(self) -> int:
		return len(self.slots)

	@property
	def retry_budget(self) -> int:
		"""单次请求最多能试几个账号（每个账号最多试一次）。"""
		return max(1, len(self.slots))

	@property
	def labels(self) -> List[str]:
		return [slot.entry.name for slot in self.slots]

	def _busy(self, slot: ClientSlot) -> int:
		"""在途请求数 + 是否正在初始化：初始化可能要几十秒，别让并发请求排在它后面干等。"""
		return slot.inflight + (1 if slot.init_lock.locked() else 0)

	def _pick(self, exclude: Set[str]) -> Optional[ClientSlot]:
		now = time.time()
		candidates = [slot for slot in self.slots if slot.entry.psid not in exclude and slot.available(now)]
		if not candidates:
			return None
		healthy = [slot for slot in candidates if slot.failures < self.degrade_after]
		if healthy:
			candidates = healthy
		if self.max_concurrency > 0:
			below_cap = [slot for slot in candidates if self._busy(slot) < self.max_concurrency]
			if below_cap:
				candidates = below_cap
		candidates.sort(key=lambda slot: (self._busy(slot), slot.last_used, slot.index))
		return candidates[0]

	async def acquire(self, exclude: Optional[Set[str]] = None) -> CookieLease:
		"""借一个可用账号；一个都借不到就抛 CookiePoolUnavailable。"""
		if not self.slots:
			raise CookiePoolUnavailable("没有配置任何 Gemini 账号（SECURE_1PSID 或 GEMINI_COOKIES）")

		skipped: Set[str] = set(exclude or set())
		errors: List[str] = []
		for _ in range(len(self.slots)):
			slot = self._pick(skipped)
			if slot is None:
				break
			skipped.add(slot.entry.psid)
			try:
				client = await self._ensure_client(slot)
			except Exception as e:
				errors.append(f"{slot.entry.name}: {e}")
				# 初始化/校验失败 = 这份 cookie 有问题，直接降级长冷却
				self.report_failure(slot, e, permanent=True)
				continue
			slot.inflight += 1
			slot.last_used = time.time()
			return CookieLease(self, slot, client)

		detail = "；".join(errors) if errors else "全部账号处于冷却或无可用账号"
		raise CookiePoolUnavailable(f"没有可用的 Gemini 账号：{detail}")

	async def _psidts_attempts(self, slot: ClientSlot) -> List[tuple]:
		"""先试配置里的 1PSIDTS，再试缓存（去重）。

		顺序不能反：缓存文件是历史遗留的只读产物（没有写入方），用户重新贴了新 Cookie
		却先拿旧缓存去 init，503/401 全挂在旧值上，表现就是「新 cookie 明明是对的却报过期」。
		"""
		attempts: List[tuple] = []
		if slot.entry.psidts:
			attempts.append(("configuration", slot.entry.psidts))

		cached = ""
		if self.psidts_cache is not None:
			try:
				cached = (self.psidts_cache(slot.entry.psid) or "").strip()
			except Exception as e:
				logger.warning("读取账号 %s 的 1PSIDTS 缓存失败：%s", slot.entry.name, e)
		if cached and cached != slot.entry.psidts:
			attempts.append(("cache", cached))
		return attempts

	async def _ensure_client(self, slot: ClientSlot) -> GeminiClient:
		"""拿到（必要时初始化）某个账号的客户端，并发下只初始化一次。"""
		if slot.client is not None:
			return slot.client

		async with slot.init_lock:
			if slot.client is not None:
				return slot.client

			attempts = await self._psidts_attempts(slot)
			if not attempts:
				raise CookiePoolUnavailable(f"账号 {slot.entry.name} 没有可用的 1PSIDTS")

			last_error: Optional[BaseException] = None
			for source, psidts in attempts:
				client = None
				try:
					logger.info("初始化账号 %s（%s 1PSIDTS）", slot.entry.name, source)
					client = GeminiClient(slot.entry.psid, psidts)
					if self.extra_cookies:
						# 整段 Cookie 里的 OSID / __Secure-OSID / SID 等：多账号会话靠它们选中登录态
						client.cookies = self.extra_cookies
					await client.init(timeout=self.init_timeout)
					if self.install_shim is not None:
						self.install_shim(client)
					if self.validator is not None:
						await self.validator(client, f"cookie-pool:{slot.entry.name}:{source}")
					slot.client = client
					slot.cooldown_until = 0.0
					slot.last_error = ""
					logger.info("账号 %s 初始化成功（%s 1PSIDTS）", slot.entry.name, source)
					return client
				except Exception as e:
					last_error = e
					logger.warning("账号 %s 用 %s 1PSIDTS 初始化失败：%s", slot.entry.name, source, e)
					if client is not None:
						await _safe_close(client)

			raise CookiePoolUnavailable(f"账号 {slot.entry.name} 初始化失败：{last_error}") from last_error

	async def warm_up(self) -> None:
		"""启动时预热所有账号（每个账号一次 init，可选一次校验）。"""
		for slot in self.slots:
			try:
				await self._ensure_client(slot)
			except Exception as e:
				logger.warning("预热账号 %s 失败：%s", slot.entry.name, e)

	def report_success(self, slot: ClientSlot) -> None:
		"""这个账号这次成功了：清空失败计数和冷却。"""
		slot.successes += 1
		slot.failures = 0
		slot.cooldown_until = 0.0
		slot.last_error = ""

	def report_failure(self, slot: ClientSlot, error: BaseException, permanent: bool = False) -> None:
		"""这个账号这次失败了：指数退避冷却，会话失效时丢掉客户端下次重建。

		``permanent=True`` 用于「连客户端都建不起来」的情况（cookie 本身坏了）：
		直接打到降级阈值并给足冷却，避免每轮请求都白花一次初始化和校验的时间。
		"""
		slot.failures += 1
		slot.last_error = str(error)[:300]
		if permanent:
			slot.failures = max(slot.failures, self.degrade_after)
		delay = min(self.cooldown_max, self.cooldown_base * (2 ** (slot.failures - 1)))
		if permanent:
			delay = self.cooldown_max
		slot.cooldown_until = time.time() + delay

		client = slot.client
		dropped = ""
		if client is not None and looks_like_auth_failure(slot.last_error):
			slot.client = None
			dropped = "，已丢弃会话待重建"
			_schedule_close(client)

		logger.warning(
			"账号 %s 失败第 %s 次，冷却 %.0fs%s（%s）",
			slot.entry.name,
			slot.failures,
			delay,
			dropped,
			slot.last_error,
		)

	def primary_client(self) -> Optional[GeminiClient]:
		"""取一个现成的客户端（不发请求、不初始化），图片代理这类旁路调用用。"""
		for slot in self.slots:
			if slot.client is not None:
				return slot.client
		return None

	def snapshot(self) -> List[Dict[str, Any]]:
		"""给管理面板用的账号状态快照。"""
		now = time.time()
		rows: List[Dict[str, Any]] = []
		for slot in self.slots:
			rows.append(
				{
					"index": slot.index,
					"label": slot.entry.name,
					"psid_masked": mask_secret(slot.entry.psid),
					"initialized": slot.client is not None,
					"inflight": slot.inflight,
					"successes": slot.successes,
					"failures": slot.failures,
					"cooling_down_seconds": round(max(0.0, slot.cooldown_until - now), 1),
					"last_error": slot.last_error,
					"quota": quota_view(slot.client),
				}
			)
		return rows

	async def close(self) -> None:
		"""关掉所有账号的客户端。"""
		for slot in self.slots:
			client = slot.client
			slot.client = None
			if client is not None:
				try:
					await client.close()
				except Exception as e:
					logger.warning("关闭账号 %s 的客户端失败：%s", slot.entry.name, e)
