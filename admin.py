"""
Gemi2Api Server 管理面板后端
提供状态监控、配置管理、日志查看等功能
"""

import hashlib
import importlib.metadata
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


def _get_version() -> str:
    try:
        return importlib.metadata.version("gemi2api-server")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# 创建路由器
router = APIRouter(prefix="/admin", tags=["admin"])

# 全局状态跟踪
_start_time = time.time()
_request_log = deque(maxlen=100)  # 保留最近100条日志


def mask_cookie(value: str) -> str:
	"""对 Cookie 值进行脱敏显示：前4位 + *** + 后4位"""
	if not value or len(value) <= 8:
		return value or ""
	return value[:4] + "***" + value[-4:]


_stats = {
	"total_requests": 0,
	"error_count": 0,
	"total_response_time": 0.0,
}

# 管理面板会话存储 { token: expire_timestamp }
_admin_sessions = {}
SESSION_EXPIRE_HOURS = 12

# 环境变量路径
ENV_FILE = Path(__file__).parent / ".env"


class ConfigUpdate(BaseModel):
	"""配置更新请求"""

	host: Optional[str] = None
	port: Optional[int] = None
	api_key: Optional[str] = None
	gem_id: Optional[str] = None
	feature: Optional[str] = None
	enabled: Optional[bool] = None


def log_request(method: str, path: str, status: int, response_time: float = 0):
	"""记录请求日志"""
	_stats["total_requests"] += 1
	_stats["total_response_time"] += response_time

	if status >= 400:
		_stats["error_count"] += 1

	_request_log.appendleft(
		{
			"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"method": method,
			"path": path,
			"status": status,
			"response_time": round(response_time * 1000, 2),
		}
	)


def format_uptime(seconds: float) -> str:
	"""格式化运行时间"""
	days = int(seconds // 86400)
	hours = int((seconds % 86400) // 3600)
	minutes = int((seconds % 3600) // 60)

	if days > 0:
		return f"{days}天 {hours}小时"
	elif hours > 0:
		return f"{hours}小时 {minutes}分钟"
	else:
		return f"{minutes}分钟"


def read_env() -> dict:
	"""读取 .env 文件"""
	env_vars = {}
	if ENV_FILE.exists():
		with open(ENV_FILE, "r") as f:
			for line in f:
				line = line.strip()
				if line and not line.startswith("#") and "=" in line:
					key, _, value = line.partition("=")
					env_vars[key.strip()] = value.strip().strip('"').strip("'")
	return env_vars


def write_env(updates: dict):
	"""更新 .env 文件"""
	env_vars = read_env()
	env_vars.update(updates)

	with open(ENV_FILE, "w") as f:
		for key, value in env_vars.items():
			f.write(f'{key}="{value}"\n')


@router.get("/", response_class=HTMLResponse)
async def admin_page():
	"""返回管理面板页面"""
	html_path = Path(__file__).parent / "templates" / "admin.html"
	if html_path.exists():
		return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
	return HTMLResponse(content="<h1>管理面板文件未找到</h1>", status_code=404)


class LoginRequest(BaseModel):
	"""登录请求"""

	api_key: str


def _generate_token(api_key: str) -> str:
	"""生成会话 token"""
	raw = f"{api_key}:{time.time()}:{os.urandom(16).hex()}"
	return hashlib.sha256(raw.encode()).hexdigest()


def _clean_expired_sessions():
	"""清理过期会话"""
	now = time.time()
	expired = [t for t, exp in _admin_sessions.items() if exp < now]
	for t in expired:
		del _admin_sessions[t]


@router.post("/api/login")
async def admin_login(req: LoginRequest):
	"""管理面板登录验证"""
	from main import API_KEY

	# 未配置 API_KEY 时不允许登录
	if not API_KEY:
		raise HTTPException(status_code=400, detail="未配置 API_KEY，管理面板不可用。请在 .env 中设置 API_KEY 后重启服务。")

	if req.api_key != API_KEY:
		raise HTTPException(status_code=401, detail="API_KEY 无效")

	# 生成 token，12小时有效
	_clean_expired_sessions()
	token = _generate_token(req.api_key)
	_admin_sessions[token] = time.time() + SESSION_EXPIRE_HOURS * 3600

	return {
		"success": True,
		"token": token,
		"expires_in": SESSION_EXPIRE_HOURS * 3600,
		"message": f"登录成功，会话有效期 {SESSION_EXPIRE_HOURS} 小时",
	}


@router.get("/api/check")
async def admin_check(token: str):
	"""检查会话是否有效"""
	_clean_expired_sessions()

	if token not in _admin_sessions:
		raise HTTPException(status_code=401, detail="会话无效或已过期")

	expire_at = _admin_sessions[token]
	remaining = int(expire_at - time.time())

	return {"valid": True, "remaining_seconds": remaining, "remaining_hours": round(remaining / 3600, 1)}


async def verify_admin_token(request: Request):
	"""验证管理面板 token 的依赖注入"""
	token = request.headers.get("X-Admin-Token") or request.query_params.get("token")
	if not token:
		raise HTTPException(status_code=401, detail="缺少管理面板 token")

	_clean_expired_sessions()
	if token not in _admin_sessions:
		raise HTTPException(status_code=401, detail="会话无效或已过期")

	return token


@router.get("/api/status")
async def get_status(token: str = Depends(verify_admin_token)):
	"""获取服务状态"""
	from main import API_KEY, AUTO_DELETE_CHAT, ENABLE_THINKING, GEM_ID, HOST, PORT, SECURE_1PSID, SECURE_1PSIDTS, TEMPORARY_CHAT

	# 检查 cookie 是否有效（简单检查是否存在）
	cookie_valid = bool(SECURE_1PSID and SECURE_1PSIDTS)

	# 计算平均响应时间
	avg_response_time = 0
	if _stats["total_requests"] > 0:
		avg_response_time = round(_stats["total_response_time"] / _stats["total_requests"] * 1000, 2)

	# 计算错误率
	error_rate = 0
	if _stats["total_requests"] > 0:
		error_rate = round(_stats["error_count"] / _stats["total_requests"] * 100, 1)

	return {
		"running": True,
		"uptime": format_uptime(time.time() - _start_time),
		"total_requests": _stats["total_requests"],
		"avg_response_time": avg_response_time,
		"error_rate": error_rate,
		"host": HOST,
		"port": PORT,
		"api_key_enabled": bool(API_KEY),
		"cookie_valid": cookie_valid,
		"secure_1psid_masked": mask_cookie(SECURE_1PSID),
		"secure_1psidts_masked": mask_cookie(SECURE_1PSIDTS),
		"thinking_enabled": ENABLE_THINKING,
		"temporary_chat": TEMPORARY_CHAT,
		"auto_delete_chat": AUTO_DELETE_CHAT,
		"gem_id": GEM_ID,
		"version": _get_version(),
		"start_time": datetime.fromtimestamp(_start_time).strftime("%Y-%m-%d %H:%M:%S"),
	}


@router.get("/api/logs")
async def get_logs(token: str = Depends(verify_admin_token)):
	"""获取最近的日志"""
	return {"logs": list(_request_log)}


@router.post("/api/config")
async def update_config(config: ConfigUpdate, token: str = Depends(verify_admin_token)):
	"""更新配置"""

	# 更新功能开关
	if config.feature and config.enabled is not None:
		env_key = None
		if config.feature == "thinking":
			env_key = "ENABLE_THINKING"
		elif config.feature == "temporary":
			env_key = "TEMPORARY_CHAT"
		elif config.feature == "autoDelete":
			env_key = "AUTO_DELETE_CHAT"

		if env_key:
			write_env({env_key: str(config.enabled).lower()})
			# 更新运行时变量
			if config.feature == "thinking":
				import main

				main.ENABLE_THINKING = config.enabled
			elif config.feature == "temporary":
				import main

				main.TEMPORARY_CHAT = config.enabled
			elif config.feature == "autoDelete":
				import main

				main.AUTO_DELETE_CHAT = config.enabled
			return {"success": True, "message": f"功能 {config.feature} 已{'启用' if config.enabled else '禁用'}"}

	# 更新网络配置
	if config.host or config.port:
		updates = {}
		if config.host:
			updates["HOST"] = config.host
		if config.port:
			updates["PORT"] = str(config.port)
		if config.api_key is not None:
			updates["API_KEY"] = config.api_key
		write_env(updates)
		return {"success": True, "message": "配置已保存，重启服务后生效"}

	# 更新 API_KEY
	if config.api_key is not None:
		write_env({"API_KEY": config.api_key})
		return {"success": True, "message": "API_KEY 已更新"}

	raise HTTPException(status_code=400, detail="无效的配置请求")


@router.post("/api/config-save-restart")
async def save_config_and_restart(config: ConfigUpdate, token: str = Depends(verify_admin_token)):
	"""保存配置并重启服务（一步完成）"""
	env_updates = {}

	if config.host is not None:
		env_updates["HOST"] = config.host
	if config.port is not None:
		env_updates["PORT"] = str(config.port)
	if config.api_key is not None:
		env_updates["API_KEY"] = config.api_key
	if config.gem_id is not None:
		env_updates["GEM_ID"] = config.gem_id

	if env_updates:
		write_env(env_updates)

	# 重启服务
	try:
		python_path = sys.executable
		script_path = os.path.abspath(__file__).replace("admin.py", "main.py")

		subprocess.Popen(
			[python_path, script_path],
			cwd=os.path.dirname(script_path),
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)

		import main

		main.os._exit(0)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"重启失败: {str(e)}")


@router.post("/api/restart")
async def restart_service(token: str = Depends(verify_admin_token)):
	"""重启服务"""
	try:
		# 获取当前进程的命令行参数
		python_path = sys.executable
		script_path = os.path.abspath(__file__).replace("admin.py", "main.py")

		# 启动新进程
		subprocess.Popen(
			[python_path, script_path],
			cwd=os.path.dirname(script_path),
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)

		# 终止当前进程
		os._exit(0)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"重启失败: {str(e)}")


class CookieUpdate(BaseModel):
	"""Cookie 更新请求"""

	secure_1psid: str
	secure_1psidts: str


@router.post("/api/cookies")
async def update_cookies(cookies: CookieUpdate, token: str = Depends(verify_admin_token)):
	"""更新 Gemini Cookie"""
	if not cookies.secure_1psid or not cookies.secure_1psidts:
		raise HTTPException(status_code=400, detail="Cookie 值不能为空")

	# 保存到 .env 文件
	write_env(
		{
			"SECURE_1PSID": cookies.secure_1psid,
			"SECURE_1PSIDTS": cookies.secure_1psidts,
		}
	)

	# 更新运行时变量
	import main

	main.SECURE_1PSID = cookies.secure_1psid
	main.SECURE_1PSIDTS = cookies.secure_1psidts

	return {"success": True, "message": "Cookie 已保存并生效"}


@router.post("/api/cookies-save-reinit")
async def save_cookies_and_reinit(cookies: CookieUpdate, token: str = Depends(verify_admin_token)):
	"""保存 Cookie 并重新连接 Gemini（一步完成）"""
	if not cookies.secure_1psid or not cookies.secure_1psidts:
		raise HTTPException(status_code=400, detail="Cookie 值不能为空")

	# 保存到 .env 文件
	write_env(
		{
			"SECURE_1PSID": cookies.secure_1psid,
			"SECURE_1PSIDTS": cookies.secure_1psidts,
		}
	)

	# 更新运行时变量
	import main

	main.SECURE_1PSID = cookies.secure_1psid
	main.SECURE_1PSIDTS = cookies.secure_1psidts

	# 重新初始化客户端
	async with main.gemini_client_lock:
		if main.gemini_client is not None:
			try:
				await main.gemini_client.close()
			except Exception:
				pass
			main.gemini_client = None

	try:
		client = await main.get_gemini_client()
		if client:
			return {"success": True, "message": "Cookie 已保存，Gemini 重新连接成功"}
		else:
			return {"success": False, "message": "Cookie 已保存，但 Gemini 连接失败"}
	except Exception as e:
		return {"success": False, "message": f"Cookie 已保存，但 Gemini 重连失败: {str(e)}"}


@router.post("/api/reinit")
async def reinit_client(token: str = Depends(verify_admin_token)):
	"""重新初始化 Gemini 客户端"""
	import main

	# 在锁内关闭旧客户端，避免竞态
	async with main.gemini_client_lock:
		if main.gemini_client is not None:
			try:
				await main.gemini_client.close()
			except Exception:
				pass
			main.gemini_client = None

	# 尝试重新初始化（get_gemini_client 内部也会获取锁）
	try:
		client = await main.get_gemini_client()
		if client:
			return {"success": True, "message": "Gemini 客户端重新连接成功"}
		else:
			raise HTTPException(status_code=500, detail="客户端初始化返回空值")
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"重新连接失败: {str(e)}")


# ===== Gemini Gems 管理 =====


class GemCreate(BaseModel):
	"""创建 Gem 请求"""

	name: str
	prompt: str
	description: str = ""


class GemUpdate(BaseModel):
	"""更新 Gem 请求（全量更新）"""

	name: str
	prompt: str
	description: str = ""


def _gem_to_dict(gem) -> dict:
	"""把 gemini_webapi.Gem 转成可序列化的 dict"""
	return {
		"id": getattr(gem, "id", ""),
		"name": getattr(gem, "name", ""),
		"description": getattr(gem, "description", None),
		"prompt": getattr(gem, "prompt", None),
		"predefined": getattr(gem, "predefined", False),
	}


@router.get("/api/gems")
async def list_gems(token: str = Depends(verify_admin_token)):
	"""列出所有 Gemini Gem（预置 + 自建）"""
	import main

	try:
		client = await main.get_gemini_client()
		if client is None:
			return {"success": False, "message": "Gemini 客户端未初始化", "gems": []}
		# include_hidden=True 拿到完整列表（含隐藏预置 gem）
		gem_jar = await client.fetch_gems(include_hidden=True)
		gems = [_gem_to_dict(g) for g in gem_jar]
		return {"success": True, "gems": gems, "current_gem_id": main.GEM_ID}
	except Exception as e:
		return {"success": False, "message": f"获取 Gem 列表失败: {str(e)}", "gems": [], "current_gem_id": getattr(main, "GEM_ID", "")}


@router.post("/api/gems")
async def create_gem(gem: GemCreate, token: str = Depends(verify_admin_token)):
	"""创建新的自定义 Gem"""
	import main

	if not gem.name or not gem.prompt:
		return {"success": False, "message": "name 和 prompt 不能为空"}

	try:
		client = await main.get_gemini_client()
		if client is None:
			return {"success": False, "message": "Gemini 客户端未初始化"}
		new_gem = await client.create_gem(name=gem.name, prompt=gem.prompt, description=gem.description or "")
		# 刷新缓存
		await client.fetch_gems(include_hidden=True)
		return {"success": True, "message": f"Gem '{gem.name}' 创建成功", "gem": _gem_to_dict(new_gem)}
	except Exception as e:
		return {"success": False, "message": f"创建 Gem 失败: {str(e)}"}


@router.put("/api/gems/{gem_id}")
async def update_gem(gem_id: str, gem: GemUpdate, token: str = Depends(verify_admin_token)):
	"""更新现有的自定义 Gem（全量更新）"""
	import main

	if not gem.name or not gem.prompt:
		return {"success": False, "message": "name 和 prompt 不能为空"}

	try:
		client = await main.get_gemini_client()
		if client is None:
			return {"success": False, "message": "Gemini 客户端未初始化"}
		updated = await client.update_gem(gem=gem_id, name=gem.name, prompt=gem.prompt, description=gem.description or "")
		# 刷新缓存（update_gem 不校验响应，靠重新 fetch 确认）
		await client.fetch_gems(include_hidden=True)
		return {"success": True, "message": f"Gem '{gem.name}' 更新成功", "gem": _gem_to_dict(updated)}
	except Exception as e:
		return {"success": False, "message": f"更新 Gem 失败: {str(e)}"}


@router.delete("/api/gems/{gem_id}")
async def delete_gem(gem_id: str, token: str = Depends(verify_admin_token)):
	"""删除自定义 Gem"""
	import main

	try:
		client = await main.get_gemini_client()
		if client is None:
			return {"success": False, "message": "Gemini 客户端未初始化"}
		await client.delete_gem(gem=gem_id)
		# 刷新缓存（delete_gem 不校验响应，靠重新 fetch 确认）
		await client.fetch_gems(include_hidden=True)
		# 如果删除的是当前激活的 Gem，清空 GEM_ID
		if getattr(main, "GEM_ID", "") == gem_id:
			write_env({"GEM_ID": ""})
			main.GEM_ID = ""
		return {"success": True, "message": "Gem 已删除"}
	except Exception as e:
		return {"success": False, "message": f"删除 Gem 失败: {str(e)}"}


@router.post("/api/gems/{gem_id}/activate")
async def activate_gem(gem_id: str, token: str = Depends(verify_admin_token)):
	"""激活指定 Gem 作为全局 system prompt（写 .env + 改运行时）"""
	import main

	# 空字符串表示取消激活
	target = gem_id if gem_id and gem_id.lower() not in ("none", "null", "0") else ""
	try:
		write_env({"GEM_ID": target})
		main.GEM_ID = target
		if target:
			return {"success": True, "message": f"已激活 Gem: {target}", "gem_id": target}
		else:
			return {"success": True, "message": "已取消 Gem 激活（使用默认行为）", "gem_id": ""}
	except Exception as e:
		return {"success": False, "message": f"激活 Gem 失败: {str(e)}"}


@router.post("/api/gems/deactivate")
async def deactivate_gem(token: str = Depends(verify_admin_token)):
	"""取消激活当前 Gem（清空 GEM_ID）"""
	import main

	try:
		write_env({"GEM_ID": ""})
		main.GEM_ID = ""
		return {"success": True, "message": "已取消 Gem 激活（使用默认行为）", "gem_id": ""}
	except Exception as e:
		return {"success": False, "message": f"取消激活失败: {str(e)}"}


def setup_middleware(app):
	"""设置请求日志中间件"""
	from starlette.middleware.base import BaseHTTPMiddleware
	from starlette.requests import Request

	class RequestLoggingMiddleware(BaseHTTPMiddleware):
		async def dispatch(self, request: Request, call_next):
			start = time.time()

			# 跳过静态资源和管理面板的请求
			path = request.url.path
			if path.startswith("/admin") or path.startswith("/static"):
				return await call_next(request)

			response = await call_next(request)

			# 记录 API 请求
			if path.startswith("/v1/"):
				duration = time.time() - start
				log_request(request.method, path, response.status_code, duration)

			return response

	app.add_middleware(RequestLoggingMiddleware)
