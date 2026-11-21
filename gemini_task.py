"""Task 模式（官网「Task / 代为执行任务」，featureMode 40）的初始化辅助。

官网第一次发 Task 前会弹 "Let's get started"：
1. 先把账号偏好 222 写成 true（RPC `L5adhe`/UPDATE_USER_PREFERENCES）；
2. 生成请求发出后，再补一条 `UVb0T` 服务条款同意（条款版本 0、界面语言、首页 thykhd 令牌、类别 12）。

不这么做时，上游只会回引导对话框文本，任务不会真正开始。这里两个 RPC 都是幂等的，
每个客户端只跑一次（失败也记已尝试，避免每次请求都重试拖慢）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from gemini_webapi.constants import GRPC
from gemini_webapi.types import RPCData

logger = logging.getLogger(__name__)

# 已经做过 Task 初始化的客户端（按 id 记，客户端重建会重新初始化）
_TASK_READY: set = set()


async def ensure_task_ready(client: Any) -> None:
	"""尽力让这个客户端可以发 Task 请求；失败只告警，不阻断主流程。"""
	if client is None:
		return
	key = id(client)
	if key in _TASK_READY:
		return

	# 1) 偏好 222 = Task 首次使用确认
	try:
		preferences = [None] * 222
		preferences[221] = 1
		payload = json.dumps([preferences, None, [222]])
		await client._batch_execute(
			[RPCData(rpcid=GRPC.UPDATE_USER_PREFERENCES, payload=payload)],
			source_path="/app",
		)
	except Exception as e:
		logger.warning("Task 偏好 222 写入失败（继续尝试）：%s", e)

	# 2) UVb0T 服务条款同意：库的 GRPC 枚举里没有这个 id，用 model_construct 绕过枚举校验
	try:
		language = getattr(client, "language", None) or "en"
		terms_token = getattr(client, "access_token", "") or ""
		terms_payload = [0, language, terms_token, None, None, None, None, None, None, None, None, None, None, 12, None, None, 2]
		terms_rpc = RPCData.model_construct(rpcid="UVb0T", payload=json.dumps(terms_payload), identifier="generic")
		await client._batch_execute([terms_rpc], source_path="/app")
	except Exception as e:
		logger.warning("Task 条款同意 UVb0T 失败（继续尝试）：%s", e)

	_TASK_READY.add(key)


def reset_task_ready() -> None:
	"""测试/调试用：清掉已初始化标记。"""
	_TASK_READY.clear()
