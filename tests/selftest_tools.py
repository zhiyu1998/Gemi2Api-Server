"""本地自检：tools / tool_calls 解析与流式闸门（不访问 Google）。

跑法：.venv/Scripts/python.exe selftest_tools.py
"""

import sys
from pathlib import Path

# 直接跑脚本时把项目根加进 sys.path，这样能 import 到 gemini_models / gemini_tools
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import gemini_tools as gt


def check(name, condition, detail=""):
	print(("  OK  " if condition else "  FAIL") + f" {name}" + (f"  {detail}" if detail else ""))
	if not condition:
		raise SystemExit(1)


TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "shell_command",
			"description": "run a shell command",
			"parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
		},
	},
	{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
]

print("== tool_choice ==")
check("none", gt.parse_tool_choice("none") == ("none", ""))
check("required", gt.parse_tool_choice("required") == ("required", ""))
check("缺省 auto", gt.parse_tool_choice(None) == ("auto", ""))
check("点名函数", gt.parse_tool_choice({"type": "function", "function": {"name": "read_file"}}) == ("required", "read_file"))

print("== 指令拼装 ==")
top, bottom = gt.build_tools_instruction(TOOLS, "auto")
check("顶部含格式说明", "tool_call" in top and "shell_command" in top)
check("末尾有再锚一次", "highest priority" in bottom and "run_command" in bottom)
check("none 不注入", gt.build_tools_instruction(TOOLS, "none") == ("", ""))
check("没有 tools 不注入", gt.build_tools_instruction([], "auto") == ("", ""))
top_forced, _ = gt.build_tools_instruction(TOOLS, {"type": "function", "function": {"name": "read_file"}})
check("点名后只留该工具", "read_file" in top_forced and "shell_command" not in top_forced)

print("== 围栏闭合（参数里带 markdown 代码块）==")
payload = '{"name": "shell_command", "arguments": {"command": "cat <<EOF\\n```bash\\necho hi\\n```\\nEOF"}}'
body, nxt, ok = gt.tool_fence_end(payload + "\n```", True)
check("闭合位置在 JSON 之后", ok and payload[:body].strip().startswith("{") and payload[:body].strip().endswith("}"), f"body={body}")
check("不会截在参数里的 ```", json.loads(payload)["arguments"]["command"].endswith("EOF"))

print("== 围栏未写完时等待 ==")
_, _, ok = gt.tool_fence_end('{"name": "shell_command", "arguments": {"comm', False)
check("流式中不判定", not ok)
_, _, ok = gt.tool_fence_end('{"name": "shell_command", "arguments": {"comm', True)
check("final 退回第一个 ```", not ok)  # 没有闭合围栏，仍判定不了

print("== parse_tool_calls ==")
text = '先看一下\n```tool_call\n{"name": "shell_command", "arguments": {"command": "ls -la"}}\n```\n看完了'
clean, calls = gt.parse_tool_calls(text)
check("正文剥掉围栏", clean == "先看一下\n\n看完了", repr(clean))
check("解析出一个调用", len(calls) == 1)
check("名字正确", calls[0]["function"]["name"] == "shell_command")
check("参数是 JSON 对象字符串", json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"})
check("id 前缀", calls[0]["id"].startswith("call_"))

one_line = '```tool_call{"name":"read_file","arguments":{"path":"a.txt"}}```'
clean, calls = gt.parse_tool_calls(one_line)
check("单行围栏也能解析", len(calls) == 1 and clean == "", repr(clean))

nested = '```tool_call\n{"function": {"name": "read_file", "arguments": {"path": "b"}}}\n```'
_, calls = gt.parse_tool_calls(nested)
check("嵌套 function 结构", calls and calls[0]["function"]["name"] == "read_file")

flat = '```tool_call\n{"tool_name": "read_file", "path": "c.txt"}\n```'
_, calls = gt.parse_tool_calls(flat)
check("平铺参数兜底", calls and json.loads(calls[0]["function"]["arguments"]) == {"path": "c.txt"})

bad = '```tool_call\n{not json at all}\n```'
clean, calls = gt.parse_tool_calls(bad)
check("坏 JSON 不产出调用", calls == [])
check("坏围栏仍从正文去掉", clean == "", repr(clean))

unclosed = '开头```tool_call\n{"name": "x"}'
clean, calls = gt.parse_tool_calls(unclosed)
check("未闭合围栏原样留下", clean == unclosed and calls == [])

print("== 流式闸门 ==")
sent = []
gate = gt.ToolFenceGate(sent.append)
for piece in ["你好", "，这是```to", "ol_call\n{\"name\": \"read_file\", \"arg", "uments\": {}}\n```", "结尾"]:
	gate.push(piece)
check("围栏内容没漏出去", "tool_call" not in "".join(sent) and "read_file" not in "".join(sent), repr("".join(sent)))
check("已发内容 = 前后正文", "".join(sent) == "你好，这是结尾", repr("".join(sent)))

full = "你好，这是```tool_call\n{\"name\": \"read_file\", \"arguments\": {}}\n```结尾"
clean, calls = gt.parse_tool_calls(full)
check("全部推送后无尾巴可补", gate.remaining(clean) == "", repr(gate.remaining(clean)))
check("收尾解析出调用", len(calls) == 1)

# 只推到围栏闭合为止，尾巴留给收尾补发（真实流式里最后一帧常被闸门扣着）
sent2 = []
gate2 = gt.ToolFenceGate(sent2.append)
for piece in ["你好，这是```to", "ol_call\n{\"name\": \"read_file\", \"arguments\": {}}\n```"]:
	gate2.push(piece)
check("扣住围栏后只发了正文", "".join(sent2) == "你好，这是", repr("".join(sent2)))
check("收尾能补发尾巴", gate2.remaining(clean) == "结尾", repr(gate2.remaining(clean)))

print("== 流式 tool_calls 带 index ==")
stream = gt.tool_calls_for_stream(calls + calls)
check("index 连续", [c["index"] for c in stream] == [0, 1])
check("保留 id/type", stream[0]["id"].startswith("call_") and stream[0]["type"] == "function")

print("== 工具结果渲染 ==")
codex_ok = "Chunk ID: 0cdbf0\nWall time: 0.07s\nProcess exited with code 0\nOriginal token count: 0\nOutput:\n"
check("exit 0 无输出", "✅" in gt.format_tool_result("shell_command", codex_ok) and "不要再跑" in gt.format_tool_result("shell_command", codex_ok))
codex_out = "Process exited with code 0\nOutput:\nhello\n"
check("exit 0 带输出", "hello" in gt.format_tool_result("shell_command", codex_out))
codex_fail = "Process exited with code 2\nOutput:\nboom\n"
check("非 0 标失败", "❌" in gt.format_tool_result("shell_command", codex_fail) and "boom" in gt.format_tool_result("shell_command", codex_fail))
check("空结果说明完成", "完成" in gt.format_tool_result("read_file", ""))
check("无退出码原样透出", "随便什么" in gt.format_tool_result("read_file", "随便什么"))

print("\n全部通过")
