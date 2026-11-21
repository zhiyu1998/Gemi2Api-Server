"""本地自检：模型解析 + 线上改写（不访问 Google）。

跑法：.venv/Scripts/python.exe selftest_wire.py
"""

import sys
from pathlib import Path

# 直接跑脚本时把项目根加进 sys.path，这样能 import 到 gemini_models / gemini_tools
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import gemini_models as gm
import gemini_wire as gw


def _fake_freq():
	"""仿 gemini-webapi 2.0.0 的 f.req（inner 69 槽）。"""
	inner = [None] * 69
	inner[0] = ["hello", 0, None, None, None, None, 0]
	inner[1] = ["en"]
	inner[2] = ["", "", "", None, None, None, None, None, None, ""]
	inner[6] = [1]
	inner[7] = 1
	inner[10] = 1
	inner[11] = 0
	inner[17] = [[0]]
	inner[18] = 0
	inner[27] = 1
	inner[30] = [4]
	inner[41] = [1]
	inner[53] = 0
	inner[59] = "REQ-UUID"
	inner[61] = []
	inner[68] = 2
	return json.dumps([None, json.dumps(inner)])


def check(name, condition, detail=""):
	print(("  OK  " if condition else "  FAIL") + f" {name}" + (f"  {detail}" if detail else ""))
	if not condition:
		raise SystemExit(1)


print("== 模型解析 ==")
cases = {
	"gemini-4-hydrogen": (gm.HEX_HYDROGEN_4, 1, 1),
	"gemini-4-carbon": (gm.HEX_ARGON_4, 8, 1),
	"gemini-4-argon": (gm.HEX_ARGON_4, 8, 1),
	"gemini-4-argon-high": (gm.HEX_ARGON_4, 8, 2),
	"gemini-4-carbon-thinking": (gm.HEX_ARGON_4, 8, 2),
	"gemini-4-carbon-thinking-medium": (gm.HEX_ARGON_4, 8, 4),
	"gemini-4-hydrogen-thinking": (gm.HEX_HYDROGEN_4, 1, 2),
	"models/gemini-4-carbon": (gm.HEX_ARGON_4, 8, 1),
	"gemini-4-carbon@think=9": (gm.HEX_ARGON_4, 8, 1),
	"gemini-3.5-flash-lite": (gm.HEX_FLASH_LITE_35, 6, 1),
	"gemini-3.8-flash": (gm.HEX_FLASH_38, 1, 1),
	"gemini-4-neon": (gm.HEX_NEON_4, 3, 1),
	"neon": (gm.HEX_NEON_4, 3, 1),
	"gemini-4-neon-thinking": (gm.HEX_NEON_4, 3, 2),
	"gemini-4-neon-high": (gm.HEX_NEON_4, 3, 2),
	"gemini-3.1-pro": (gm.HEX_PRO_31, 3, 1),
	"gemini-4-hydrogen-task": (gm.HEX_HYDROGEN_4, 1, 1),
	"gemini-4-carbon-task": (gm.HEX_ARGON_4, 8, 1),
	"gemini-4-argon-task": (gm.HEX_ARGON_4, 8, 1),
}
for raw, expected in cases.items():
	spec = gm.resolve_builtin(gm.normalize_model_name(raw))
	got = (spec.hex_id, spec.mode, spec.think_wire) if spec else None
	check(raw, got == expected, f"-> {got}")

check("3.1-pro 不再标记下线", gm.resolve_builtin("gemini-3.1-pro").retired == "")
check("关键词兜底 neon->neon", gm.fuzzy_builtin("gpt-neon").name == "gemini-4-neon")
check("关键词兜底 pro->carbon", gm.fuzzy_builtin("gpt-pro").name == "gemini-4-carbon")
check("关键词兜底 flash->hydrogen", gm.fuzzy_builtin("gpt-4-flash").name == "gemini-4-hydrogen")
check("未知名字不兜底", gm.fuzzy_builtin("totally-unknown") is None)

print("== reasoning_effort → 思考档 ==")
for effort, expected in (("low", 1), ("medium", 4), ("high", 2), ("minimal", 1), ("HIGH", 2), ("medium", 4)):
	check(f"effort={effort}", gm.think_from_effort(effort) == expected, f"-> {gm.think_from_effort(effort)}")
check("没给 effort 返回 0", gm.think_from_effort(None, None, None) == 0)
check("别名 thinking_effort 生效", gm.think_from_effort(None, "high") == 2)
check("非法值忽略", gm.think_from_effort("bogus") == 0)
check("线值直传 4", gm.think_from_effort("4") == 4)

print("== 旧自定义 header 反解 ==")
legacy = (
	'[1,null,null,null,"95a4c0abcba24364",null,null,0,[4,5,6,8,16,19,20,4,5,6,8,16,19,20],'
	'null,null,2,null,null,8,2,"FF5B6EAC-EBC6-4770-9A3A-812978C5C314",'
	'["boq_gemini-web-uiserver_20261021.01_p0"],null,[[2,452699999],[1792651926,126000000]]]'
)
spec = gm.parse_header_spec(legacy, "gemini-4-argon-high")
check("legacy hex/mode/think/capacity", (spec.hex_id, spec.mode, spec.think_wire, spec.capacity) == (gm.HEX_ARGON_4, 8, 2, 2), str(spec))

print("== 请求头 ==")
header = gm.build_header_string(gm.HEX_ARGON_4, 2, 8, 2, "SESSION-UUID")
parsed = json.loads(header)
check("header 长度 17", len(parsed) == 17, str(len(parsed)))
check("index4=hex", parsed[4] == gm.HEX_ARGON_4)
check("index11=capacity", parsed[11] == 2)
check("index14=mode", parsed[14] == 8)
check("index15=think", parsed[15] == 2)
check("index16=uuid", parsed[16] == "SESSION-UUID")
check("index8=[4,5,6,8]", parsed[8] == [4, 5, 6, 8])

print("== f.req 改写：4 Carbon High ==")
spec = gm.resolve_builtin("gemini-4-carbon-thinking")
out = gw._rewrite_freq(_fake_freq(), spec)
outer = json.loads(out)
inner = json.loads(outer[1])
check("补到 99 槽", len(inner) == 99, str(len(inner)))
check("inner[79]=mode 8", inner[79] == 8)
check("inner[80]=think 2", inner[80] == 2)
check("inner[49]=40", inner[49] == 40)
check("inner[68]=2", inner[68] == 2)
check("inner[83]=1", inner[83] == 1)
check("inner[91]=1", inner[91] == 1)
check("inner[96]=1", inner[96] == 1)
check("inner[98]=1", inner[98] == 1)
check("提示词保留", inner[0][0] == "hello")
check("inner[59] 保留", inner[59] == "REQ-UUID")

print("== f.req 改写：4 Neon（档位 3）==")
spec = gm.resolve_builtin("gemini-4-neon")
neon_header = json.loads(gm.build_header_string(spec.hex_id, spec.capacity, spec.mode, spec.think_wire, "SESSION-UUID"))
check("Neon header index4=hex", neon_header[4] == gm.HEX_NEON_4)
check("Neon header index14=mode 3", neon_header[14] == 3)
out = gw._rewrite_freq(_fake_freq(), spec)
inner = json.loads(json.loads(out)[1])
check("补到 99 槽", len(inner) == 99, str(len(inner)))
check("inner[79]=mode 3", inner[79] == 3)
check("inner[80]=think 1", inner[80] == 1)
check("inner[96]=0", inner[96] == 0)
check("不补 inner[49]（档位 8 专属）", inner[49] is None)
check("inner[68] 保留库值", inner[68] == 2)
check("提示词保留", inner[0][0] == "hello")

print("== f.req 改写：Task（featureMode 40）==")
spec = gm.resolve_builtin("gemini-4-carbon-task")
out = gw._rewrite_freq(_fake_freq(), spec)
inner = json.loads(json.loads(out)[1])
check("inner[49]=featureMode 40", inner[49] == 40)
check("inner[83]=1", inner[83] == 1)
check("inner[6]=[0]", inner[6] == [0])
check("inner[68]=1", inner[68] == 1)
check("inner[30]=Task 工具列表", inner[30] == [4, 27, 28, 16, 19, 12])
check("inner[79]=mode 8", inner[79] == 8)
check("inner[98]=1", inner[98] == 1)

print("== f.req 改写：普通模型不动长度 ==")
spec = gm.resolve_builtin("gemini-3.5-flash-lite")
out = gw._rewrite_freq(_fake_freq(), spec)
inner = json.loads(json.loads(out)[1])
check("仍是 69 槽", len(inner) == 69, str(len(inner)))
check("inner[68] 保留库值", inner[68] == 2)
check("提示词保留", inner[0][0] == "hello")

print("== 深度研究不被覆盖 inner[49] ==")
inner = json.loads(json.loads(_fake_freq())[1])
inner[3] = "!" + "x" * 40
inner[49] = 1
out = gw._rewrite_freq(json.dumps([None, json.dumps(inner)]), gm.resolve_builtin("gemini-4-carbon"))
inner2 = json.loads(json.loads(out)[1])
check("deep research inner[49] 保持 1", inner2[49] == 1, str(inner2[49]))

print("\n全部通过")
