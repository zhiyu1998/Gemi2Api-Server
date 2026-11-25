# Gemi2Api-Server
[HanaokaYuzu / Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 的服务端简单实现

| 原始界面 | 管理面板 |
|:---:|:---:|
| <img width="2160" height="1266" alt="beautified-screenshot (1)" src="https://github.com/user-attachments/assets/f67fce48-2555-4c97-b3da-c04aa3236eac" /> | <img width="2160" height="1266" alt="beautified-screenshot (2)" src="https://github.com/user-attachments/assets/fa75e6df-eb27-4cbf-b538-2a7ee9dc8d9c" /> |

## 在线体验

| 平台 | 地址 | 面板密码 |
|------|------|---------|
| Render | [gemi2api-server-1ol5.onrender.com/admin/](https://gemi2api-server-1ol5.onrender.com/admin/) | `Gemi2Api-Server`（demo 专用，请勿用于生产） |
| HuggingFace | [zhiyu1998-gemi2api-server.hf.space/admin/](https://zhiyu1998-gemi2api-server.hf.space/admin/) | `Gemi2Api-Server`（demo 专用，请勿用于生产） |

> [!NOTE]
> 以上体验地址未配置 Gemini Cookie，仅展示管理面板界面。如需完整功能请自行部署。

## 快捷部署

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zhiyu1998/Gemi2Api-Server)

### HuggingFace

[![Follow us on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-lg-dark.svg)](https://huggingface.co/new-space?duplicate=true&repo=zhiyu1998/Gemi2Api-Server)

## 直接运行

0. 填入 `SECURE_1PSID` 和 `SECURE_1PSIDTS`（登录 Gemini 在浏览器开发工具中查找 Cookie），有必要的话可以填写 `API_KEY`

```properties
SECURE_1PSID = "COOKIE VALUE HERE"
SECURE_1PSIDTS = "COOKIE VALUE HERE"
API_KEY= "API_KEY VALUE HERE"
TEMPORARY_CHAT = "false" # 使用临时对话模式，此模式会禁用部分功能如思考、图片生成等，默认关闭。
AUTO_DELETE_CHAT = "true" # 生成结束后自动从web端删除对话记录，默认开启。TEMPORARY_CHAT为true时，此项无效。
PUBLIC_BASE_URL = "https://your-domain.com" # 外部URL，用于生成图片代理链接，不填则会使用内部地址。使用反向代理时必填，否则可能导致图片无法访问。
CUSTOM_MODELS_FILE = "" # 可选，默认留空（只用内置模型）。如需覆写/新增 Gemini 模型定义（YAML/JSON），指向自己的文件；custom_models.yaml 已被 .gitignore 忽略，仓库不附带，文件不存在时服务仅告警并跳过。
DISABLE_BUILTIN_MODELS = "false" # 可选，默认 false。设为 true 时，/v1/models 仅返回 CUSTOM_MODELS(_FILE) 中定义的模型，不再追加 gemini_webapi 内置模型（未配置自定义模型则返回空列表）。此开关在服务启动时读取，修改 .env 后需重启生效（区别于 CUSTOM_MODELS_FILE 的免重启热加载）；也可在管理面板「功能开关」中切换，即时生效并写回 .env。
```

`custom_models.yaml` 示例（本地自建该文件并用 `CUSTOM_MODELS_FILE` 指向它即可；该文件已被 .gitignore 忽略，仓库不附带）：

```yaml
models:
  - model_name: "gemini-3.5-flash-lite" # 快速解答
    model_header:
      x-goog-ext-525001261-jspb: '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8],null,null,1,null,null,6]'

  - model_name: "gemini-4-hydrogen" # 日常帮助
    model_header:
      x-goog-ext-525001261-jspb: '[1,null,null,null,"56fdd199312815e2",null,null,0,[4,5,6,8],null,null,2,null,null,1]'

  - model_name: "gemini-3.1-pro" # 深度研究（旧名，上游已换成 4 Neon，同一条路由）
    model_header:
      x-goog-ext-525001261-jspb: '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4,5,6,8],null,null,1,null,null,3]'

  - model_name: "gemini-4-neon" # 深度分析（接替 3.1 Pro，付费/Ultra 号）
    model_header:
      x-goog-ext-525001261-jspb: '[1,null,null,null,"e6fa609c3fa255c0",null,null,0,[4,5,6,8],null,null,2,null,null,3]'

  - model_name: "gemini-4-argon" # 代码与分析（上游名 4 Carbon）
    model_header:
      x-goog-ext-525001261-jspb: '[1,null,null,null,"95a4c0abcba24364",null,null,0,[4,5,6,8],null,null,2,null,null,8]'
```

> `model_header` 只需要三个值：下标 4 是模型 hex，下标 11 是账号档位，下标 14 是路由档位。**思考档不要写在这里**，它由请求参数 `reasoning_effort` 决定（见下文「模型与思考档」）。
> 服务端会把这条 header 解析成模型规格再重建真正的线上请求头，所以不用手抄 UUID、`["boq_..."]`、时间戳这类抓包残留，写了也会被忽略。
> `x-goog-ext-73010989-jspb` / `x-goog-ext-73010990-jspb` 由服务端补齐；`x-goog-ext-525005358-jspb` 每次请求自动生成，都不用写进 YAML。
> `CUSTOM_MODELS_FILE` 会在每次请求 / 读取模型列表时重新加载，改完文件不需要重启服务。

> [!TIP]
> 管理面板的「一键粘贴 Cookie」支持直接粘贴浏览器里复制的**整段 Cookie**：除 `__Secure-1PSID` / `__Secure-1PSIDTS` 外，`OSID`、`__Secure-OSID`、`SID`、`SAPISID`、`HSID` 等会存进 `GEMINI_EXTRA_COOKIES` 并随请求发给上游。Google 多账号会话缺这些 cookie 会被重定向到 accounts.google.com，表现为「Cookie 保存成功但请求一直挂」。
> 若初始化探针（建会话 + 读回历史）对某些账号类型误报，可在 `.env` 设 `COOKIE_VALIDATE_ON_INIT=false` 跳过该校验（`init()` 本身仍会验证登录态）。

1. 安装依赖

```bash
uv init
uv add fastapi uvicorn gemini-webapi httpx h2
```

> [!NOTE]
> 如果存在 `pyproject.toml` 那么就使用下面的命令：

```bash
uv sync
```

或者 `pip` 也可以：

```bash
pip install fastapi uvicorn gemini-webapi httpx h2
```

2. 激活环境

```bash
source venv/bin/activate
```

3. 启动

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 7860
```

> [!WARNING]
> tips: `.env.example` 中默认设置了 `API_KEY=Gemi2Api-Server`，复制到 `.env` 后即可登录管理面板。**正式部署前请务必更换为强密码。**

## 多 Cookie 轮询（多账号）

一个进程可以同时挂多个 Gemini 账号：请求按「在途少的优先」轮流分配，某个号失败就冷却并换下一个，正常的号继续参与轮询。

三种配置写法按 `1PSID` 去重后合并，`SECURE_1PSID` 永远排第一位，所以单账号老配置不用改：

```bash
# 1) 老配置：单账号
SECURE_1PSID=...
SECURE_1PSIDTS=...

# 2) 多账号：JSON 数组（也接受 {"cookies": [...]} / {"accounts": [...]}）
GEMINI_COOKIES=[{"label":"号2","psid":"...","psidts":"..."},{"label":"号3","psid":"...","psidts":"..."}]

# 3) 多账号：文件，每行 psid,psidts[,label]，支持 # 注释
COOKIES_FILE=secrets/cookies.txt
```

`secrets/cookies.txt`（默认路径就是它，存在才读）：

```text
# 备注：号4
g.a000xxxxx,1PSIDTS的值,号4
```

调参（都有默认值，不配也能跑）：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `COOKIE_MAX_CONCURRENCY` | `0` | 单个账号同时在途的请求上限，0 表示不限 |
| `COOKIE_MAX_RETRIES` | `0` | 单次请求最多试几个账号，0 表示等于账号数量 |
| `COOKIE_COOLDOWN_BASE` / `COOKIE_COOLDOWN_MAX` | `15` / `300` | 失败后的冷却退避区间（秒） |
| `COOKIE_VALIDATE_ON_INIT` | `true` | 新号第一次使用时跑一遍「建会话 + 读回」探针 |
| `COOKIE_EAGER_INIT` | `false` | 启动时预热全部账号（否则等流量打到才初始化） |

行为说明：

- **分配**：冷却中的号跳过；其余按（在途数, 上次使用时间）取最小，所以空闲号会被轮流用到，慢号也不会被压满。正在初始化的号按「忙」处理，并发请求不会排在它后面干等。
- **失败冷却**：生成失败按 15s → 30s → 60s → … → 300s 退避；初始化/校验失败说明这份 cookie 本身坏了，直接打到降级阈值并给足最长冷却。
- **降级**：连续失败 3 次的号变成「兜底号」，只有别的号都不可用时才轮到它。
- **重试**：非流式请求失败会自动换号重试（最多等于账号数）；流式请求会占住分配到的号直到流结束，中途不换号。
- **观测**：管理面板 `/admin` 的服务状态里会列出每个号的在途数、成功/失败次数和剩余冷却（接口字段 `cookie_accounts`），日志里也会打印 `Cookie 池：账号 xxx 失败第 N 次，冷却 Ns`。账号池卡片还会显示该号套餐（FREE / PRO / ULTRA）和 5 小时 / 每周额度，点「刷新额度」可实时重新拉取（接口 `POST /admin/api/quota-refresh`）。

> [!NOTE]
> 多个号必须是不同的 Google 账号；同一个 `1PSID` 只保留第一次出现的那份配置（例如同时写在 `SECURE_1PSID` 和 `GEMINI_COOKIES` 里时以环境变量为准）。

## 支持的协议

同一套模型与 Cookie 池对外提供四种协议（`protocols.py`）：

| 协议 | 端点 |
| --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses`（含 `stream` SSE 事件） |
| Anthropic Messages | `POST /v1/messages`、`POST /v1/messages/count_tokens`（`x-api-key` 或 Bearer） |
| Gemini 原生 | `POST /v1beta/models/{model}:generateContent`、`:streamGenerateContent`（`alt=sse` 走 SSE，否则 JSON 数组） |

鉴权与 `/v1/chat/completions` 一致（`Authorization: Bearer $API_KEY`；Anthropic 也认 `x-api-key`）。文本、system/instructions、temperature/top_p、stop_sequences、tools（提示词围栏模拟）都已支持；图片/音乐/视频等富媒体仍走 chat 里的 `gemini-image` / `gemini-music` / `gemini-video` / `gemini-canvas`。

```bash
# Anthropic
curl http://127.0.0.1:8001/v1/messages -H "x-api-key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-4-hydrogen","max_tokens":128,"messages":[{"role":"user","content":"你好"}]}'

# OpenAI Responses
curl http://127.0.0.1:8001/v1/responses -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-4-hydrogen","input":"你好"}'

# Gemini 原生
curl http://127.0.0.1:8001/v1beta/models/gemini-4-hydrogen:generateContent -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' -d '{"contents":[{"role":"user","parts":[{"text":"你好"}]}]}'
```

## 模型与思考档

内置模型目录（`gemini_models.py`）已覆盖下表这些路由，`/v1/models` 默认返回：

| 用途 | 模型名 | 上游 | hex | 档位 |
| --- | --- | --- | --- | --- |
| 快速解答 | `gemini-3.5-flash-lite` | 3.5 Flash-Lite | `cf41b0e0dd7d53e5` | 6 |
| 日常帮助 | `gemini-4-hydrogen` | 4 Hydrogen | `56fdd199312815e2` | 1 |
| 深度研究 | `gemini-4-neon`（旧名 `gemini-3.1-pro`） | 4 Neon（Deeper analysis，接替 3.1 Pro） | `e6fa609c3fa255c0` / `9d8ca3786ebdfbea` | 3 |
| 代码与分析 | `gemini-4-argon` | 4 Carbon（原 Argon） | `95a4c0abcba24364` | 8 |
| 任务执行 | `gemini-4-hydrogen-task`、`gemini-4-carbon-task`、`gemini-4-argon-task` | Task（代为执行任务） | 同对应 Chat 模型 | 1 / 8 |

思考档走 OpenAI 规范的 `reasoning_effort`，**不靠模型名后缀**，同一模型三档都可用：

| `reasoning_effort` | 网页 UI | 思考线值 | 说明 |
| --- | --- | --- | --- |
| `low`（或 `minimal`） | 低 Fast and efficient | 1 | 最快，适合短问答 |
| `medium` | 中 深度适中 | 4 | 服务端内部思考，不吐思考链 |
| `high` | 高 超级详尽 | 2 | 付费号会返回思考链 |

- 也接受 `thinking_effort` / `think_effort` 两个别名，以及直接写线值 `1` / `4` / `2`。
- 不传就用该模型默认档（1）。
- 别名照旧：`gemini-4-carbon` / `gemini-4-pro` 都指 4 Argon 这条路由，`gemini-3.1-pro` / `neon` 都指 4 Neon 这条路由；不认识的模型名直接返回 400，不会偷偷换成别的模型。

```bash
curl https://your-host/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemini-4-argon","reasoning_effort":"high","messages":[{"role":"user","content":"你好"}]}'
```

> [!NOTE]
> 4 Argon 走的是档位 8，服务端会自动补 `inner[49]=40` 等槽位，否则付费号上请求会挂到超时；代价是这一档拿不到 High 思考链。想要思考链用 `gemini-4-hydrogen` + `reasoning_effort: "high"`。

> [!NOTE]
> 4 Neon 走的是档位 3（`inner[79]=3`），接替原来 3.1 Pro 的位置，模型列表里显示为「Deeper analysis」。2026-11-19 在 Ultra 号上实测：普通档回报 `4 Neon`，`reasoning_effort: "high"` 回报 `4 Neon High` 并带 `reasoning_content`；免费号发档位 3 会挂到超时，所以这条路由要付费 / Ultra 账号的 cookie。`gemini-3.1-pro` 不再是下线模型，它和 `gemini-4-neon` 是同一条路由的两个名字。Neon 不是 Argon 的继任者：Argon 改名成了 Carbon，Neon 是另一个模型。

> [!NOTE]
> Task 生成的文件（PDF/文档）正文里是 `[file-tag: 文件名]`。服务会把它替换成对应 Gemini 会话链接；用户在已登录同一 Google 账号的浏览器里打开会话，点文件预览的「导出 → 下载」即可。服务端不下载、不缓存文件，避免 Google 对 `contribution.usercontent.com` 的 403 限制。多账号可用 `GEMINI_WEB_AUTHUSER` 指定 `/u/N`。

> [!NOTE]
> Task 模型（`*-task`，官网「Task / 代为执行任务」）走 `featureMode 40`：请求写 `inner[49]=40`、`inner[83]=1`，并强制使用普通会话（不能临时会话）。首次使用前服务会自动写账号偏好 222、再发 `UVb0T` 条款（见 `gemini_task.py`）；长任务的回答可能先以「运行中」结束，服务会再从会话里读回正文。只有账号的 Task 菜单里出现对应模型时才真正生效。

## 使用 Docker 运行（推荐）

### 快速开始

1. 克隆本项目
   ```bash
   git clone https://github.com/zhiyu1998/Gemi2Api-Server.git
   ```

2. 创建 `.env` 文件并填入你的 Gemini Cookie 凭据:
   ```bash
   cp .env.example .env
   # 用编辑器打开 .env 文件，填入你的 Cookie 值
   ```

3. 启动服务:
   ```bash
   docker-compose up -d
   ```

4. 服务将在 http://127.0.0.1:7860 上运行

### 其他 Docker 命令

```bash
# 查看日志
docker-compose logs

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 重新构建并启动
docker-compose up -d --build
```

## API 端点

- `GET /`: 服务状态检查
- `GET /v1/models`: 获取可用模型列表
- `POST /v1/chat/completions`: 与模型聊天（OpenAI 兼容，支持 `stream` 与 `reasoning_effort`）
- `GET /gemini-proxy/image`: 图片代理接口（有生成图片需求时，需要保证此端点可直接访问，如果使用反向代理则需要填写 `PUBLIC_BASE_URL` 环境变量）
- `GET /admin`: 管理面板

## 管理面板

访问 `/admin` 进入管理面板，支持以下功能：

- 查看服务状态和 Gemini 连接状态
- 配置 Gemini Cookie（无需重启）
- 修改 API_KEY、限流等参数
- 查看实时日志
- 在线测试聊天

> [!NOTE]
> 未设置 `API_KEY` 时管理面板不可用。密码通过 `.env` 文件中的 `API_KEY` 配置，`.env.example` 提供了默认值 `Gemi2Api-Server`，**正式部署前请务必更换**。

## 常见问题

### 服务器报 500 问题解决方案

500 的问题一般是 IP 不太行 或者 请求太频繁（后者等待一段时间或者重新新建一个隐身标签登录一下重新给 Secure_1PSID 和 Secure_1PSIDTS 即可），见 issue：
- [__Secure-1PSIDTS · Issue #6 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/6)
- [Failed to initialize client. SECURE_1PSIDTS could get expired frequently · Issue #72 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/72)

解决步骤：
1. 使用隐身标签访问 [Google Gemini](https://gemini.google.com/) 并登录
2. 打开浏览器开发工具 (F12)
3. 切换到 "Application" 或 "应用程序" 标签
4. 在左侧找到 "Cookies" > "gemini.google.com"
5. 复制 `__Secure-1PSID` 和 `__Secure-1PSIDTS` 的值
6. 更新 `.env` 文件
7. 重新构建并启动: `docker-compose up -d --build`

## 致谢

- 图片去水印算法基于 [journey-ad/gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover) 以及 [allenk/GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) 实现，并直接使用了其中的两张 png 图片。

## 贡献

同时感谢以下开发者对 `Gemi2Api-Server` 作出的贡献：

<a href="https://github.com/zhiyu1998/Gemi2Api-Server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zhiyu1998/Gemi2Api-Server&max=1000" />
</a>
