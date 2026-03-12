# Gemi2Api-Server
[HanaokaYuzu / Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 的服务端简单实现

[![pE79pPf.png](https://s21.ax1x.com/2025/04/28/pE79pPf.png)](https://imgse.com/i/pE79pPf)

## 快捷部署

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zhiyu1998/Gemi2Api-Server)

### HuggingFace（由佬友@qqrr部署）

[![Deploy to HuggingFace](https://img.shields.io/badge/%E7%82%B9%E5%87%BB%E9%83%A8%E7%BD%B2-%F0%9F%A4%97-fff)](https://huggingface.co/spaces/ykl45/gmn2a)

## 直接运行

0. 配置 `.env` 文件（登录 Gemini 在浏览器开发工具中查找 Cookie），有必要的话可以填写 `API_KEY`

**方式一：多账号池（推荐）** — 支持粘性会话 + 并发限流
```properties
# 多账号 JSON 数组（每个对象包含一组 Cookie）
GEMINI_COOKIES_POOL = '[{"__Secure-1PSID":"账号1_PSID","__Secure-1PSIDTS":"账号1_PSIDTS"},{"__Secure-1PSID":"账号2_PSID","__Secure-1PSIDTS":"账号2_PSIDTS"}]'
MAX_CONCURRENT_PER_ACCOUNT = 3 # 每个账号最大并发请求数，默认3
API_KEY= "API_KEY VALUE HERE"
```
> [!TIP]
> 粘性会话机制：相同的 `Authorization` header 会通过哈希始终路由到同一个账号，保证对话上下文连贯。不同用户/token 自动分散到不同账号。

**方式二：单账号（传统模式）** — 如果只有一个账号
```properties
SECURE_1PSID = "COOKIE VALUE HERE"
SECURE_1PSIDTS = "COOKIE VALUE HERE"
API_KEY= "API_KEY VALUE HERE"
```

**通用配置项**
```properties
TEMPORARY_CHAT = "false" # 使用临时对话模式，此模式会禁用部分功能如思考、图片生成等，默认关闭。
AUTO_DELETE_CHAT = "true" # 生成结束后自动从web端删除对话记录，默认开启。TEMPORARY_CHAT为true时，此项无效。
PUBLIC_BASE_URL = "https://your-domain.com" # 外部URL，用于生成图片代理链接，不填则会使用内部地址。使用反向代理时必填，否则可能导致图片无法访问。
```
1. `uv` 安装一下依赖
> uv init
> 
> uv add fastapi uvicorn gemini-webapi httpx h2

> [!NOTE]  
> 如果存在`pyproject.toml` 那么就使用下面的命令：  
> uv sync

或者 `pip` 也可以

> pip install fastapi uvicorn gemini-webapi httpx h2

2. 激活一下环境
> source venv/bin/activate

3. 启动
> uvicorn main:app --reload --host 127.0.0.1 --port 8000

> [!WARNING] 
> tips: 如果不填写 API_KEY ，那么就直接使用

## 使用Docker运行（推荐）

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

4. 服务将在 http://0.0.0.0:8000 上运行

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

## API端点

- `GET /`: 服务状态检查
- `GET /v1/models`: 获取可用模型列表
- `POST /v1/chat/completions`: 与模型聊天 (类似OpenAI接口)
- `GET /gemini-proxy/image`: 图片代理接口（有生成图片需求时，需要保证此端点可直接访问，如果使用反向代理则需要填写`PUBLIC_BASE_URL`环境变量）

## 多账号粘性会话 (Sticky Session)

本项目支持通过 `GEMINI_COOKIES_POOL` 配置多个 Gemini 账号，实现**负载分散 + 会话上下文连贯**。

### 工作原理

```
用户 A (Bearer token-xxx) → MD5 哈希 → 始终命中 Client #2
用户 B (Bearer token-yyy) → MD5 哈希 → 始终命中 Client #0
用户 C (Bearer token-zzz) → MD5 哈希 → 始终命中 Client #1
```

- ✅ **同一用户的多轮对话** 始终路由到同一个账号，上下文不丢失
- ✅ **不同用户/token** 自动分散到不同账号，负载均衡
- ✅ **每个账号并发限流**（默认 3 个并发，可通过 `MAX_CONCURRENT_PER_ACCOUNT` 调整）

### 启动日志示例

```
INFO:main:🏊 Loaded 3 account(s) from GEMINI_COOKIES_POOL
INFO:main:✅ Client #0 initialized (PSID=g.a00074...)
INFO:main:✅ Client #1 initialized (PSID=g.a00074...)
INFO:main:✅ Client #2 initialized (PSID=g.a00074...)
INFO:main:🚀 Client pool ready: 3 active client(s), max 3 concurrent per account
```

### 请求分发日志

```
INFO:main:🔀 Sticky dispatch: token_hash=a1b2c3d4... → client #2 (pool_size=3)
```

### 容量估算

| 账号数 | 每账号并发 | 全局最大同时请求 |
|--------|-----------|-----------------|
| 3      | 3         | 9               |
| 5      | 3         | 15              |
| 5      | 5         | 25              |

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

- 图片去水印算法基于 [journey-ad/gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover)以及[allenk/GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool)实现，并直接使用了其中的两张png图片。

## 贡献

同时感谢以下开发者对 `Gemi2Api-Server` 作出的贡献：

<a href="https://github.com/zhiyu1998/Gemi2Api-Server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zhiyu1998/Gemi2Api-Server&max=1000" />
</a>