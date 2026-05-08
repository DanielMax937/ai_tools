# NVIDIA Nemotron 3 VoiceChat — 接入指南

> 基于 WebSocket 的实时全双工语音对话 CLI 工具。说麦克风，听模型语音回复，支持打断（barge-in）。

## 目录

- [快速开始](#快速开始)
- [如何接入](#如何接入)
- [CLI 参数](#cli-参数)
- [通过代理运行](#通过代理运行)
- [架构概览](#架构概览)
- [碰到的问题与解决方案](#碰到的问题与解决方案)
- [使用场景](#使用场景)

---

## 快速开始

```bash
# 1. 安装依赖
npm install

# 2. 列出音频设备
node nvidia-voicechat-interactive.js --list-devices

# 3. 启动语音对话
node nvidia-voicechat-interactive.js

# 4. Ctrl+C 退出
```

API Key 从 `.env` 文件自动读取：

```
NVIDIA_API_KEY="nvapi-..."
```

---

## 如何接入

### 前置条件

| 依赖 | 用途 | 安装 |
|------|------|------|
| Node.js ≥ 18 | 运行时 | `brew install node` |
| ffmpeg + ffplay | 麦克风采集 & 扬声器播放 | `brew install ffmpeg` |
| NVIDIA_API_KEY | 调用 Nemotron 3 VoiceChat | 从 NVIDIA NGC 获取 |

Node 包依赖（`npm install` 自动安装）：

```
ws                    # WebSocket 客户端
https-proxy-agent     # HTTP/HTTPS 代理（可选）
socks-proxy-agent     # SOCKS5 代理（可选）
```

### 协议流程

WebSocket 连接到 NVIDIA NGC 云端，协议兼容 OpenAI Realtime API：

```
Client                           Server (NVIDIA NGC)
  │                                   │
  │──── WebSocket connect ───────────>│  wss://api.ngc.nvidia.com/...
  │                                   │
  │──── { functionId: "42c86b5f..." }─>│  注册 function
  │                                   │
  │──── session.update ──────────────>│  配置音频格式 + persona + VAD
  │     { type: "session.update",     │
  │       session: {                  │
  │         audio: {                  │
  │           input:  { format: "pcm16" },  │
  │           output: { format: "pcm16" }   │
  │         },                       │
  │         turn_detection: {         │
  │           type: "server_vad"      │  ← 服务端 VAD 自动检测说话结束
  │         }                         │
  │       }                           │
  │     }                             │
  │                                   │
  │──── input_audio_buffer.append ───>│  持续发送麦克风音频 (base64 PCM16)
  │──── input_audio_buffer.append ───>│
  │──── input_audio_buffer.append ───>│
  │                                   │
  │<─── input_audio_buffer.speech_started ─│  VAD 检测到用户开始说话
  │<─── input_audio_buffer.speech_stopped  ─│  VAD 检测到用户停止说话
  │                                   │  （服务端自动 commit，无需手动操作）
  │                                   │
  │<─── response.audio.delta ────────│  模型音频流 (base64 PCM16)
  │<─── response.audio_transcript.delta ─│  模型文字转录
  │<─── response.audio.done ─────────│  音频结束
  │                                   │
  │──── response.cancel ─────────────>│  打断（用户在模型说话时插话）
  │<─── response.cancelled ──────────│
```

### 关键设计

- **音频格式**：24kHz 单声道 PCM16，这是 Nemotron VoiceChat 的原生采样率
- **帧大小**：~240ms（11520 bytes），麦克风音频以此粒度发送
- **Jitter Buffer**：~500ms（48000 bytes），收到模型音频后先缓冲这么多再开始播放，平滑网络抖动
- **Barge-in**：收到 `speech_started` 时立即发送 `response.cancel` 并清空播放缓冲，实现打断

### 音频设备选择

```bash
# macOS：ffmpeg 通过 AVFoundation 枚举设备
node nvidia-voicechat-interactive.js --list-devices
# 输出：
#   [0] “iPhone 12”的麦克风
#   [1] MacBook Pro麦克风

# 指定内置麦克风
node nvidia-voicechat-interactive.js --input-device 1
```

默认使用 `:default`（系统默认输入设备），无需手动指定。

---

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key <key>` | `NVIDIA_API_KEY` 或 `NGC_API_KEY` 环境变量 | API 密钥 |
| `--base-url <url>` | `wss://api.ngc.nvidia.com/...` | WebSocket 端点 |
| `--persona <text>` | "You are a friendly..." | 语音助手人设 |
| `--timeout-ms <n>` | `30000` | 连接超时（毫秒） |
| `--input-device <id>` | 系统默认 (`:default`) | 麦克风设备索引 |
| `--output-device <id>` | 系统默认 | 扬声器设备索引 |
| `--list-devices` | - | 列出音频设备后退出 |
| `--help`, `-h` | - | 显示帮助 |

**示例**：

```bash
# 自定义人设
node nvidia-voicechat-interactive.js \
  --persona "You are a pirate captain. Speak like one."

# 自定义端点 + 超时
node nvidia-voicechat-interactive.js \
  --base-url wss://custom-endpoint/v1/realtime \
  --timeout-ms 60000
```

---

## 通过代理运行

脚本自动检测标准代理环境变量，无需额外配置：

```bash
# SOCKS5 代理（优先）
export ALL_PROXY=socks5://127.0.0.1:1080

# HTTP/HTTPS 代理（备用）
export http_proxy=http://127.0.0.1:1087
export https_proxy=http://127.0.0.1:1087

node nvidia-voicechat-interactive.js
```

**代理类型选择优先级**：`ALL_PROXY`（SOCKS5）→ `HTTPS_PROXY` → `HTTP_PROXY`

### 代理 vs 直连延迟对比

在测试环境（macOS, Node v23, 代理位于 `127.0.0.1`）下：

| 方式 | 平均延迟 | 说明 |
|------|----------|------|
| 直连 | ~1775ms | 基准 |
| HTTP 代理 | **~1659ms** | 快 ~6.5% |
| SOCKS5 代理 | ~1796ms | 与直连持平 |

代理不仅完全可用，HTTP 代理在某些网络环境下甚至略快于直连。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                     Terminal UI                          │
│  [●] Connected  [● REC]  [▶] Agent is speaking...       │
│  AI: Hello! How can I help you today?                   │
│  Press Ctrl+C to exit                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ ffmpeg   │───>│ micBuffer│───>│ input_audio_buffer│   │
│  │ (mic)    │    │ (11520B  │    │ .append (base64)  │   │
│  │ PCM16    │    │  frames) │    │                   │   │
│  │ 24kHz    │    └──────────┘    └────────┬──────────┘   │
│  └──────────┘                              │              │
│                                            ▼              │
│                                    ┌──────────────┐      │
│                                    │   WebSocket   │      │
│                                    │   wss://...   │      │
│                                    └──────┬───────┘      │
│                                           │              │
│                              ┌────────────┼────────────┐ │
│                              │            ▼            │ │
│                              │  response.audio.delta   │ │
│                              │  (base64 PCM16)         │ │
│                              └────────────┬────────────┘ │
│                                           │              │
│  ┌──────────┐    ┌──────────────┐         │              │
│  │ ffplay   │<───│ JitterBuffer │<────────┘              │
│  │(speaker) │    │ (~500ms      │                         │
│  │ PCM16    │    │  threshold)  │                         │
│  │ 24kHz    │    └──────────────┘                         │
│  └──────────┘                                              │
│                                                          │
│  ┌─────────────────────────┐                             │
│  │ Server VAD (服务端)      │                             │
│  │ speech_started → cancel │ ← Barge-in                  │
│  │ speech_stopped → commit │ ← Auto-process              │
│  └─────────────────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

### 数据流

1. **Mic → WebSocket**：ffmpeg 从 AVFoundation 采集 → stdout 输出 PCM16 → Node.js 按 240ms 帧切分 → base64 编码 → `input_audio_buffer.append`
2. **WebSocket → Speaker**：收到 `response.audio.delta` → 解码 base64 → Jitter Buffer 缓冲 500ms → ffplay stdin
3. **Barge-in**：收到 `speech_started` → 发送 `response.cancel` → 清空 Jitter Buffer → 杀掉 ffplay 进程 → 重新 spawn

---

## 碰到的问题与解决方案

### 问题 1：naudiodon 在 macOS + Node v23 上 segfault

**现象**：`node nvidia-voicechat-interactive.js --list-devices` 直接崩溃，SIGSEGV in `PaContext::PaContext`

**根因**：naudiodon（Node.js PortAudio 绑定）内置的 PortAudio 库与 macOS Sequoia 的 CoreAudio 权限模型不兼容，`Pa_Initialize()` 在执行音频设备枚举时直接段错误。Node v23 的 N-API 版本与 naudiodon 预编译二进制不匹配也放大了这个问题。

**解决**：放弃 naudiodon，改用 **ffmpeg + ffplay 子进程管道**：

```
麦克风: ffmpeg -f avfoundation -i :default -f s16le -ar 24000 -ac 1 -
扬声器: ffplay -f s16le -sample_rate 24000 -nodisp -autoexit -
```

优点：
- 无原生依赖，不依赖特定 Node 版本
- macOS/Linux/Windows 均可用（输入后端不同）
- ffmpeg 已处理所有底层音频 API 适配

### 问题 2：ffplay 不支持 `-ar` / `-ac` 参数

**现象**：`ffplay -f s16le -ar 24000 -ac 1` 报错 `Option not found`

**解决**：ffplay 的 raw audio demuxer 使用不同的参数名：
- `-sample_rate 24000`（不是 `-ar`）
- 默认单声道，无需指定 channels
- 完整命令：`ffplay -f s16le -sample_rate 24000 -nodisp -autoexit -`

### 问题 3：`--list-devices` 显示 "(none found)"

**现象**：设备枚举能运行但解析不出设备名

**根因**：ffmpeg stderr 输出格式为 `[AVFoundation indev @ 0x...] [0] Device Name`，原始正则 `/^\[\d+\]/` 无法匹配（因为行首是 `[AVFoundation` 前缀）

**解决**：改为过滤含 `[AVFoundation` 的行，再用 `/\[(\d+)\]\s+(.+)$/` 提取索引和名称。

### 问题 4：通过代理连接时 timeout

**现象**：`ws` 库本身不支持 `http_proxy` 环境变量

**解决**：安装 `https-proxy-agent` 和 `socks-proxy-agent`，在 WebSocket 构造函数中传入 `agent` 参数：

```js
const { HttpsProxyAgent } = require('https-proxy-agent');
const agent = new HttpsProxyAgent('http://127.0.0.1:1087');
const ws = new WebSocket(url, { agent });
```

### 问题 5：WebSocket 连上后无 session.updated

**现象**：发送 `functionId` 后收到 `session.created`，但 `session.updated` 永远不来

**根因**：只发送了 `functionId`，没有发送 `session.update` 消息。服务器不会主动推送 `session.updated`，必须由客户端发起配置。

**解决**：在 `functionId` 之后立即发送：

```json
{
  "type": "session.update",
  "session": {
    "audio": { "input": { "format": "pcm16" }, "output": { "format": "pcm16" } },
    "turn_detection": { "type": "server_vad" }
  }
}
```

---

## 使用场景

### 1. 实时语音助手

```bash
# 通用助手
node nvidia-voicechat-interactive.js

# 编程导师
node nvidia-voicechat-interactive.js \
  --persona "You are an expert programming mentor. Explain concepts clearly and give practical code examples."
```

### 2. 语言练习

```bash
# 英语口语陪练
node nvidia-voicechat-interactive.js \
  --persona "You are a patient English tutor. Correct grammar gently, maintain conversation flow, and use vocabulary appropriate for intermediate learners."
```

### 3. 角色扮演 / 面试模拟

```bash
# 模拟技术面试
node nvidia-voicechat-interactive.js \
  --persona "You are a senior engineering interviewer at a FAANG company. Ask system design and coding questions. Provide constructive feedback after each answer."
```

### 4. 通过代理加速（海外 API）

```bash
# HTTP 代理平均比直连快 ~6%，SOCKS5 持平
export http_proxy=http://127.0.0.1:1087
export ALL_PROXY=socks5://127.0.0.1:1080
node nvidia-voicechat-interactive.js
```

### 5. 开发调试

```bash
# 查看完整 WebSocket 消息类型
# 脚本默认会打印未知类型的消息名，方便理解协议

# 列出音频设备
node nvidia-voicechat-interactive.js --list-devices

# 自定义端点测试
node nvidia-voicechat-interactive.js \
  --base-url wss://localhost:9000/v1/realtime
```

### 6. 自动化 / 集成

```bash
# 通过环境变量注入配置
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_VOICECHAT_WS_URL="wss://proxy/v1/realtime"
node nvidia-voicechat-interactive.js --persona "You are a receptionist."

# 可配合 tmux 或 screen 保持会话
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `nvidia-voicechat-interactive.js` | 交互式语音对话主程序 |
| `nvidia-voicechat-test.js` | WebSocket 连通性测试 & 常量定义 |
| `test/nvidia-speech-models.test.js` | 单元测试（含 interactive 部分） |

## 可调参数

在 `nvidia-voicechat-interactive.js` 中可直接修改：

| 常量 | 默认值 | 含义 |
|------|--------|------|
| `JITTER_THRESHOLD_BYTES` | `48000` (~500ms) | 播放前缓冲量：越小延迟越低但越容易卡顿 |
| `FRAME_BYTES` | `11520` (~240ms) | 麦克风发送粒度 |
| `SAMPLE_RATE` | `24000` | 采样率（不要改，这是模型原生格式） |
