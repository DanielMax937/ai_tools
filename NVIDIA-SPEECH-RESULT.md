# NVIDIA Speech NIM 接入总结

> 会话时间：2026-05-08
> 涵盖模型：ASR（语音识别）、TTS（语音合成）、NMT（机器翻译）

---

## 一、模型概览

NVIDIA Speech NIM 提供三类语音 AI 微服务，通过 NVCF Cloud（`grpc.nvcf.nvidia.com:443`）或本地 Docker 容器暴露。

| 服务 | NVCF 模型 | 函数 ID | 协议 | Cloud 状态 |
|------|----------|---------|------|-----------|
| **TTS** | `ai-magpie-tts-zeroshot` | `55cf67bf-600f-4b04-8eac-12ed39537a08` | gRPC | ✅ 可用（仅英文） |
| **TTS** | `ai-magpie-tts-multilingual` | `877104f7-e885-42b9-8de8-f6e4c6303969` | gRPC | ❌ 挂起/超时 |
| **ASR** | `ai-canary-1b-asr` | `b0e8b4a5-217c-40b7-9b96-17d84e666317` | gRPC | ✅ 可用（en-US + zh-CN） |
| **ASR** | `ai-parakeet-ctc-1_1b-asr` | `1598d209-5e27-4d3c-8079-4751568b1081` | gRPC | ❌ 参数不匹配 |
| **ASR** | `ai-nemotron-asr-streaming` | `bb0837de-8c7b-481f-9ec8-ef5663e9c1fa` | gRPC | ❌ 参数不匹配 |
| **ASR** | `ai-parakeet-1_1b-rnnt-multilingual-asr` | `71203149-d3b7-4460-8231-1be2543a1fca` | gRPC | ❌ 工作线程超时 |
| **NMT** | `nvidia/riva-translate-4b-instruct-v1.1` | HTTP | integrate API | ✅ 可用（12 种语言） |
| **NMT** | `ai-megatron-1b-nmt` | `647147c1-9c23-496c-8304-2e29e7574510` | gRPC | ❌ 工作线程超时 |

---

## 二、接入方式

### 2.1 环境准备

```bash
pip install nvidia-riva-client
```

API Key 存放在项目根目录 `.env` 文件中：

```
NVIDIA_API_KEY=nvapi-...
```

### 2.2 TTS（语音合成）— gRPC

```python
import socket, struct
import riva.client
from riva.client.proto import riva_audio_pb2

# 强制 IPv4 DNS 解析（macOS 上 IPv6 会导致 gRPC 挂起）
addrs = socket.getaddrinfo(
    'grpc.nvcf.nvidia.com', 443,
    socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
)
ipv4 = addrs[0][4][0]

# 创建认证（函数 ID + Bearer token 作为 gRPC metadata）
auth = riva.client.Auth(
    uri=f'{ipv4}:443',
    use_ssl=True,
    metadata_args=[
        ['function-id', '55cf67bf-600f-4b04-8eac-12ed39537a08'],
        ['authorization', f'Bearer {api_key}'],
    ],
    options=[('grpc.ssl_target_name_override', 'grpc.nvcf.nvidia.com')],
)

service = riva.client.SpeechSynthesisService(auth)

# 内置语音合成
response = service.synthesize(
    text='Hello world',
    voice_name='Magpie-ZeroShot.Female-1',
    language_code='en-US',
    encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=22050,
)

# gRPC 返回 RAW PCM，需手动添加 WAV 文件头
wav = pcm_to_wav(response.audio, sample_rate=22050)
```

**语音克隆模式：**

```python
from riva.client.proto import riva_tts_pb2

# 语音克隆需要使用低级 API 来设置 zero_shot_data.sample_rate_hz
# （high-level API 不设置该字段，服务器会 reject）
with open('prompt.wav', 'rb') as f:
    audio_bytes = f.read()

# 从 WAV 头检测音频提示的采样率
audio_sr = struct.unpack('<I', audio_bytes[24:28])[0]

req = riva_tts_pb2.SynthesizeSpeechRequest(
    text='Hello with cloned voice',
    language_code='en-US',
    encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=22050,
)
req.zero_shot_data.audio_prompt = audio_bytes
req.zero_shot_data.sample_rate_hz = audio_sr  # ← 必须设置！
req.zero_shot_data.encoding = riva_audio_pb2.AudioEncoding.LINEAR_PCM
req.zero_shot_data.quality = 20

response = service.stub.Synthesize(req, metadata=auth.get_auth_metadata())
```

**可用语音：**

```
Magpie-ZeroShot.Female-1       Magpie-ZeroShot.Female-Neutral
Magpie-ZeroShot.Female-Angry   Magpie-ZeroShot.Female-Fearful
Magpie-ZeroShot.Female-Calm    Magpie-ZeroShot.Female-Happy
Magpie-ZeroShot.Male-1         Magpie-ZeroShot.Male-Calm
Magpie-ZeroShot.Male-Neutral   Magpie-ZeroShot.Male-Angry
Magpie-ZeroShot.Male-Fearful
```

### 2.3 ASR（语音识别）— gRPC

```python
import array
from riva.client.proto import riva_asr_pb2

auth = riva.client.Auth(...)  # 函数 ID: b0e8b4a5-217c-40b7-9b96-17d84e666317
service = riva.client.ASRService(auth)

# 读取 WAV，提取 PCM，重采样到 16kHz
# (ASR 模型要求 16kHz 采样率)
pcm_bytes, sr, bits = read_wav_pcm('audio.wav')
if sr != 16000:
    pcm_bytes = resample_pcm(pcm_bytes, sr, 16000)

config = riva_asr_pb2.RecognitionConfig(
    encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code='en-US',  # 也支持 'zh-CN'
    enable_automatic_punctuation=True,
)

response = service.offline_recognize(pcm_bytes, config)
transcript = response.results[0].alternatives[0].transcript
```

**语言支持：** `canary-1b` 模型仅支持 `en-US` 和 `zh-CN`。尝试 `ja`、`fr` 等会返回 `INVALID_ARGUMENT: Unavailable model requested`。

### 2.4 NMT（机器翻译）— HTTP

```python
import json, urllib.request

body = {
    'model': 'nvidia/riva-translate-4b-instruct-v1.1',
    'messages': [
        {'role': 'system', 'content': 'You are an expert at translating text from English to Simplified Chinese.'},
        {'role': 'user', 'content': 'What is the Simplified Chinese translation of the sentence: Hello world?'},
    ],
    'max_tokens': 512,
    'temperature': 0,
}

req = urllib.request.Request(
    'https://integrate.api.nvidia.com/v1/chat/completions',
    data=json.dumps(body).encode(),
    headers={
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    },
)
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())
    translation = result['choices'][0]['message']['content']
```

**支持的 12 种语言：**

```
en     English               pt-BR  Brazilian Portuguese
de     German                ru     Russian
es-ES  European Spanish      zh-CN  Simplified Chinese
es-US  LATAM Spanish         zh-TW  Traditional Chinese
fr     French                ja     Japanese
                             ko     Korean
                             ar     Arabic
```

---

## 三、问题和解决方案

### 3.1 Node.js HTTP pexec 端点返回 500

**现象：** `https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{id}` 始终返回 500。

**原因：** NVCF pexec HTTP REST 端点不支持 Speech NIM 模型。这些模型仅通过 **gRPC** (`grpc.nvcf.nvidia.com:443`) 暴露。

**解决：** 对 TTS 和 ASR 使用 Python `nvidia-riva-client` gRPC，对 NMT 使用 `integrate.api.nvidia.com` HTTP。

### 3.2 macOS 上 gRPC 挂起（IPv6 问题）

**现象：** gRPC 连接无限期挂起，无报错，无超时。

**原因：** macOS 系统 DNS 解析器可能返回 IPv6 地址，但 NVCF gRPC 端点不支持 IPv6，导致 TCP 连接静默失败。

**解决：** 强制 IPv4 DNS 解析：
```python
addrs = socket.getaddrinfo(host, port, socket.AF_INET, ...)
ipv4 = addrs[0][4][0]
auth = riva.client.Auth(
    uri=f'{ipv4}:443',
    options=[('grpc.ssl_target_name_override', host)],  # TLS 证书校验仍用原主机名
)
```

### 3.3 语音克隆返回 `config format doesn't match`

**现象：** 使用 high-level `service.synthesize()` API 进行语音克隆时报错格式不匹配。

**原因：** `service.synthesize()` 内部的 `zero_shot_data` 构建**不设置 `sample_rate_hz` 字段**。gRPC 服务器需要该字段来验证音频提示格式。

**解决：** 使用低级 `service.stub.Synthesize()` 并手动设置 `req.zero_shot_data.sample_rate_hz` 为音频提示文件的实际采样率（从 WAV 文件头读取）。

### 3.4 gRPC 响应返回原始 PCM，非 WAV

**现象：** TTS `response.audio` 数据不是有效 WAV 文件。

**原因：** gRPC `SynthesizeSpeechResponse.audio` 包含**原始 PCM 数据**，无 WAV 文件头。

**解决：** 手动添加 44 字节 WAV 文件头：
```python
def pcm_to_wav(pcm, sample_rate=22050, bits=16, channels=1):
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(pcm), b'WAVE', b'fmt ', 16,
        1, channels, sample_rate, sample_rate * channels * bits // 8,
        channels * bits // 8, bits, b'data', len(pcm),
    )
    return header + pcm
```

### 3.5 ASR 需要 16kHz 采样率

**现象：** 部分模型报 `Unavailable model requested given these parameters: sample_rate=22050`。

**原因：** ASR 模型（canary-1b）要求 16kHz 输入。

**解决：** 自动将音频重采样到 16kHz：
```python
src = array.array('h'); src.frombytes(pcm)
ratio = src_rate / 16000.0
for i in range(int(len(src) / ratio)):
    dst.append(src[min(int(i * ratio), len(src) - 1)])
```

### 3.6 多语言 TTS 模型不可达

**现象：** `ai-magpie-tts-multilingual` 通过 gRPC 挂起，通过 HTTP pexec 超时。

**原因：** 尽管模型在 NVCF 上列为 ACTIVE，但推理后端无响应。该模型可能需要本地 NIM Docker 部署。

**解决：** 中文 TTS 目前无法通过 NVCF Cloud 实现。需要本地部署：
```bash
docker run -it --rm --gpus all \
  -e NGC_API_KEY \
  -p 9000:9000 -p 50051:50051 \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
```

### 3.7 NVCF 速率限制

**现象：** 连续约 3-4 次 TTS 请求后收到 `RESOURCE_EXHAUSTED: exceeded rate limit`；响应时间从约 5 秒逐渐增加到约 118 秒。

**原因：** NVCF Cloud 免费套餐强制实施请求速率限制。

**解决：** 请求之间间隔 8-10 秒：
```python
time.sleep(8)  # rate-limit cooldown
```

### 3.8 字段命名：camelCase vs snake_case

**现象：** HTTP pexec 使用 camelCase 字段名时服务器无响应。

**原因：** NVCF 将请求 JSON 透传给 gRPC 后端。gRPC proto 使用 snake_case（`zero_shot_data`、`voice_name`、`language_code`），camelCase 字段（`zeroShotData`）会被静默丢弃。

**解决：** 始终在 JSON/gRPC 请求中使用 snake_case 字段名与 proto 定义匹配。

### 3.9 采样率不匹配

**现象：** 早期代码使用 `sample_rate_hz: 48000`。

**原因：** `magpie-tts-zeroshot` 的 native 输出是 22.05 kHz（如 NVIDIA 文档所述）。

**解决：** 对 TTS 输出使用 `sample_rate_hz: 22050`。

---

## 四、脚本文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `nvidia-tts-grpc-test.py` | gRPC | TTS 测试：内置语音 + 语音克隆 |
| `nvidia-asr-grpc-test.py` | gRPC | ASR 测试：WAV 文件转文字 |
| `nvidia-nmt-grpc-test.py` | HTTP | NMT 测试：12 种语言互译 |
| `nvidia-speech-pipeline.py` | 混合 | 复合流水线：TTS → ASR（+ NMT） |
| `nvidia-tts-zeroshot-test.js` | HTTP | ❌ 不可用 — NVCF pexec 返回 500 |

### 运行示例

```bash
# TTS
python3 nvidia-tts-grpc-test.py --voice Magpie-ZeroShot.Male-1
python3 nvidia-tts-grpc-test.py --audio-prompt voice.wav --quality 20

# ASR
python3 nvidia-asr-grpc-test.py --audio recording.wav
python3 nvidia-asr-grpc-test.py --audio recording.wav --language zh-CN

# NMT
python3 nvidia-nmt-grpc-test.py --text "Hello world" --from en --to zh-CN
python3 nvidia-nmt-grpc-test.py --text "你好世界" --from zh-CN --to fr

# Pipeline
python3 nvidia-speech-pipeline.py -s 2          # 英文→英文
python3 nvidia-speech-pipeline.py -s 4          # 中文→英文 (NMT)
python3 nvidia-speech-pipeline.py --save-audio  # 保存生成音频
```

---

## 五、使用场景

### 5.1 语音合成 (TTS)

- **有声内容生成：** 将文本自动转化为自然语音，适用于播客、有声书
- **语音助手：** 为 AI 助手提供语音输出能力
- **多情感表达：** 使用不同语音风格（愤怒、平静、快乐、恐惧等）
- **语音克隆：** 用 3-10 秒的音频样本克隆特定人声
- **本地化语音：** 用多语言 TTS（需本地 NIM）生成各语种语音

### 5.2 语音识别 (ASR)

- **会议转录：** 自动将会议录音转为文字记录
- **字幕生成：** 为视频/直播生成实时字幕
- **语音搜索：** 将语音查询转为文本进行搜索
- **客服质检：** 将客服通话转文字进行分析
- **中英双语识别：** canary-1b 支持 en-US 和 zh-CN

### 5.3 机器翻译 (NMT)

- **多语言内容：** 翻译网站、文档、营销材料
- **实时翻译：** 结合 ASR + NMT + TTS 实现语音到语音翻译
- **跨语言客服：** 将客户消息翻译为客服语言
- **内容本地化：** 12 种语言互译，覆盖全球主要市场

### 5.4 复合流水线

| 场景 | 流程 | 状态 |
|------|------|------|
| 英文→英文 | TTS 生成语音 → ASR 转回文字 | ✅ 验证通过，100% 准确率 |
| 中文→中文 | TTS → ASR | ❌ 中文 TTS 不可用 |
| 英文→中文 | NMT 翻译 → 中文 TTS → ASR | ❌ 中文 TTS 不可用 |
| 中文→英文 | NMT 翻译 → 英文 TTS → 英文 ASR | ✅ 应可用（速率限制未测） |

### 5.5 完整语音翻译管道（理想场景）

```
语音输入(中文) → ASR(zh-CN) → 文字 → NMT(zh→en) → TTS(en) → 语音输出(英文)
                      ✅                ✅            ✅
```

所有组件在 NVCF Cloud 上均可用，但中文 TTS（反方向）需要本地 NIM 部署。
