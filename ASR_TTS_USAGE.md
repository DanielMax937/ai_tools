# ASR / TTS Usage Reference

This document summarizes the ASR, TTS, and realtime voice models currently used or tested in this repository.

No plaintext API keys are included here. Use the environment variable names listed below.

## Environment Variables

| Variable | Used By | Purpose |
|---|---|---|
| `OPENAI_API_KEY_MIRACLE` | `audio/audio_transcriber.py`, `audio/chatgpt.py` | OpenAI-compatible proxy key |
| `OPENAI_API_KEY` | optional OpenAI direct usage | Standard OpenAI key |
| `NVIDIA_API_KEY` | NVIDIA ASR/TTS/VoiceChat scripts | NVIDIA / NGC API key |
| `NGC_API_KEY` | `nvidia-voicechat-test.js`, local NIM Docker | Alternative NVIDIA key name |
| `NVIDIA_VOICECHAT_WS_URL` | `nvidia-voicechat-test.js`, `nvidia-voicechat-interactive.js` | Override VoiceChat WebSocket endpoint |
| `NVIDIA_TTS_URL` | `nvidia-tts-zeroshot-test.js` | Override HTTP TTS endpoint |

## ASR: OpenAI Whisper

### Script

```bash
audio/audio_transcriber.py
```

### Model

```text
whisper-1
```

### Base URL

The script creates a default `OpenAI` client. If no custom client base URL is provided, it uses the OpenAI SDK default endpoint:

```text
https://api.openai.com/v1
```

### API Key

```text
OPENAI_API_KEY_MIRACLE
```

The usage docs mention `OPENAI_API_KEY_MIRACLE`; depending on client initialization, make sure this is mapped to the OpenAI client key expected by the script/environment.

### Usage

```bash
cd /Users/daniel/Desktop/git/ai_tools
python3 audio/audio_transcriber.py "path/to/audio.mp3"
python3 audio/audio_transcriber.py "path/to/audio.mp3" --language zh
python3 audio/audio_transcriber.py "path/to/audio.mp3" --prompt "金融播客访谈"
python3 audio/audio_transcriber.py "path/to/audio.mp3" --output-format json
```

### Notes

- Supports `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`.
- Splits large files into smaller chunks with `pydub`.
- Saves text or JSON transcription output next to the input file.

## ASR: GPT-4o Transcribe via MiraclePlus Proxy

### Script

```bash
audio/chatgpt.py
```

### Model

```text
gpt-4o-transcribe
```

### Base URL

```text
http://openai-proxy.miracleplus.com/v1
```

### API Key

```text
OPENAI_API_KEY_MIRACLE
```

### Usage

The script currently hardcodes an example input path:

```python
audio_file = open("/Users/caoxiaopeng/Desktop/ai_tools/downloads/test.mp3", "rb")
```

Run:

```bash
cd /Users/daniel/Desktop/git/ai_tools
python3 audio/chatgpt.py
```

To reuse it, change the `audio_file` path or refactor it into a CLI argument.

## ASR: NVIDIA Riva / NIM gRPC

### Script

```bash
nvidia-asr-grpc-test.py
```

### Protocol

```text
gRPC
```

### Base URL / Host

```text
grpc.nvcf.nvidia.com:443
```

The script resolves the host to IPv4 and sets TLS override to `grpc.nvcf.nvidia.com` to avoid macOS IPv6 hangs.

### API Key

```text
NVIDIA_API_KEY
```

### Models

| CLI Key | Model Name | NVCF Function ID | Status From Project Notes |
|---|---|---|---|
| `canary-1b` | `ai-canary-1b-asr` | `b0e8b4a5-217c-40b7-9b96-17d84e666317` | usable; supports `en-US` and `zh-CN` |
| `nemotron` | `ai-nemotron-asr-streaming` | `bb0837de-8c7b-481f-9ec8-ef5663e9c1fa` | parameter mismatch in prior tests |
| `parakeet-1.1b` | `ai-parakeet-ctc-1_1b-asr` | `1598d209-5e27-4d3c-8079-4751568b1081` | parameter mismatch in prior tests |
| `parakeet-0.6b` | `ai-parakeet-ctc-0_6b-asr` | `d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965` | listed in script |
| `parakeet-multilingual` | `ai-parakeet-1_1b-rnnt-multilingual-asr` | `71203149-d3b7-4460-8231-1be2543a1fca` | timeout in prior tests |

### Recommended Model

```text
canary-1b
```

### Usage

```bash
cd /Users/daniel/Desktop/git/ai_tools
NVIDIA_API_KEY=nvapi-... python3 nvidia-asr-grpc-test.py --audio recording.wav
NVIDIA_API_KEY=nvapi-... python3 nvidia-asr-grpc-test.py --audio recording.wav --language zh-CN
NVIDIA_API_KEY=nvapi-... python3 nvidia-asr-grpc-test.py --list-models
```

### Audio Requirements

- Best input format: WAV, PCM16 mono.
- The script resamples to `16000 Hz` for ASR.
- `canary-1b` supports `en-US` and `zh-CN`.

## TTS: NVIDIA Magpie TTS Zero-shot gRPC

### Script

```bash
nvidia-tts-grpc-test.py
```

### Protocol

```text
gRPC
```

### Base URL / Host

```text
grpc.nvcf.nvidia.com:443
```

### API Key

```text
NVIDIA_API_KEY
```

### Model

```text
ai-magpie-tts-zeroshot
```

### NVCF Function ID

```text
55cf67bf-600f-4b04-8eac-12ed39537a08
```

### Default Voice

```text
Magpie-ZeroShot.Female-1
```

### Available Voices

```text
Magpie-ZeroShot.Female-1
Magpie-ZeroShot.Female-Neutral
Magpie-ZeroShot.Female-Angry
Magpie-ZeroShot.Female-Fearful
Magpie-ZeroShot.Female-Calm
Magpie-ZeroShot.Female-Happy
Magpie-ZeroShot.Male-1
Magpie-ZeroShot.Male-Calm
Magpie-ZeroShot.Male-Neutral
Magpie-ZeroShot.Male-Angry
Magpie-ZeroShot.Male-Fearful
```

### Usage

```bash
cd /Users/daniel/Desktop/git/ai_tools
NVIDIA_API_KEY=nvapi-... python3 nvidia-tts-grpc-test.py \
  --text "Hello, this is a TTS test." \
  --voice Magpie-ZeroShot.Female-1 \
  --output output.wav
```

List voices:

```bash
python3 nvidia-tts-grpc-test.py --list-voices
```

Voice cloning mode:

```bash
NVIDIA_API_KEY=nvapi-... python3 nvidia-tts-grpc-test.py \
  --text "Hello with cloned voice." \
  --audio-prompt prompt.wav \
  --quality 20 \
  --output cloned.wav
```

### Notes

- Project notes say this cloud model is usable but English-only.
- Native output is raw PCM at `22050 Hz`; the script wraps it with a WAV header.
- For voice cloning, the lower-level gRPC API is used to set `zero_shot_data.sample_rate_hz`.

## TTS: NVIDIA Magpie TTS Zero-shot HTTP Test

### Script

```bash
nvidia-tts-zeroshot-test.js
```

### Protocol

```text
HTTP NVCF pexec
```

### Cloud Base URL

```text
https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/55cf67bf-600f-4b04-8eac-12ed39537a08
```

### Local Base URL

```text
http://localhost:9000/v1/audio/synthesize
```

### API Key

```text
NVIDIA_API_KEY
```

### Model / Function ID

```text
ai-magpie-tts-zeroshot
55cf67bf-600f-4b04-8eac-12ed39537a08
```

### Usage

```bash
cd /Users/daniel/Desktop/git/ai_tools
npm run test:tts-zeroshot -- --text "Hello world" --output output.wav
```

### Notes

Project notes say the HTTP pexec path was not usable for this model and returned server errors. Prefer the gRPC script above.

## TTS: NVIDIA Magpie TTS Multilingual

### Model

```text
ai-magpie-tts-multilingual
```

### NVCF Function ID

```text
877104f7-e885-42b9-8de8-f6e4c6303969
```

### Status

Project notes mark this as unavailable on NVCF Cloud due to hang/timeout. Use local NIM deployment if multilingual TTS is required.

## Realtime Speech-to-Speech: NVIDIA Nemotron VoiceChat

### Scripts

```bash
nvidia-voicechat-test.js
nvidia-voicechat-interactive.js
```

### Model

```text
nvidia/nemotron-voicechat
```

### NVCF Function ID

```text
42c86b5f-545a-4b2f-a83b-90fd71da9912
```

### Cloud WebSocket URL

```text
wss://api.ngc.nvidia.com/v2/predict/artifactname/websocket/v1/realtime?nv-function-id=42c86b5f-545a-4b2f-a83b-90fd71da9912
```

### Queue / Health URL

```text
https://api.ngc.nvidia.com/v2/predict/queues/models/qc69jvmznzxy/nemotron-voicechat
```

### Local WebSocket URL

```text
ws://localhost:9000/v1/realtime
```

### API Key

```text
NVIDIA_API_KEY
NGC_API_KEY
```

### Usage

Health check:

```bash
cd /Users/daniel/Desktop/git/ai_tools
NVIDIA_API_KEY=nvapi-... npm run test:voicechat -- --mode health
```

WebSocket test:

```bash
NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-test.js --mode websocket
```

Interactive voice chat:

```bash
NVIDIA_API_KEY=nvapi-... npm run voicechat:interactive
```

Local NIM:

```bash
node nvidia-voicechat-test.js \
  --base-url ws://localhost:9000/v1/realtime \
  --mode websocket
```

### Audio Format

```text
Input:  PCM16, mono, 24 kHz
Output: PCM16, mono, 24 kHz
```

### Notes

- This is not a plain ASR or TTS model. It is full-duplex speech-to-speech.
- It returns both model audio and text transcript events.
- Project notes describe it as English-only, single voice, no tool calling.

## Recommended Choices

| Need | Recommended Script | Model |
|---|---|---|
| General file transcription | `audio/audio_transcriber.py` | `whisper-1` |
| Higher quality proxy transcription | `audio/chatgpt.py` | `gpt-4o-transcribe` |
| NVIDIA ASR, including Chinese | `nvidia-asr-grpc-test.py` | `ai-canary-1b-asr` |
| English TTS | `nvidia-tts-grpc-test.py` | `ai-magpie-tts-zeroshot` |
| Realtime voice assistant | `nvidia-voicechat-interactive.js` | `nvidia/nemotron-voicechat` |

## Dependency Notes

Python:

```bash
pip install openai pydub nvidia-riva-client grpcio
```

Node:

```bash
npm install
```

The project uses `ws` for WebSocket voice chat and NVIDIA Riva Python client for gRPC ASR/TTS.
