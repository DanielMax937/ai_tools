#!/usr/bin/env python3
"""
NVIDIA Speech Pipeline — Composite Test Script

Chains TTS → ASR (and optionally NMT) to verify end-to-end speech
processing across four scenarios:

  1. 中文→中文  Chinese TTS → Chinese ASR (roundtrip)
  2. 英文→英文  English TTS → English ASR (roundtrip)
  3. 英文→中文  English → NMT → Chinese TTS → Chinese ASR
  4. 中文→英文  Chinese → NMT → English TTS → English ASR

Each scenario compares the original input text with the final ASR
transcription and reports the word/character accuracy.

Requires: pip install nvidia-riva-client
          NVIDIA_API_KEY environment variable or .env file
"""

import os
import sys
import time
import struct
import socket
import argparse
import json
import urllib.request
import grpc
import riva.client
from riva.client.proto import riva_tts_pb2, riva_asr_pb2, riva_audio_pb2


# ── Constants ─────────────────────────────────────────────────────────
GRPC_HOST = 'grpc.nvcf.nvidia.com'
GRPC_PORT = 443

# NVCF function IDs
TTS_ZEROSHOT_ID = '55cf67bf-600f-4b04-8eac-12ed39537a08'       # English only
TTS_MULTILINGUAL_ID = '877104f7-e885-42b9-8de8-f6e4c6303969'    # 12+ languages
ASR_FUNCTION_ID = 'b0e8b4a5-217c-40b7-9b96-17d84e666317'        # canary-1b

# NMT (uses HTTP chat completions API, not gRPC)
NMT_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
NMT_MODEL = 'nvidia/riva-translate-4b-instruct-v1.1'

LANG_MAP = {
    'zh': {'name': 'Chinese',       'nm': 'Simplified Chinese', 'tts': 'zh-CN',
           'voice_en': 'Magpie-ZeroShot.Female-1',
           'voice_ml': 'Magpie-Multilingual.ZH-CN.Aria'},
    'en': {'name': 'English',        'nm': 'English',            'tts': 'en-US',
           'voice_en': 'Magpie-ZeroShot.Female-1',
           'voice_ml': 'Magpie-Multilingual.EN-US.Aria'},
}

TEST_TEXTS = {
    'zh': '今天天气真好，阳光明媚，微风轻拂，让人心情愉快。',
    'en': 'The weather is beautiful today, with clear skies and a gentle breeze.',
}


# ── Utilities ─────────────────────────────────────────────────────────
def load_env_file(env_path='.env'):
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value


def resolve_ipv4():
    addrs = socket.getaddrinfo(GRPC_HOST, GRPC_PORT, socket.AF_INET,
                               socket.SOCK_STREAM, socket.IPPROTO_TCP)
    return addrs[0][4][0]


def create_auth(api_key, function_id):
    ipv4 = resolve_ipv4()
    return riva.client.Auth(
        uri=f'{ipv4}:{GRPC_PORT}',
        use_ssl=True,
        metadata_args=[
            ['function-id', function_id],
            ['authorization', f'Bearer {api_key}'],
        ],
        options=[('grpc.ssl_target_name_override', GRPC_HOST)],
    )


def pcm_to_wav(pcm, sample_rate=22050, bits=16, channels=1):
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(pcm), b'WAVE', b'fmt ', 16,
        1, channels, sample_rate, byte_rate, block_align, bits,
        b'data', len(pcm),
    )
    return header + pcm


def read_wav_pcm(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF':
        return data, 16000, 16
    sr = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if chunk_id == b'data':
            return data[pos + 8:pos + 8 + chunk_size], sr, bits
        pos += 8 + chunk_size
    return data[44:], sr, bits


def resample_pcm(pcm, src_rate, dst_rate=16000):
    if src_rate == dst_rate:
        return pcm
    import array
    src = array.array('h'); src.frombytes(pcm)
    ratio = src_rate / dst_rate
    dst = array.array('h')
    for i in range(int(len(src) / ratio)):
        dst.append(src[min(int(i * ratio), len(src) - 1)])
    return dst.tobytes()


# ── NMT (HTTP) ───────────────────────────────────────────────────────
def translate(text, source_lang, target_lang, api_key):
    """Translate via integrate.api.nvidia.com chat completions API."""
    src_name = LANG_MAP[source_lang]['nm']
    tgt_name = LANG_MAP[target_lang]['nm']

    body = {
        'model': NMT_MODEL,
        'messages': [
            {'role': 'system', 'content': f'You are an expert at translating text from {src_name} to {tgt_name}.'},
            {'role': 'user', 'content': f'What is the {tgt_name} translation of the sentence: {text}?'},
        ],
        'max_tokens': 512,
        'temperature': 0,
    }
    req = urllib.request.Request(
        NMT_URL,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if data.get('choices'):
        return data['choices'][0]['message']['content'].strip()
    return ''


# ── TTS (gRPC) ────────────────────────────────────────────────────────
def synthesize(text, language, cfg, api_key):
    """Generate speech via TTS gRPC, return WAV bytes.

    NOTE: Only English is supported on NVCF Cloud via the magpie-tts-zeroshot
    model. Chinese TTS requires the magpie-tts-multilingual model which needs
    a local NIM Docker deployment.
    """
    is_english = language.startswith('en')

    if not is_english:
        raise RuntimeError(
            f'Chinese TTS ({language}) is not available on NVCF Cloud.\n'
            f'  → The magpie-tts-zeroshot model only supports English.\n'
            f'  → For Chinese TTS, deploy the magpie-tts-multilingual NIM container locally.\n'
            f'  → Docker: nvcr.io/nim/nvidia/magpie-tts-multilingual:latest'
        )

    # English: use zeroshot model (proven to work)
    auth = create_auth(api_key, TTS_ZEROSHOT_ID)
    service = riva.client.SpeechSynthesisService(auth)

    response = service.synthesize(
        text=text,
        voice_name=cfg['voice_en'],
        language_code=language,
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=22050,
    )
    if response.audio:
        return pcm_to_wav(response.audio, sample_rate=22050)
    raise RuntimeError('Empty TTS response')


# ── ASR (gRPC) ────────────────────────────────────────────────────────
def transcribe(audio_wav_bytes, language, api_key):
    """Transcribe WAV audio via ASR gRPC, return text."""
    auth = create_auth(api_key, ASR_FUNCTION_ID)
    service = riva.client.ASRService(auth)

    pcm, sr, bits = read_wav_pcm_from_bytes(audio_wav_bytes)
    if sr != 16000:
        pcm = resample_pcm(pcm, sr, 16000)
        sr = 16000

    config = riva_asr_pb2.RecognitionConfig(
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sr,
        language_code=language,
        enable_automatic_punctuation=True,
        max_alternatives=1,
    )
    response = service.offline_recognize(pcm, config)

    transcript = ''
    if response.results:
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript
    return transcript.strip()


def read_wav_pcm_from_bytes(data):
    """Like read_wav_pcm but from bytes instead of file path."""
    if data[:4] != b'RIFF':
        return data, 16000, 16
    sr = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if chunk_id == b'data':
            return data[pos + 8:pos + 8 + chunk_size], sr, bits
        pos += 8 + chunk_size
    return data[44:], sr, bits


# ── Accuracy ──────────────────────────────────────────────────────────
def calc_accuracy(original, transcribed):
    """Calculate character-level accuracy (Levenshtein ratio)."""
    original = original.strip().lower()
    transcribed = transcribed.strip().lower()
    if not original:
        return 0.0

    # Simple character overlap (for CJK: char-level; for Latin: word-level)
    if any('\u4e00' <= c <= '\u9fff' for c in original):
        # Chinese: character-level
        orig_chars = list(original.replace(' ', '').replace(',', '').replace('.', ''))
        trans_chars = list(transcribed.replace(' ', '').replace(',', '').replace('.', ''))
        matches = sum(1 for c in orig_chars if c in trans_chars)
        return matches / max(len(orig_chars), 1)
    else:
        # English: word-level
        import re
        orig_words = re.findall(r'\w+', original)
        trans_words = re.findall(r'\w+', transcribed)
        if not orig_words:
            return 0.0
        matches = sum(1 for w in orig_words if w in trans_words)
        return matches / max(len(orig_words), 1)


# ── Pipeline scenarios ────────────────────────────────────────────────
def run_pipeline(api_key, scenarios, save_audio=False):
    """Run all selected TTS→ASR pipeline scenarios."""
    results = []

    for idx, (label, source_lang, target_lang, use_nmt) in enumerate(scenarios, 1):
        # Rate-limit: add delay between scenarios (NVCF free tier)
        if idx > 1:
            delay_s = 8
            print(f'(waiting {delay_s}s for rate-limit...)\n')
            time.sleep(delay_s)

        print(f'Scenario {idx}: {label}')
        print('-' * 50)

        src_cfg = LANG_MAP[source_lang]
        tgt_cfg = LANG_MAP[target_lang]

        # Step 1: get text (translate if needed)
        source_text = TEST_TEXTS[source_lang]
        tts_text = source_text

        if use_nmt:
            print(f'  [NMT] Translating {src_cfg["name"]} → {tgt_cfg["name"]}...')
            started = time.time()
            try:
                tts_text = translate(source_text, source_lang, target_lang, api_key)
                print(f'     "{source_text[:50]}..."')
                print(f'  →  "{tts_text[:50]}..."')
                print(f'     ({int((time.time() - started) * 1000)}ms)')
            except Exception as exc:
                print(f'  ✗ NMT failed: {exc}')
                results.append({'label': label, 'ok': False, 'stage': 'nmt', 'error': str(exc)})
                continue

        # Step 2: TTS
        tts_lang = tgt_cfg['tts']
        print(f'  [TTS] Synthesizing {tgt_cfg["name"]} speech ({tts_lang})...')
        started = time.time()
        try:
            audio_wav = synthesize(tts_text, tts_lang, tgt_cfg, api_key)
            tts_ms = int((time.time() - started) * 1000)
            print(f'     {len(audio_wav)} bytes ({tts_ms}ms)')
            if save_audio:
                fname = f'/tmp/pipeline-{idx}-tts.wav'
                with open(fname, 'wb') as f:
                    f.write(audio_wav)
                print(f'     Saved → {fname}')
        except Exception as exc:
            print(f'  ✗ TTS failed: {exc}')
            results.append({'label': label, 'ok': False, 'stage': 'tts', 'error': str(exc)})
            continue

        # Step 3: ASR
        asr_lang = tgt_cfg['tts']
        print(f'  [ASR] Transcribing...')
        started = time.time()
        try:
            transcript = transcribe(audio_wav, asr_lang, api_key)
            asr_ms = int((time.time() - started) * 1000)
            accuracy = calc_accuracy(tts_text, transcript)
            print(f'     "{transcript[:80]}{"..." if len(transcript)>80 else ""}"')
            print(f'     ({asr_ms}ms, accuracy={accuracy:.1%})')
        except Exception as exc:
            print(f'  ✗ ASR failed: {exc}')
            results.append({'label': label, 'ok': False, 'stage': 'asr', 'error': str(exc)})
            continue

        result = {
            'label': label,
            'ok': True,
            'source_text': source_text,
            'tts_text': tts_text,
            'transcript': transcript,
            'tts_ms': tts_ms,
            'asr_ms': asr_ms,
            'accuracy': accuracy,
        }
        results.append(result)
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print('=' * 60)
    print('PIPELINE SUMMARY')
    print('=' * 60)
    for r in results:
        if r['ok']:
            print(f'  [{r["label"]}]  accuracy={r["accuracy"]:.1%}  (TTS={r["tts_ms"]}ms  ASR={r["asr_ms"]}ms)')
            print(f'    Expected:  "{r["tts_text"][:60]}{"..." if len(r["tts_text"])>60 else ""}"')
            print(f'    Got:       "{r["transcript"][:60]}{"..." if len(r["transcript"])>60 else ""}"')
        else:
            print(f'  [{r["label"]}]  FAILED at {r["stage"]}: {r["error"][:80]}')

    ok = sum(1 for r in results if r['ok'])
    print(f'\n  Passed: {ok}/{len(results)}')
    return results


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description='NVIDIA Speech Pipeline: TTS → ASR (optionally with NMT)',
    )
    parser.add_argument('--api-key', default=os.environ.get('NVIDIA_API_KEY'))
    parser.add_argument('--scenario', '-s', choices=['1','2','3','4','all'],
                        default='all', help='Which scenario to run (default: all)')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save generated TTS audio to /tmp/pipeline-*.wav')
    args = parser.parse_args()

    if not args.api_key:
        print('ERROR: Set NVIDIA_API_KEY or pass --api-key.', file=sys.stderr)
        sys.exit(1)

    all_scenarios = [
        ('中文→中文',    'zh', 'zh', False),   # 1: zh TTS → zh ASR
        ('英文→英文',    'en', 'en', False),   # 2: en TTS → en ASR
        ('英文→中文 NMT', 'en', 'zh', True),   # 3: en→zh NMT → zh TTS → zh ASR
        ('中文→英文 NMT', 'zh', 'en', True),   # 4: zh→en NMT → en TTS → en ASR
    ]

    if args.scenario == 'all':
        scenarios = all_scenarios
    else:
        scenarios = [all_scenarios[int(args.scenario) - 1]]

    run_pipeline(args.api_key, scenarios, save_audio=args.save_audio)


if __name__ == '__main__':
    main()
