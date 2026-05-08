#!/usr/bin/env python3
"""
NVIDIA ASR NIM — gRPC Test Script

Tests NVIDIA Automatic Speech Recognition (ASR) models via gRPC to
grpc.nvcf.nvidia.com:443 (the NVCF cloud endpoint).

Supports offline (batch) recognition of WAV audio files using models
from the Parakeet and Nemotron ASR families.

Requires: pip install nvidia-riva-client
          NVIDIA_API_KEY environment variable or .env file

Usage:
  NVIDIA_API_KEY=nvapi-... python3 nvidia-asr-grpc-test.py --audio test.wav
  NVIDIA_API_KEY=nvapi-... python3 nvidia-asr-grpc-test.py --audio test.wav --model nemotron
"""

import os
import sys
import time
import struct
import socket
import argparse
import grpc
import riva.client
from riva.client.proto import riva_asr_pb2, riva_audio_pb2

# ── Model registry (name → NVCF function ID) ──────────────────────────
ASR_MODELS = {
    'nemotron': {
        'id': 'bb0837de-8c7b-481f-9ec8-ef5663e9c1fa',
        'name': 'ai-nemotron-asr-streaming',
        'description': 'Nemotron ASR (streaming, latest)',
    },
    'parakeet-1.1b': {
        'id': '1598d209-5e27-4d3c-8079-4751568b1081',
        'name': 'ai-parakeet-ctc-1_1b-asr',
        'description': 'Parakeet CTC 1.1B (English)',
    },
    'parakeet-0.6b': {
        'id': 'd8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965',
        'name': 'ai-parakeet-ctc-0_6b-asr',
        'description': 'Parakeet CTC 0.6B (English)',
    },
    'canary-1b': {
        'id': 'b0e8b4a5-217c-40b7-9b96-17d84e666317',
        'name': 'ai-canary-1b-asr',
        'description': 'Canary 1B ASR (English)',
    },
    'parakeet-multilingual': {
        'id': '71203149-d3b7-4460-8231-1be2543a1fca',
        'name': 'ai-parakeet-1_1b-rnnt-multilingual-asr',
        'description': 'Parakeet 1.1B RNNT Multilingual',
    },
}

GRPC_HOST = 'grpc.nvcf.nvidia.com'
GRPC_PORT = 443
DEFAULT_MODEL = 'canary-1b'


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


def read_wav_pcm(file_path: str) -> tuple:
    """Read a WAV file and return (raw_pcm_bytes, sample_rate, bits_per_sample)."""
    with open(file_path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF':
        # Assume raw PCM; guess 16-bit mono 16kHz
        return data, 16000, 16
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits = struct.unpack('<H', data[34:36])[0]
    # Find 'data' chunk
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if chunk_id == b'data':
            pcm_start = pos + 8
            return data[pcm_start:pcm_start + chunk_size], sample_rate, bits
        pos += 8 + chunk_size
    # fallback: skip 44-byte header
    return data[44:], sample_rate, bits


def resolve_ipv4() -> str:
    """Resolve gRPC host to IPv4 address (avoids IPv6 hang on macOS)."""
    addrs = socket.getaddrinfo(
        GRPC_HOST, GRPC_PORT,
        socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
    )
    return addrs[0][4][0]


def create_auth(api_key: str, function_id: str) -> riva.client.Auth:
    """Create a riva.client.Auth object with IPv4-only gRPC channel."""
    ipv4 = resolve_ipv4()
    metadata = [
        ['function-id', function_id],
        ['authorization', f'Bearer {api_key}'],
    ]
    channel_options = [('grpc.ssl_target_name_override', GRPC_HOST)]
    return riva.client.Auth(
        uri=f'{ipv4}:{GRPC_PORT}',
        use_ssl=True,
        metadata_args=metadata,
        options=channel_options,
    )


def resample_pcm(pcm_bytes: bytes, src_rate: int, dst_rate: int = 16000) -> bytes:
    """Simple linear resampling of 16-bit mono PCM to a target sample rate."""
    if src_rate == dst_rate:
        return pcm_bytes
    import array
    src = array.array('h')
    src.frombytes(pcm_bytes)
    ratio = src_rate / dst_rate
    dst = array.array('h')
    for i in range(int(len(src) / ratio)):
        src_idx = min(int(i * ratio), len(src) - 1)
        dst.append(src[src_idx])
    return dst.tobytes()


def transcribe(audio_path: str, api_key: str, model_key: str = DEFAULT_MODEL,
               language: str = 'en-US', punctuate: bool = True) -> dict:
    """Transcribe a WAV audio file and return result dict."""
    model_info = ASR_MODELS[model_key]
    auth = create_auth(api_key, model_info['id'])
    service = riva.client.ASRService(auth)

    pcm_bytes, sample_rate, bits_per_sample = read_wav_pcm(audio_path)

    # ASR models work best at 16 kHz — resample if needed
    audio_dur_s = len(pcm_bytes) / (sample_rate * (bits_per_sample / 8))
    target_rate = 16000
    if sample_rate != target_rate:
        pcm_bytes = resample_pcm(pcm_bytes, sample_rate, target_rate)
        sample_rate = target_rate

    config = riva_asr_pb2.RecognitionConfig(
        encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        language_code=language,
        enable_automatic_punctuation=punctuate,
        max_alternatives=1,
    )

    response = service.offline_recognize(pcm_bytes, config)

    transcript = ''
    if response.results:
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript
                if punctuate is False:
                    transcript += '. '
            transcript = transcript.strip()

    return {
        'transcript': transcript,
        'audio_duration_s': audio_dur_s,
        'sample_rate': sample_rate,
    }


def run_test(options):
    print('NVIDIA ASR NIM — gRPC Test')
    print('=' * 40)
    print(f'Server:   {GRPC_HOST}:{GRPC_PORT}')
    print(f'Model:    {ASR_MODELS[options["model"]]["description"]}')
    print(f'Audio:    {options["audio"]}')
    print(f'Language: {options.get("language", "en-US")}')
    print()

    started = time.time()
    try:
        result = transcribe(
            options['audio'],
            options['api_key'],
            model_key=options['model'],
            language=options.get('language', 'en-US'),
            punctuate=not options.get('no_punctuation', False),
        )
        total_ms = int((time.time() - started) * 1000)

        print(f'  ✓ Transcription complete ({total_ms}ms)')
        print(f'  Audio duration: {result["audio_duration_s"]:.1f}s')
        print(f'  Sample rate:    {result["sample_rate"]} Hz')
        print(f'  Transcript:')
        print(f'    "{result["transcript"]}"')

    except grpc.RpcError as exc:
        total_ms = int((time.time() - started) * 1000)
        print(f'  ✗ Failed ({total_ms}ms): {exc.code()}: {exc.details()}')
    except Exception as exc:
        total_ms = int((time.time() - started) * 1000)
        print(f'  ✗ Failed ({total_ms}ms): {exc}')


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description='Test NVIDIA ASR NIM models via gRPC',
    )
    parser.add_argument('--api-key', default=os.environ.get('NVIDIA_API_KEY'))
    parser.add_argument('--audio', '-a',
                        help='Path to WAV audio file to transcribe')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                        choices=list(ASR_MODELS.keys()),
                        help=f'ASR model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--language', '-l', default='en-US',
                        help='Language code (default: en-US)')
    parser.add_argument('--no-punctuation', action='store_true',
                        help='Disable automatic punctuation')
    parser.add_argument('--list-models', action='store_true',
                        help='List available ASR models and exit')
    args = parser.parse_args()

    if args.list_models:
        print('Available ASR models:\n')
        for key, info in ASR_MODELS.items():
            print(f'  {key:25s}  {info["description"]}')
            print(f'  {"":25s}  NVCF: {info["id"]}')
            print()
        return

    if not args.api_key:
        print('ERROR: Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.',
              file=sys.stderr)
        sys.exit(1)

    if not args.audio:
        print('ERROR: --audio/-a is required for transcription. Use --list-models to see available models.',
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.audio):
        print(f'ERROR: Audio file not found: {args.audio}', file=sys.stderr)
        sys.exit(1)

    run_test(vars(args))


if __name__ == '__main__':
    main()
