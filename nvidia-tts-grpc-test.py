#!/usr/bin/env python3
"""
NVIDIA Magpie TTS Zero-shot Test — gRPC backend

Uses the nvidia-riva-client Python library to call the TTS model
via gRPC to grpc.nvcf.nvidia.com:443 (the NVCF cloud endpoint).

Requires: pip install nvidia-riva-client
          NVIDIA_API_KEY environment variable or .env file
"""

import os
import sys
import time
import struct
import socket
import argparse
from pathlib import Path
import grpc
import riva.client
from riva.client.proto import riva_audio_pb2

NVCF_FUNCTION_ID = '55cf67bf-600f-4b04-8eac-12ed39537a08'
GRPC_HOST = 'grpc.nvcf.nvidia.com'
GRPC_PORT = 443
GRPC_SERVER = f'{GRPC_HOST}:{GRPC_PORT}'
DEFAULT_TEXT = (
    'Experience the future with Riva, where every word comes to life '
    'with clarity and emotion.'
)
DEFAULT_VOICE = 'Magpie-ZeroShot.Female-1'
DEFAULT_LANGUAGE = 'en-US'
DEFAULT_QUALITY = 20

AVAILABLE_VOICES = [
    'Magpie-ZeroShot.Female-1',
    'Magpie-ZeroShot.Female-Neutral',
    'Magpie-ZeroShot.Female-Angry',
    'Magpie-ZeroShot.Female-Fearful',
    'Magpie-ZeroShot.Female-Calm',
    'Magpie-ZeroShot.Female-Happy',
    'Magpie-ZeroShot.Male-1',
    'Magpie-ZeroShot.Male-Calm',
    'Magpie-ZeroShot.Male-Neutral',
    'Magpie-ZeroShot.Male-Angry',
    'Magpie-ZeroShot.Male-Fearful',
]


def load_env_file(env_path='.env'):
    """Load KEY=VALUE pairs from a .env file (skip already-set vars)."""
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


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 22050,
                bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """Prepend a minimal WAV (RIFF) header to raw PCM16 audio data."""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,               # chunk size (PCM)
        1,                # audio format (PCM = 1)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header + pcm_bytes


def synthesize(options):
    """Make a single TTS gRPC call and return the WAV audio bytes."""

    # --- Force IPv4: resolve hostname before creating the gRPC channel ---
    # On macOS, gRPC may hang when trying IPv6 first. We explicitly resolve
    # the hostname to an IPv4 address and set ssl_target_name_override so
    # TLS certificate validation still uses the original hostname.
    addrs = socket.getaddrinfo(
        GRPC_HOST, GRPC_PORT,
        socket.AF_INET,       # IPv4 only
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
    )
    ipv4_address = addrs[0][4][0]
    ipv4_target = f'{ipv4_address}:{GRPC_PORT}'

    # Channel options: force IPv4-only connection with proper TLS hostname
    channel_options = [
        ('grpc.ssl_target_name_override', GRPC_HOST),
    ]

    # Build gRPC metadata — each pair is [key, value] per riva.client.Auth spec
    metadata = [
        ['function-id', NVCF_FUNCTION_ID],
        ['authorization', f'Bearer {options["api_key"]}'],
    ]

    # Auth object: connect to the IPv4 address, validate TLS against hostname
    auth = riva.client.Auth(
        uri=ipv4_target,
        use_ssl=True,
        metadata_args=metadata,
        options=channel_options,
    )

    # Create the TTS service stub
    service = riva.client.SpeechSynthesisService(auth)

    # Use the high-level API — handles protobuf construction internally
    encoding = riva_audio_pb2.AudioEncoding.LINEAR_PCM
    quality = options.get('quality', DEFAULT_QUALITY)
    audio_prompt = options.get('audio_prompt')

    if audio_prompt:
        # Voice cloning mode: use the lower-level stub to control zero_shot_data
        # fields fully (the high-level API doesn't set sample_rate_hz on
        # zero_shot_data, which the server requires).
        from riva.client.proto import riva_tts_pb2

        # Read audio prompt and detect its format
        with open(audio_prompt, 'rb') as f:
            audio_bytes = f.read()
        audio_prompt_sr = 22050
        if audio_bytes[:4] == b'RIFF':
            audio_prompt_sr = struct.unpack('<I', audio_bytes[24:28])[0]

        req = riva_tts_pb2.SynthesizeSpeechRequest(
            text=options['text'],
            language_code=options.get('language', DEFAULT_LANGUAGE),
            encoding=encoding,
            sample_rate_hz=22050,
        )
        req.zero_shot_data.audio_prompt = audio_bytes
        req.zero_shot_data.sample_rate_hz = audio_prompt_sr
        req.zero_shot_data.encoding = encoding
        req.zero_shot_data.quality = quality

        response = service.stub.Synthesize(
            req,
            metadata=auth.get_auth_metadata(),
        )
    else:
        # Built-in voice: use the high-level API
        response = service.synthesize(
            text=options['text'],
            voice_name=options.get('voice') or DEFAULT_VOICE,
            language_code=options.get('language', DEFAULT_LANGUAGE),
            encoding=encoding,
            sample_rate_hz=22050,
        )

    if response.audio:
        # gRPC returns raw PCM — wrap in a WAV header for playback
        return pcm_to_wav(response.audio, sample_rate=22050)
    raise RuntimeError('Empty audio response')


def run_test(options):
    """Run the test with the given options."""
    runs = options.get('runs', 3)
    delay = options.get('delay_ms', 2000) / 1000.0

    print('NVIDIA Magpie TTS Zero-shot Test (gRPC)')
    print('=' * 40)
    print(f'Server:   {GRPC_SERVER}')
    print(f'Text:     "{options["text"]}"')
    mode = 'Voice Cloning' if options.get('audio_prompt') else f'Built-in Voice ({options.get("voice", DEFAULT_VOICE)})'
    print(f'Mode:     {mode}')
    print(f'Language: {options.get("language", DEFAULT_LANGUAGE)}')
    print(f'Runs:     {runs}')
    print()

    results = []

    for attempt in range(1, runs + 1):
        started = time.time()
        try:
            audio_bytes = synthesize(options)
            total_ms = int((time.time() - started) * 1000)
            # Rough duration estimate: 22050 Hz, 16-bit mono -> 44100 bytes/sec
            duration_s = len(audio_bytes) / 44100.0

            results.append({
                'attempt': attempt,
                'ok': True,
                'total_ms': total_ms,
                'audio_bytes': len(audio_bytes),
                'duration_s': round(duration_s, 1),
            })

            print(
                f'  attempt {attempt}: ok, {total_ms}ms, '
                f'{len(audio_bytes)} bytes (~{duration_s:.1f}s audio)'
            )

            if attempt == runs and options.get('output'):
                with open(options['output'], 'wb') as fh:
                    fh.write(audio_bytes)
                print(f'  -> Saved to {options["output"]}')

        except grpc.RpcError as exc:
            total_ms = int((time.time() - started) * 1000)
            code = exc.code()
            detail = exc.details() or ''
            results.append({
                'attempt': attempt,
                'ok': False,
                'total_ms': total_ms,
                'error': f'{code}: {detail}',
            })
            print(f'  attempt {attempt}: failed after {total_ms}ms, {code}: {detail}')

        except Exception as exc:
            total_ms = int((time.time() - started) * 1000)
            results.append({
                'attempt': attempt,
                'ok': False,
                'total_ms': total_ms,
                'error': str(exc),
            })
            print(f'  attempt {attempt}: failed after {total_ms}ms, {exc}')

        if attempt < runs and delay > 0:
            time.sleep(delay)

    # Summary
    successes = [r for r in results if r['ok']]
    avg_ms = (
        int(sum(r['total_ms'] for r in successes) / len(successes))
        if successes else None
    )

    print()
    print('Summary:')
    print(f'  Successes: {len(successes)}/{len(results)}')
    if avg_ms is not None:
        print(f'  Avg latency: {avg_ms}ms')
        print(f'  Min latency: {min(r["total_ms"] for r in successes)}ms')
        print(f'  Max latency: {max(r["total_ms"] for r in successes)}ms')

    return results


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description='Test NVIDIA Magpie TTS Zero-shot model via gRPC',
    )
    parser.add_argument('--api-key', default=os.environ.get('NVIDIA_API_KEY'))
    parser.add_argument('--text', default=DEFAULT_TEXT)
    parser.add_argument('--voice', default=DEFAULT_VOICE)
    parser.add_argument('--audio-prompt', default=None)
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY)
    parser.add_argument('--language', default=DEFAULT_LANGUAGE)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--timeout-ms', type=int, default=60000)
    parser.add_argument('--delay-ms', type=int, default=2000)
    parser.add_argument('--list-voices', action='store_true')
    args = parser.parse_args()

    if args.list_voices:
        print('Available built-in voices for Magpie TTS Zeroshot:\n')
        for v in AVAILABLE_VOICES:
            print(f'  {v}')
        return

    if not args.api_key:
        print('ERROR: Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.',
              file=sys.stderr)
        sys.exit(1)

    if args.audio_prompt and not os.path.exists(args.audio_prompt):
        print(f'ERROR: Audio prompt file not found: {args.audio_prompt}',
              file=sys.stderr)
        sys.exit(1)

    run_test(vars(args))


if __name__ == '__main__':
    main()
