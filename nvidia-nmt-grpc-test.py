#!/usr/bin/env python3
"""
NVIDIA NMT NIM — HTTP Test Script

Tests the Riva Translate 4B Instruct model via the NVIDIA integrate API
(OpenAI-compatible chat completions endpoint).

The model supports 12 languages: en, de, es-ES, es-US, fr, pt-BR, ru,
zh-CN, zh-TW, ja, ko, ar.

This is a FREE cloud endpoint — no gRPC or local NIM container needed.

Requires: NVIDIA_API_KEY environment variable or .env file

Usage:
  NVIDIA_API_KEY=nvapi-... python3 nvidia-nmt-grpc-test.py --text "Hello world" --from en --to zh-CN
  NVIDIA_API_KEY=nvapi-... python3 nvidia-nmt-grpc-test.py --text "Hello" --from en --to fr
"""

import os
import sys
import time
import argparse
import urllib.request
import json

INTEGRATE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
MODEL_ID = 'nvidia/riva-translate-4b-instruct-v1.1'

# Supported languages and their display names
SUPPORTED_LANGUAGES = {
    'en':    'English',
    'de':    'German',
    'es-ES': 'European Spanish',
    'es-US': 'LATAM Spanish',
    'fr':    'French',
    'pt-BR': 'Brazilian Portuguese',
    'ru':    'Russian',
    'zh-CN': 'Simplified Chinese',
    'zh-TW': 'Traditional Chinese',
    'ja':    'Japanese',
    'ko':    'Korean',
    'ar':    'Arabic',
}


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


def translate(text: str, source_lang: str, target_lang: str,
              api_key: str) -> str:
    """Translate text using the Riva Translate 4B model via HTTP."""
    src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
    tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

    body = {
        'model': MODEL_ID,
        'messages': [
            {
                'role': 'system',
                'content': (
                    f'You are an expert at translating text '
                    f'from {src_name} to {tgt_name}.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'What is the {tgt_name} translation of '
                    f'the sentence: {text}?'
                ),
            },
        ],
        'max_tokens': 512,
        'temperature': 0,
    }

    req = urllib.request.Request(
        INTEGRATE_URL,
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


def run_test(options):
    print('NVIDIA NMT NIM — HTTP Test')
    print('=' * 40)
    print(f'Endpoint:  {INTEGRATE_URL}')
    print(f'Model:     {MODEL_ID}')
    print(f'Source:    {options["source_lang"]} ({SUPPORTED_LANGUAGES.get(options["source_lang"], "?")})')
    print(f'Target:    {options["target_lang"]} ({SUPPORTED_LANGUAGES.get(options["target_lang"], "?")})')
    print(f'Input:     "{options["text"]}"')
    print()

    started = time.time()
    try:
        result = translate(
            options['text'],
            options['source_lang'],
            options['target_lang'],
            options['api_key'],
        )
        total_ms = int((time.time() - started) * 1000)

        print(f'  OK ({total_ms}ms)')
        print(f'  Translation:')
        print(f'    "{result}"')

    except urllib.error.HTTPError as exc:
        total_ms = int((time.time() - started) * 1000)
        body = exc.read().decode('utf-8', errors='replace')
        print(f'  FAILED ({total_ms}ms): HTTP {exc.code}')
        print(f'    {body[:300]}')
    except Exception as exc:
        total_ms = int((time.time() - started) * 1000)
        print(f'  FAILED ({total_ms}ms): {exc}')


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description='Test NVIDIA NMT model via HTTP (chat completions API)',
    )
    parser.add_argument('--api-key', default=os.environ.get('NVIDIA_API_KEY'))
    parser.add_argument('--text', '-t',
                        help='Text to translate')
    parser.add_argument('--from', dest='source_lang', default='en',
                        help='Source language code (default: en)')
    parser.add_argument('--to', dest='target_lang', default='zh-CN',
                        help='Target language code (default: zh-CN)')
    parser.add_argument('--list-languages', action='store_true',
                        help='List supported languages and exit')
    args = parser.parse_args()

    if args.list_languages:
        print('Supported languages for Riva Translate 4B:\n')
        for code, name in SUPPORTED_LANGUAGES.items():
            print(f'  {code:8s}  {name}')
        return

    if not args.text:
        print('ERROR: --text/-t is required for translation. Use --list-languages to see supported languages.',
              file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        print('ERROR: Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.',
              file=sys.stderr)
        sys.exit(1)

    if args.source_lang not in SUPPORTED_LANGUAGES:
        print(f'WARNING: "{args.source_lang}" is not in the known language list.', file=sys.stderr)
    if args.target_lang not in SUPPORTED_LANGUAGES:
        print(f'WARNING: "{args.target_lang}" is not in the known language list.', file=sys.stderr)

    run_test(vars(args))


if __name__ == '__main__':
    main()
