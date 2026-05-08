#!/usr/bin/env node

/**
 * NVIDIA Magpie TTS Zero-shot Test Script
 *
 * Tests the Riva TTS Zero-shot model via the NVIDIA NIM HTTP API.
 * Supports both built-in voices and voice cloning with an audio prompt.
 *
 * Usage:
 *   node nvidia-tts-zeroshot-test.js [options]
 *
 * Requires: NVIDIA_API_KEY environment variable or .env file
 */

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const CLOUD_BASE_URL = 'https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/55cf67bf-600f-4b04-8eac-12ed39537a08';
const LOCAL_BASE_URL = 'http://localhost:9000/v1/audio/synthesize';
const NVCF_FUNCTION_ID = '55cf67bf-600f-4b04-8eac-12ed39537a08';
const DEFAULT_TEXT = 'Experience the future with Riva, where every word comes to life with clarity and emotion.';
const DEFAULT_VOICE = 'Magpie-ZeroShot.Female-1';
const AVAILABLE_VOICES = [
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
];

function stripOptionalQuotes(value) {
  const trimmed = value.trim();
  const first = trimmed[0];
  const last = trimmed[trimmed.length - 1];
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function loadEnvFile(envPath = path.resolve('.env'), env = process.env) {
  if (!fs.existsSync(envPath)) {
    return false;
  }
  const content = fs.readFileSync(envPath, 'utf8');
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const normalizedLine = line.startsWith('export ') ? line.slice('export '.length).trim() : line;
    const separatorIndex = normalizedLine.indexOf('=');
    if (separatorIndex === -1) continue;
    const key = normalizedLine.slice(0, separatorIndex).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || Object.prototype.hasOwnProperty.call(env, key)) continue;
    const value = normalizedLine.slice(separatorIndex + 1);
    env[key] = stripOptionalQuotes(value);
  }
  return true;
}

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    apiKey: process.env.NVIDIA_API_KEY,
    baseURL: process.env.NVIDIA_TTS_URL || CLOUD_BASE_URL,
    text: DEFAULT_TEXT,
    voice: DEFAULT_VOICE,
    audioPrompt: null,
    quality: 20,
    language: 'en-US',
    output: null,
    runs: 3,
    timeoutMs: 60_000,
    delayMs: 2000,
    listVoices: false,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`Missing value for ${arg}.`);
      return argv[index];
    };

    if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--api-key') {
      options.apiKey = next();
    } else if (arg === '--base-url') {
      options.baseURL = next();
    } else if (arg === '--text') {
      options.text = next();
    } else if (arg === '--voice') {
      options.voice = next();
    } else if (arg === '--audio-prompt') {
      options.audioPrompt = next();
    } else if (arg === '--quality') {
      options.quality = Number.parseInt(next(), 10);
      if (options.quality < 1 || options.quality > 40) throw new Error('--quality must be between 1 and 40.');
    } else if (arg === '--language') {
      options.language = next();
    } else if (arg === '--output' || arg === '-o') {
      options.output = next();
    } else if (arg === '--runs') {
      options.runs = Number.parseInt(next(), 10);
      if (options.runs < 1) throw new Error('--runs must be a positive integer.');
    } else if (arg === '--timeout-ms') {
      options.timeoutMs = Number.parseInt(next(), 10);
    } else if (arg === '--delay-ms') {
      options.delayMs = Number.parseInt(next(), 10);
    } else if (arg === '--list-voices') {
      options.listVoices = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  console.log(`Usage: node nvidia-tts-zeroshot-test.js [options]

Tests the NVIDIA Magpie TTS Zero-shot model (Riva TTS) via HTTP API.

Options:
  --api-key <key>         NVIDIA API key. Defaults to NVIDIA_API_KEY env var.
  --base-url <url>        API base URL. Default: ${CLOUD_BASE_URL}
  --text <text>           Text to synthesize. Default: "${DEFAULT_TEXT}"
  --voice <name>          Built-in voice name. Default: ${DEFAULT_VOICE}
  --audio-prompt <path>   Path to WAV file for voice cloning (3-10s, 16-bit mono, 22.05kHz+).
  --quality <1-40>        Voice cloning quality (higher = slower but better match). Default: 20
  --language <code>       Language code. Default: en-US
  --output, -o <path>     Save last synthesized audio to this path.
  --runs <n>              Number of synthesis attempts. Default: 3
  --timeout-ms <n>        Timeout per request. Default: 60000
  --delay-ms <n>          Delay between requests. Default: 2000
  --list-voices           List available built-in voices and exit.
  --help, -h              Show this help.

Examples:
  # Test with built-in voice
  NVIDIA_API_KEY=nvapi-... node nvidia-tts-zeroshot-test.js --voice Magpie-ZeroShot.Male-1

  # Test voice cloning
  NVIDIA_API_KEY=nvapi-... node nvidia-tts-zeroshot-test.js --audio-prompt prompt.wav --output cloned.wav

  # Benchmark multiple runs
  NVIDIA_API_KEY=nvapi-... node nvidia-tts-zeroshot-test.js --runs 5 --text "Hello world"
`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Read basic metadata from a WAV file header (44 bytes).
 * Returns { sampleRate, bitsPerSample, channels, durationSec } or null on failure.
 */
function readWavHeader(filePath) {
  try {
    const fd = fs.openSync(filePath, 'r');
    const header = Buffer.alloc(44);
    fs.readSync(fd, header, 0, 44, 0);
    fs.closeSync(fd);

    const riff = header.toString('utf8', 0, 4);
    const wave = header.toString('utf8', 8, 12);
    if (riff !== 'RIFF' || wave !== 'WAVE') return null;

    const channels = header.readUInt16LE(22);
    const sampleRate = header.readUInt32LE(24);
    const bitsPerSample = header.readUInt16LE(34);
    const fmtSize = header.readUInt32LE(16);
    // dataSize is at byte 40; use it to compute duration
    const dataSize = header.readUInt32LE(40);
    const durationSec = dataSize / (sampleRate * channels * (bitsPerSample / 8));

    return { sampleRate, bitsPerSample, channels, durationSec };
  } catch {
    return null;
  }
}

function round(value) {
  return Math.round(value);
}

async function synthesize(options, fetchFn = fetch) {
  const url = options.baseURL;
  const isCloud = url.includes('nvcf.nvidia.com') || url.includes('ngc.nvidia.com');

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs);

  try {
    let response;

    if (isCloud) {
      // NVIDIA Cloud API format: multipart with "config" JSON + "audio" WAV.
      // The config maps to the gRPC SynthesizeSpeechRequest proto (snake_case).
      const config = {
        language_code: options.language,
        encoding: 'LINEAR_PCM',
        sample_rate_hz: 22050,
        text: options.text,
      };

      if (!options.audioPrompt) {
        config.voice_name = options.voice;
      } else {
        const wavMeta = readWavHeader(options.audioPrompt);
        config.zero_shot_data = {
          quality: options.quality,
          sample_rate_hz: wavMeta ? wavMeta.sampleRate : 22050,
          encoding: 'LINEAR_PCM',
        };
      }

      const formData = new FormData();
      formData.append('config', JSON.stringify(config));

      if (options.audioPrompt) {
        const audioBuffer = fs.readFileSync(options.audioPrompt);
        const blob = new Blob([audioBuffer], { type: 'audio/wav' });
        formData.append('audio', blob, path.basename(options.audioPrompt));
      }

      response = await fetchFn(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${options.apiKey}`,
          Accept: 'audio/wav',
        },
        body: formData,
        signal: controller.signal,
        duplex: 'half',
      });
    } else {
      // Local NIM container format
      const formData = new FormData();
      formData.append('language', options.language);
      formData.append('text', options.text);

      if (options.audioPrompt) {
        const audioBuffer = fs.readFileSync(options.audioPrompt);
        const blob = new Blob([audioBuffer], { type: 'audio/wav' });
        formData.append('audio_prompt', blob, path.basename(options.audioPrompt));
        formData.append('quality', String(options.quality));
      } else {
        formData.append('voice', options.voice);
      }

      response = await fetchFn(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${options.apiKey}`,
          Accept: 'audio/wav',
        },
        body: formData,
        signal: controller.signal,
      });
    }

    clearTimeout(timer);

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      const message = [];
      message.push('HTTP ' + response.status + ': ' + (errorText || response.statusText));

      if (response.status === 500 && (errorText.includes('internal-server-error') || errorText.includes('inference'))) {
        message.push('');
        message.push('  NOTE: The Magpie TTS Zeroshot model is restricted access. You may need to:');
        message.push('    1. Apply for access at https://developer.nvidia.com/riva-tts-zeroshot-models');
        message.push('    2. Use the local NIM Docker container instead: http://localhost:9000');
        message.push('    3. Try the gRPC endpoint: grpc.nvcf.nvidia.com:443 (see python-clients/scripts/tts)');
      }

      throw new Error(message.join('\n'));
    }

    // Cloud API returns text/event-stream with base64 audio chunks
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('text/event-stream')) {
      const text = await response.text();
      const audioChunks = [];
      for (const line of text.split('\n')) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data && data !== '[DONE]') {
            try {
              const parsed = JSON.parse(data);
              if (parsed.audio) {
                audioChunks.push(Buffer.from(parsed.audio, 'base64'));
              }
            } catch { /* skip non-JSON lines */ }
          }
        }
      }
      if (audioChunks.length === 0) {
        throw new Error('No audio data in SSE response');
      }
      return Buffer.concat(audioChunks);
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  } catch (error) {
    clearTimeout(timer);
    throw error;
  }
}

async function runTest(options, fetchFn = fetch) {
  const results = [];

  console.log('NVIDIA Magpie TTS Zero-shot Test');
  console.log('================================');
  console.log(`Text: "${options.text}"`);
  console.log(`Mode: ${options.audioPrompt ? `Voice Cloning (quality: ${options.quality})` : `Built-in Voice (${options.voice})`}`);
  console.log(`Language: ${options.language}`);
  console.log(`Runs: ${options.runs}`);
  console.log('');

  for (let attempt = 1; attempt <= options.runs; attempt += 1) {
    const startedAt = performance.now();

    try {
      const audioBuffer = await synthesize(options, fetchFn);
      const totalMs = round(performance.now() - startedAt);
      const durationEstimate = audioBuffer.length > 44 ? ((audioBuffer.length - 44) / (22050 * 2)).toFixed(1) : '?';

      const result = {
        attempt,
        ok: true,
        totalMs,
        audioBytes: audioBuffer.length,
        estimatedDurationSec: durationEstimate,
      };
      results.push(result);

      console.log(`  attempt ${attempt}: ok, ${totalMs}ms, ${audioBuffer.length} bytes (~${durationEstimate}s audio)`);

      if (attempt === options.runs && options.output) {
        fs.writeFileSync(options.output, audioBuffer);
        console.log(`  -> Saved to ${options.output}`);
      }
    } catch (error) {
      const totalMs = round(performance.now() - startedAt);
      const result = {
        attempt,
        ok: false,
        totalMs,
        error: { name: error.name, message: error.message },
      };
      results.push(result);

      console.log(`  attempt ${attempt}: failed after ${totalMs}ms, ${error.name}: ${error.message}`);
    }

    if (attempt < options.runs && options.delayMs > 0) {
      await sleep(options.delayMs);
    }
  }

  const successes = results.filter((r) => r.ok);
  const avgMs = successes.length > 0
    ? round(successes.reduce((sum, r) => sum + r.totalMs, 0) / successes.length)
    : null;

  console.log('');
  console.log('Summary:');
  console.log(`  Successes: ${successes.length}/${results.length}`);
  if (avgMs !== null) {
    console.log(`  Avg latency: ${avgMs}ms`);
    console.log(`  Min latency: ${Math.min(...successes.map((r) => r.totalMs))}ms`);
    console.log(`  Max latency: ${Math.max(...successes.map((r) => r.totalMs))}ms`);
  }

  return { results, summary: { successes: successes.length, total: results.length, avgMs } };
}

async function main() {
  loadEnvFile();
  const options = parseArgs();

  if (options.help) {
    printHelp();
    return;
  }

  if (options.listVoices) {
    console.log('Available built-in voices for Magpie TTS Zeroshot:\n');
    for (const voice of AVAILABLE_VOICES) {
      console.log(`  ${voice}`);
    }
    return;
  }

  if (!options.apiKey) {
    throw new Error('Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.');
  }

  if (options.audioPrompt && !fs.existsSync(options.audioPrompt)) {
    throw new Error(`Audio prompt file not found: ${options.audioPrompt}`);
  }

  await runTest(options);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = {
  AVAILABLE_VOICES,
  loadEnvFile,
  parseArgs,
  readWavHeader,
  runTest,
  synthesize,
};
