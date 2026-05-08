#!/usr/bin/env node

/**
 * NVIDIA Nemotron 3 VoiceChat Test Script
 *
 * Tests the Nemotron 3 VoiceChat model — a 12B full-duplex, end-to-end
 * speech-to-speech model for real-time conversational AI.
 *
 * The model uses WebSocket for real-time audio streaming:
 *   - Input: PCM16 audio at 24 kHz
 *   - Output: PCM16 audio at 24 kHz + text transcription
 *   - Protocol: JSON messages over WebSocket (similar to OpenAI Realtime API)
 *
 * Usage:
 *   node nvidia-voicechat-test.js [options]
 *
 * Requires: NVIDIA_API_KEY or NGC_API_KEY environment variable
 */

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const WebSocket = require('ws');

const NVCF_FUNCTION_ID = '42c86b5f-545a-4b2f-a83b-90fd71da9912';
const CLOUD_WS_URL = `wss://api.ngc.nvidia.com/v2/predict/artifactname/websocket/v1/realtime?nv-function-id=${NVCF_FUNCTION_ID}`;
const CLOUD_QUEUE_URL = `https://api.ngc.nvidia.com/v2/predict/queues/models/qc69jvmznzxy/nemotron-voicechat`;
const LOCAL_WS_URL = 'ws://localhost:9000/v1/realtime';
const MODEL_ID = 'nvidia/nemotron-voicechat';

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
    apiKey: process.env.NVIDIA_API_KEY || process.env.NGC_API_KEY,
    wsURL: process.env.NVIDIA_VOICECHAT_WS_URL || CLOUD_WS_URL,
    audioFile: null,
    persona: null,
    timeoutMs: 30_000,
    mode: 'health',
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
      options.wsURL = next();
    } else if (arg === '--audio-file') {
      options.audioFile = next();
    } else if (arg === '--persona') {
      options.persona = next();
    } else if (arg === '--timeout-ms') {
      options.timeoutMs = Number.parseInt(next(), 10);
    } else if (arg === '--mode') {
      const mode = next();
      if (!['health', 'info', 'websocket'].includes(mode)) {
        throw new Error(`--mode must be one of: health, info, websocket`);
      }
      options.mode = mode;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  console.log(`Usage: node nvidia-voicechat-test.js [options]

Tests the NVIDIA Nemotron 3 VoiceChat model connectivity and capabilities.

Nemotron 3 VoiceChat is a 12B full-duplex speech-to-speech model:
  - Architecture: Hybrid Mamba/Transformer
  - Input: PCM16 audio at 24 kHz
  - Output: PCM16 audio at 24 kHz + text transcription
  - Full-duplex: listens while speaking, natural barge-in
  - Persona control via text role prompts (PersonaPlex)
  - Protocol: WebSocket with JSON messages

Modes:
  health      Check model queue status and endpoint reachability (default)
  info        Retrieve model spec from NGC API
  websocket   Test WebSocket connection to the realtime endpoint

Options:
  --api-key <key>         API key. Defaults to NVIDIA_API_KEY or NGC_API_KEY.
  --base-url <url>        WebSocket URL. Default: cloud endpoint
                          For local NIM: ws://localhost:9000/v1/realtime
  --audio-file <path>     Path to 24kHz mono PCM16 WAV file for websocket mode.
  --persona <text>        Persona prompt for the voice agent.
  --timeout-ms <n>        Connection timeout. Default: 30000
  --mode <mode>           Test mode: health, info, or websocket. Default: health
  --help, -h              Show this help.

Model Details:
  Model ID:       ${MODEL_ID}
  Function ID:    ${NVCF_FUNCTION_ID}
  Parameters:     ~12B
  Runtime:        vLLM
  Hardware:       H100, A100, B200, RTX 6000-class
  Release:        2026-03-16 (Early Access)
  Limitations:    English only, single voice, no tool calling

WebSocket Protocol:
  1. Connect to: ${CLOUD_WS_URL}
  2. Send: {"functionId": "<nvcf-function-id>"}
  3. Send: {"type": "session.update", "session": {"audio": {"input": {"format": "pcm16"}, "output": {"format": "pcm16"}}}}
  4. Stream PCM16 audio chunks (24kHz, mono)
  5. Receive: agent audio + text transcription

Examples:
  # Check cloud model status
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-test.js --mode health

  # Test WebSocket connection
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-test.js --mode websocket

  # Test with local NIM deployment
  node nvidia-voicechat-test.js --base-url ws://localhost:9000/v1/realtime --mode websocket

Docker deployment:
  docker run -it --rm --name nemotron-voicechat \\
    --runtime=nvidia --gpus 'device=0' --shm-size=8GB \\
    -e NGC_API_KEY -e NIM_HTTP_API_PORT=9000 \\
    -p 9000:9000 -p 50051:50051 \\
    nvcr.io/nim/nvidia/nemotron-voicechat:latest
`);
}

function round(value) {
  return Math.round(value);
}

async function checkHealth(options) {
  console.log('Nemotron 3 VoiceChat - Health Check');
  console.log('===================================');
  console.log(`Function ID: ${NVCF_FUNCTION_ID}`);
  console.log('');

  // Check model queue status via NGC API
  console.log('  Checking model queue status...');
  const startedAt = performance.now();

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs);

    const response = await fetch(CLOUD_QUEUE_URL, {
      method: 'GET',
      headers: options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {},
      signal: controller.signal,
    });

    clearTimeout(timer);
    const totalMs = round(performance.now() - startedAt);

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      console.log(`  ✗ Queue status: HTTP ${response.status} (${totalMs}ms)`);
      console.log(`    ${errorText.slice(0, 200)}`);
      return;
    }

    const data = await response.json();
    console.log(`  ✓ Queue status fetched (${totalMs}ms)`);

    if (data.queues && data.queues.length > 0) {
      for (const queue of data.queues) {
        const statusIcon = queue.functionStatus === 'ACTIVE' ? '✓' : '✗';
        console.log(`  ${statusIcon} ${queue.functionName}: ${queue.functionStatus}`);
        console.log(`    Queue depth: ${queue.queueDepth}`);
        console.log(`    Version: ${queue.functionVersionId}`);
      }
    } else {
      console.log('  ✗ No queues found');
    }
  } catch (error) {
    const totalMs = round(performance.now() - startedAt);
    console.log(`  ✗ ${error.name}: ${error.message} (${totalMs}ms)`);
  }

  console.log('');
  console.log(`  WebSocket endpoint: ${options.wsURL}`);
}

async function getModelInfo(options) {
  console.log('Nemotron 3 VoiceChat - Model Info');
  console.log('=================================');
  console.log('');

  const url = `https://api.ngc.nvidia.com/v2/endpoints/qc69jvmznzxy/nemotron-voicechat/spec?`;
  const startedAt = performance.now();

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs);

    const response = await fetch(url, {
      method: 'GET',
      headers: options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {},
      signal: controller.signal,
    });

    clearTimeout(timer);
    const totalMs = round(performance.now() - startedAt);

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
    }

    const data = await response.json();
    console.log(`  Fetched in ${totalMs}ms`);
    console.log(`  Artifact: ${data.artifactName}`);
    console.log(`  Namespace: ${data.namespace}`);
    console.log(`  Function ID: ${data.nvcfFunctionId}`);
    console.log(`  Created: ${data.createdDate}`);
    console.log(`  Updated: ${data.updatedDate}`);
  } catch (error) {
    console.log(`  Error: ${error.message}`);
  }
}

async function testWebSocket(options) {
  console.log('Nemotron 3 VoiceChat - WebSocket Test');
  console.log('=====================================');
  console.log(`URL: ${options.wsURL}`);
  if (options.persona) {
    console.log(`Persona: ${options.persona}`);
  }
  console.log('');

  return new Promise((resolve) => {
    const startedAt = performance.now();
    const messages = [];
    let connected = false;

    const headers = {};
    if (options.apiKey) {
      headers.Authorization = `Bearer ${options.apiKey}`;
    }

    console.log('  Connecting...');
    const ws = new WebSocket(options.wsURL, { headers });

    const timer = setTimeout(() => {
      const totalMs = round(performance.now() - startedAt);
      if (!connected) {
        console.log(`  ✗ Connection timed out after ${totalMs}ms`);
      } else {
        console.log(`  Connection closed after ${totalMs}ms (timeout)`);
      }
      ws.close();
      resolve({ connected, messages });
    }, options.timeoutMs);

    ws.on('open', () => {
      connected = true;
      const totalMs = round(performance.now() - startedAt);
      console.log(`  ✓ Connected in ${totalMs}ms`);

      // Send function ID (first message per protocol)
      const initMsg = { functionId: NVCF_FUNCTION_ID };
      ws.send(JSON.stringify(initMsg));
      console.log(`  → Sent: ${JSON.stringify(initMsg)}`);

      // Send session.update
      const sessionMsg = {
        type: 'session.update',
        event_id: `evt_${Date.now()}`,
        session: {
          audio: {
            input: { format: 'pcm16' },
            output: { format: 'pcm16' },
          },
        },
      };
      if (options.persona) {
        sessionMsg.session.instructions = options.persona;
      }
      ws.send(JSON.stringify(sessionMsg));
      console.log(`  → Sent: session.update`);

      // If we have an audio file, send it
      if (options.audioFile && fs.existsSync(options.audioFile)) {
        const audioBuffer = fs.readFileSync(options.audioFile);
        // Skip WAV header (44 bytes) and send raw PCM
        const pcmData = audioBuffer.slice(44);
        const base64Audio = pcmData.toString('base64');
        const audioMsg = {
          type: 'input_audio_buffer.append',
          event_id: `evt_${Date.now()}_audio`,
          audio: base64Audio,
        };
        ws.send(JSON.stringify(audioMsg));
        console.log(`  → Sent: input_audio_buffer.append (${pcmData.length} bytes PCM)`);

        // Signal end of audio
        const commitMsg = {
          type: 'input_audio_buffer.commit',
          event_id: `evt_${Date.now()}_commit`,
        };
        ws.send(JSON.stringify(commitMsg));
        console.log(`  → Sent: input_audio_buffer.commit`);
      }
    });

    ws.on('message', (data) => {
      const totalMs = round(performance.now() - startedAt);
      try {
        const msg = JSON.parse(data.toString());
        messages.push(msg);
        // Don't print full audio data
        if (msg.type && msg.type.includes('audio')) {
          console.log(`  ← [${totalMs}ms] ${msg.type} (audio data)`);
        } else {
          const preview = JSON.stringify(msg).slice(0, 200);
          console.log(`  ← [${totalMs}ms] ${preview}`);
        }
      } catch {
        console.log(`  ← [${totalMs}ms] (binary: ${data.length} bytes)`);
      }
    });

    ws.on('error', (error) => {
      const totalMs = round(performance.now() - startedAt);
      console.log(`  ✗ Error after ${totalMs}ms: ${error.message}`);
    });

    ws.on('close', (code, reason) => {
      clearTimeout(timer);
      const totalMs = round(performance.now() - startedAt);
      console.log(`  Connection closed: code=${code}, reason="${reason || ''}" (${totalMs}ms)`);
      console.log('');
      console.log(`  Summary: connected=${connected}, messages_received=${messages.length}`);
      resolve({ connected, messages });
    });
  });
}

async function main() {
  loadEnvFile();
  const options = parseArgs();

  if (options.help) {
    printHelp();
    return;
  }

  if (options.mode === 'health') {
    await checkHealth(options);
  } else if (options.mode === 'info') {
    await getModelInfo(options);
  } else if (options.mode === 'websocket') {
    if (!options.apiKey) {
      throw new Error('Missing API key. Set NVIDIA_API_KEY/NGC_API_KEY or pass --api-key.');
    }
    await testWebSocket(options);
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = {
  CLOUD_WS_URL,
  MODEL_ID,
  NVCF_FUNCTION_ID,
  checkHealth,
  getModelInfo,
  loadEnvFile,
  parseArgs,
  testWebSocket,
};
