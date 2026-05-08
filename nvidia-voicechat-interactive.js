#!/usr/bin/env node

/**
 * NVIDIA Nemotron 3 VoiceChat — Interactive Voice CLI
 *
 * Real-time voice chat with the Nemotron 3 VoiceChat model.
 * Speak into your microphone and hear the model's audio response
 * through your speakers — full-duplex with barge-in support.
 *
 * Audio backend: FFmpeg (mic capture) + FFplay (speaker playback)
 * No native addons required — uses stdin/stdout pipes.
 *
 * Usage:
 *   NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js
 *   NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js --list-devices
 *   NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js --input-device 0
 */

const { spawn } = require('node:child_process');
const WebSocket = require('ws');

const {
  NVCF_FUNCTION_ID,
  CLOUD_WS_URL,
  loadEnvFile,
} = require('./nvidia-voicechat-test');

// ── Proxy support ────────────────────────────────────────────────────────────

let HttpsProxyAgent, SocksProxyAgent;
try { HttpsProxyAgent = require('https-proxy-agent').HttpsProxyAgent; } catch {}
try { SocksProxyAgent = require('socks-proxy-agent').SocksProxyAgent; } catch {}

/**
 * Create a proxy agent from standard proxy environment variables.
 * Checks: ALL_PROXY → HTTPS_PROXY/https_proxy → HTTP_PROXY/http_proxy
 *
 * @returns {import('https').Agent | undefined}
 */
function createProxyAgent() {
  const allProxy = process.env.ALL_PROXY || process.env.all_proxy;

  if (allProxy && SocksProxyAgent) {
    try {
      return new SocksProxyAgent(allProxy);
    } catch {
      // Fall through to HTTP proxy
    }
  }

  const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy ||
                     process.env.HTTP_PROXY || process.env.http_proxy;

  if (httpsProxy && HttpsProxyAgent) {
    try {
      return new HttpsProxyAgent(httpsProxy);
    } catch {
      // No proxy available
    }
  }

  return undefined;
}

// ── Audio constants ──────────────────────────────────────────────────────────
const SAMPLE_RATE = 24000;
const CHANNELS = 1;
const BYTES_PER_SAMPLE = 2; // 16-bit PCM
const FRAME_DURATION_MS = 240;
const FRAME_BYTES = Math.round(SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE * (FRAME_DURATION_MS / 1000)); // 11520

// Jitter buffer: how much audio to buffer before starting playback.
// Trade-off — lower = faster response but risk of audio gaps;
// higher = smoother playback but more perceived latency.
// 48000 bytes ≈ 500ms at 24kHz PCM16 mono.
const JITTER_THRESHOLD_BYTES = 48000;

// ── ANSI escape codes ────────────────────────────────────────────────────────
const ANSI = {
  RESET: '\x1b[0m',
  BOLD: '\x1b[1m',
  DIM: '\x1b[2m',
  RED: '\x1b[31m',
  GREEN: '\x1b[32m',
  YELLOW: '\x1b[33m',
  CYAN: '\x1b[36m',
  CLEAR_LINE: '\x1b[2K',
};

// ── Paths ────────────────────────────────────────────────────────────────────
const FFMPEG_PATH = 'ffmpeg';
const FFPLAY_PATH = 'ffplay';

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    apiKey: process.env.NVIDIA_API_KEY || process.env.NGC_API_KEY,
    wsURL: process.env.NVIDIA_VOICECHAT_WS_URL || CLOUD_WS_URL,
    persona: 'You are a friendly and helpful voice assistant. Keep your responses concise and conversational.',
    timeoutMs: 30_000,
    inputDevice: -1,
    outputDevice: -1,
    listDevices: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`Missing value for ${arg}.`);
      return argv[i];
    };

    if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--api-key') {
      options.apiKey = next();
    } else if (arg === '--base-url') {
      options.wsURL = next();
    } else if (arg === '--persona') {
      options.persona = next();
    } else if (arg === '--timeout-ms') {
      options.timeoutMs = Number.parseInt(next(), 10);
    } else if (arg === '--list-devices') {
      options.listDevices = true;
    } else if (arg === '--input-device') {
      options.inputDevice = Number.parseInt(next(), 10);
    } else if (arg === '--output-device') {
      options.outputDevice = Number.parseInt(next(), 10);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  console.log(`Usage: node nvidia-voicechat-interactive.js [options]

Real-time voice chat with NVIDIA Nemotron 3 VoiceChat.

Speak into your microphone and hear the model respond through your
speakers. Supports full-duplex audio with barge-in — interrupt the
model at any time by speaking.

Uses ffmpeg/ffplay for audio I/O (no native addons required).

Options:
  --api-key <key>          API key. Defaults to NVIDIA_API_KEY or NGC_API_KEY.
  --base-url <url>         WebSocket URL. Default: cloud endpoint
  --persona <text>         Persona prompt for the voice agent.
  --timeout-ms <n>         Connection timeout. Default: 30000
  --list-devices           List audio devices and exit.
  --input-device <id>      Microphone device index. Default: system default
  --output-device <id>     Speaker device index. Default: system default
  --help, -h               Show this help.

Controls:
  Ctrl+C                   Disconnect and exit.

Examples:
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js --list-devices
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js --input-device 1
  NVIDIA_API_KEY=nvapi-... node nvidia-voicechat-interactive.js --persona "You are a pirate"
`);
}

/**
 * List audio devices using ffmpeg's AVFoundation device enumeration.
 */
function listDevices() {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn(FFMPEG_PATH, [
      '-f', 'avfoundation', '-list_devices', 'true', '-i', '""',
    ]);

    let stderr = '';

    ffmpeg.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    ffmpeg.on('close', (code) => {
      // ffmpeg always exits with error during -list_devices (it can't open "")
      const audioSection = stderr.split('AVFoundation audio devices:')[1] || '';
      const lines = audioSection.split('\n')
        .filter(l => /\[AVFoundation/.test(l))
        .map(l => l.trim());

      console.log('Available Audio Devices');
      console.log('═'.repeat(50));

      if (lines.length === 0) {
        console.log('\n  (no audio devices found)');
        resolve();
        return;
      }

      console.log('\nInput (Microphone):');
      for (const line of lines) {
        const match = line.match(/\[(\d+)\]\s+(.+)$/);
        if (match) {
          const idx = parseInt(match[1], 10);
          const name = match[2].replace(/^[""](.+?)[""]$/, '$1');
          console.log(`  [${idx}] ${name}`);
        }
      }

      console.log('\nOutput (Speaker):');
      console.log('  Use System Settings > Sound to select output device.');
      resolve();
    });

    ffmpeg.on('error', (err) => {
      reject(new Error(`ffmpeg not found or not executable: ${err.message}\n` +
        'Install ffmpeg: brew install ffmpeg'));
    });
  });
}

// ── Audio helpers ────────────────────────────────────────────────────────────

/**
 * Spawn ffmpeg for microphone capture.
 * Outputs raw PCM16 mono at SAMPLE_RATE Hz on stdout.
 *
 * @param {number} deviceIndex - AVFoundation audio device index (-1 = system default)
 * @returns {import('child_process').ChildProcess}
 */
function spawnMicCapture(deviceIndex) {
  const deviceArg = deviceIndex >= 0 ? `:${deviceIndex}` : ':default';

  const ffmpeg = spawn(FFMPEG_PATH, [
    '-f', 'avfoundation',
    '-i', deviceArg,
    '-f', 's16le',
    '-ar', String(SAMPLE_RATE),
    '-ac', '1',
    '-loglevel', 'quiet',
    '-',
  ], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return ffmpeg;
}

/**
 * Spawn ffplay for speaker playback.
 * Reads raw PCM16 mono at SAMPLE_RATE Hz from stdin.
 *
 * @param {number} deviceIndex - output device index (-1 = system default)
 * @returns {import('child_process').ChildProcess}
 */
function spawnSpeakerPlayback(deviceIndex) {
  const args = [
    '-f', 's16le',
    '-sample_rate', String(SAMPLE_RATE),
    '-nodisp',
    '-autoexit',
    '-loglevel', 'quiet',
  ];

  // Note: ffplay doesn't expose audio device selection directly.
  // To select a specific output device, set it in System Settings > Sound.
  if (deviceIndex >= 0) {
    args.push('-audio_device_index', String(deviceIndex));
  }

  args.push('-');

  const ffplay = spawn(FFPLAY_PATH, args, {
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  return ffplay;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  loadEnvFile();
  const options = parseArgs();

  if (options.help) {
    printHelp();
    return;
  }

  if (options.listDevices) {
    await listDevices();
    return;
  }

  if (!options.apiKey) {
    throw new Error('Missing API key. Set NVIDIA_API_KEY/NGC_API_KEY or pass --api-key.');
  }

  // ── State ────────────────────────────────────────────────────────────────
  let ws = null;
  let micProcess = null;
  let speakerProcess = null;
  let micBuffer = Buffer.alloc(0);
  let eventIdCounter = 0;
  let isConnected = false;
  let shuttingDown = false;
  let transcriptLineActive = false;
  let lastSpeakerProcessKilled = false;

  // Playback jitter buffer state
  let playbackStarted = false;
  const playbackQueue = [];
  let playbackBytes = 0;

  // ── Helpers ──────────────────────────────────────────────────────────────

  function nextEventId() {
    return `evt_${Date.now()}_${++eventIdCounter}`;
  }

  function endTranscriptLine() {
    if (transcriptLineActive) {
      process.stdout.write('\n');
      transcriptLineActive = false;
    }
  }

  function statusLine(icon, color, text) {
    endTranscriptLine();
    console.log(`${color}${ANSI.BOLD}[${icon}]${ANSI.RESET} ${text}`);
  }

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  // ── Speaker process management ───────────────────────────────────────────

  function ensureSpeakerProcess() {
    if (speakerProcess && !lastSpeakerProcessKilled && speakerProcess.exitCode === null) {
      return;
    }
    // Spawn a fresh speaker process — it waits for data on stdin
    speakerProcess = spawnSpeakerPlayback(options.outputDevice);
    lastSpeakerProcessKilled = false;

    speakerProcess.on('error', (err) => {
      if (!shuttingDown) {
        statusLine('✕', ANSI.RED, `Speaker error: ${err.message}`);
      }
    });

    speakerProcess.stderr.on('data', () => {
      // ffplay sends status info to stderr, we use -loglevel quiet to suppress
    });
  }

  // ── Playback management ──────────────────────────────────────────────────

  function pushPlaybackAudio(pcmData) {
    if (playbackStarted) {
      // Already past threshold — pipe directly to speaker process
      if (speakerProcess && speakerProcess.exitCode === null) {
        speakerProcess.stdin.write(pcmData);
      }
    } else {
      // Buffering: wait until we have enough audio to start
      playbackQueue.push(pcmData);
      playbackBytes += pcmData.length;

      if (playbackBytes >= JITTER_THRESHOLD_BYTES) {
        playbackStarted = true;
        statusLine('▶', ANSI.CYAN, 'Agent is speaking...');

        // Ensure speaker process is running
        ensureSpeakerProcess();

        // Flush buffered audio
        for (const chunk of playbackQueue) {
          if (speakerProcess && speakerProcess.exitCode === null) {
            speakerProcess.stdin.write(chunk);
          }
        }
        playbackQueue.length = 0;
        playbackBytes = 0;
      }
    }
  }

  function clearPlayback() {
    playbackQueue.length = 0;
    playbackBytes = 0;
    playbackStarted = false;

    // Kill the current speaker process to flush audio output immediately.
    // A new one will be spawned when the next response starts.
    if (speakerProcess && speakerProcess.exitCode === null) {
      lastSpeakerProcessKilled = true;
      speakerProcess.kill('SIGTERM');
      // Give it a moment to exit, then forcibly kill
      setTimeout(() => {
        try { speakerProcess.kill('SIGKILL'); } catch {}
      }, 200);
    }
  }

  // ── Graceful shutdown ────────────────────────────────────────────────────

  async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;

    endTranscriptLine();
    statusLine('○', ANSI.YELLOW, 'Disconnecting...');

    if (micProcess) {
      try { micProcess.kill('SIGTERM'); } catch {}
    }
    if (speakerProcess) {
      try { speakerProcess.kill('SIGTERM'); } catch {}
    }
    if (ws) {
      try { ws.close(); } catch {}
    }

    // Wait briefly for child processes to exit
    await new Promise(r => setTimeout(r, 300));

    if (micProcess) {
      try { micProcess.kill('SIGKILL'); } catch {}
    }
    if (speakerProcess) {
      try { speakerProcess.kill('SIGKILL'); } catch {}
    }

    statusLine('○', ANSI.YELLOW, 'Goodbye!');
    process.exit(0);
  }

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  // ── Connect WebSocket ────────────────────────────────────────────────────

  console.log('');
  console.log(`${ANSI.BOLD}Nemotron 3 VoiceChat — Interactive Mode${ANSI.RESET}`);
  console.log(`${ANSI.DIM}──────────────────────────────────────────${ANSI.RESET}`);
  console.log('');

  const safeURL = options.wsURL.replace(/nv-function-id=[^&]+/, 'nv-function-id=...');
  statusLine('○', ANSI.YELLOW, `Connecting to ${safeURL}...`);

  await new Promise((resolve, reject) => {
    const headers = {};
    if (options.apiKey) {
      headers.Authorization = `Bearer ${options.apiKey}`;
    }

    const agent = createProxyAgent();

    const connectTimer = setTimeout(() => {
      reject(new Error(`Connection timed out after ${options.timeoutMs}ms`));
    }, options.timeoutMs);

    ws = new WebSocket(options.wsURL, { headers, agent });

    ws.on('open', () => {
      clearTimeout(connectTimer);
      isConnected = true;
      statusLine('●', ANSI.GREEN, 'Connected');

      // 1. Send function ID (first message per protocol)
      send({ functionId: NVCF_FUNCTION_ID });

      // 2. Send session.update with persona, audio config, and server VAD
      send({
        type: 'session.update',
        event_id: nextEventId(),
        session: {
          instructions: options.persona,
          audio: {
            input: { format: 'pcm16' },
            output: { format: 'pcm16' },
          },
          turn_detection: {
            type: 'server_vad',
          },
        },
      });

      // 3. Start mic capture
      startMicCapture();

      resolve();
    });

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        handleServerMessage(msg);
      } catch {
        // Ignore non-JSON messages
      }
    });

    ws.on('error', (err) => {
      clearTimeout(connectTimer);
      statusLine('✕', ANSI.RED, `WebSocket error: ${err.message}`);
      reject(err);
    });

    ws.on('close', (code) => {
      clearTimeout(connectTimer);
      isConnected = false;
      if (!shuttingDown) {
        statusLine('✕', ANSI.RED, `Disconnected (code=${code})`);
        shutdown();
      }
    });
  });

  // ── Mic capture ──────────────────────────────────────────────────────────

  function startMicCapture() {
    try {
      micProcess = spawnMicCapture(options.inputDevice);
    } catch (err) {
      statusLine('✕', ANSI.RED, `Failed to start mic capture: ${err.message}`);
      shutdown();
      return;
    }

    micProcess.on('error', (err) => {
      if (!shuttingDown) {
        statusLine('✕', ANSI.RED, `Mic error: ${err.message}`);
      }
    });

    micProcess.stderr.on('data', (chunk) => {
      if (!shuttingDown) {
        // ffmpeg writes status/logs to stderr, which we suppress with -loglevel quiet
        // but unexpected data may appear — log it as a warning
        const text = chunk.toString().trim();
        if (text) {
          statusLine('·', ANSI.DIM, `mic: ${text}`);
        }
      }
    });

    // Read raw PCM16 data from ffmpeg stdout, chunk into frames, send via WebSocket
    micProcess.stdout.on('data', (data) => {
      if (shuttingDown || !isConnected) return;

      micBuffer = Buffer.concat([micBuffer, data]);

      while (micBuffer.length >= FRAME_BYTES) {
        const frame = micBuffer.subarray(0, FRAME_BYTES);
        micBuffer = micBuffer.subarray(FRAME_BYTES);

        send({
          type: 'input_audio_buffer.append',
          event_id: nextEventId(),
          audio: frame.toString('base64'),
        });
      }
    });

    micProcess.stdout.on('end', () => {
      if (!shuttingDown) {
        statusLine('○', ANSI.YELLOW, 'Mic stream ended');
      }
    });

    // Pre-spawn the speaker process so it's ready for audio playback
    ensureSpeakerProcess();

    statusLine('●', ANSI.GREEN, 'Recording — speak into your microphone');
    console.log(`${ANSI.DIM}  Press Ctrl+C to exit${ANSI.RESET}`);
    console.log('');
  }

  // ── Server message handler ───────────────────────────────────────────────

  let currentTranscript = '';

  function handleServerMessage(msg) {
    switch (msg.type) {
      case 'session.created':
        statusLine('●', ANSI.GREEN, 'Session created');
        break;

      case 'session.updated':
        statusLine('●', ANSI.GREEN, 'Session configured — persona active');
        break;

      case 'input_audio_buffer.speech_started':
        // Barge-in: user started speaking while model is responding.
        // This is where YOU make a key UX decision about interruption behavior.
        handleBargeIn();
        break;

      case 'input_audio_buffer.speech_stopped':
        // Server VAD detected speech end — it will auto-process
        statusLine('○', ANSI.DIM, 'Processing your speech...');
        break;

      case 'input_audio_buffer.committed':
        // Audio buffer committed for processing — no action needed
        break;

      case 'response.created':
        currentTranscript = '';
        ensureSpeakerProcess();
        break;

      case 'response.audio.delta':
        // Decode base64 PCM16 → jitter buffer → speaker
        if (msg.delta) {
          const pcmData = Buffer.from(msg.delta, 'base64');
          pushPlaybackAudio(pcmData);
        }
        break;

      case 'response.audio_transcript.delta':
        // Stream agent's speech transcript in-place on a single line
        if (msg.delta) {
          if (!transcriptLineActive) {
            process.stdout.write(`${ANSI.CYAN}Agent: ${ANSI.RESET}`);
            transcriptLineActive = true;
          }
          currentTranscript += msg.delta;
          process.stdout.write(msg.delta);
        }
        break;

      case 'response.audio.done':
        // Flush any remaining buffered audio
        if (playbackQueue.length > 0 && speakerProcess && speakerProcess.exitCode === null) {
          const remaining = Buffer.concat(playbackQueue);
          speakerProcess.stdin.write(remaining);
          playbackQueue.length = 0;
          playbackBytes = 0;
        }
        break;

      case 'response.audio_transcript.done':
        // Finalize transcript line
        endTranscriptLine();
        currentTranscript = '';
        break;

      case 'response.done':
        playbackStarted = false;
        statusLine('●', ANSI.GREEN, 'Listening...');
        break;

      case 'response.cancelled':
        clearPlayback();
        endTranscriptLine();
        currentTranscript = '';
        statusLine('●', ANSI.GREEN, 'Listening...');
        break;

      case 'error':
        statusLine('✕', ANSI.RED,
          `Server error: ${msg.error?.message || JSON.stringify(msg)}`);
        break;

      default:
        // Debug: log unknown message types (skip noisy rate limit messages)
        if (msg.type && !msg.type.startsWith('rate_limits')) {
          statusLine('·', ANSI.DIM, msg.type);
        }
        break;
    }
  }

  /**
   * Handle barge-in: the user started speaking while the model is responding.
   *
   * You get to decide the interruption strategy:
   *   - Always cancel immediately (current behavior): best responsiveness,
   *     but background noise can cause false triggers
   *   - Conditional cancel: only cancel if model has been speaking for >N ms,
   *     which reduces false triggers but adds a delay before interruption
   *   - Graceful cancel: let current audio buffer drain before cancelling,
   *     which sounds more natural but adds latency
   */
  function handleBargeIn() {
    statusLine('●', ANSI.YELLOW, 'Barge-in detected — interrupting response');
    send({
      type: 'response.cancel',
      event_id: nextEventId(),
    });
    clearPlayback();
  }

  // Keep the process alive — WebSocket and child processes hold the event loop
  return new Promise(() => {});
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`${ANSI.RED}Error: ${ANSI.RESET}${error.message}`);
    process.exit(1);
  });
}

module.exports = {
  parseArgs,
  listDevices,
  SAMPLE_RATE,
  FRAME_BYTES,
  JITTER_THRESHOLD_BYTES,
  spawnSpeakerPlayback,
};
