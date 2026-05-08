const assert = require('node:assert/strict');
const test = require('node:test');

const {
  AVAILABLE_VOICES,
  parseArgs: parseTtsArgs,
} = require('../nvidia-tts-zeroshot-test');

const {
  MODEL_ID,
  NVCF_FUNCTION_ID,
  CLOUD_WS_URL,
  parseArgs: parseVoicechatArgs,
} = require('../nvidia-voicechat-test');



// --- TTS Zeroshot Tests ---

test('tts-zeroshot: parseArgs keeps defaults', () => {
  const options = parseTtsArgs([]);
  assert.equal(options.voice, 'Magpie-ZeroShot.Female-1');
  assert.equal(options.language, 'en-US');
  assert.equal(options.quality, 20);
  assert.equal(options.runs, 3);
  assert.equal(options.timeoutMs, 60_000);
  assert.equal(options.delayMs, 2000);
  assert.equal(options.audioPrompt, null);
  assert.equal(options.listVoices, false);
});

test('tts-zeroshot: parseArgs accepts overrides', () => {
  const options = parseTtsArgs([
    '--text', 'Hello',
    '--voice', 'Magpie-ZeroShot.Male-1',
    '--quality', '30',
    '--language', 'zh-CN',
    '--runs', '5',
    '--output', 'test.wav',
    '--audio-prompt', 'prompt.wav',
  ]);
  assert.equal(options.text, 'Hello');
  assert.equal(options.voice, 'Magpie-ZeroShot.Male-1');
  assert.equal(options.quality, 30);
  assert.equal(options.language, 'zh-CN');
  assert.equal(options.runs, 5);
  assert.equal(options.output, 'test.wav');
  assert.equal(options.audioPrompt, 'prompt.wav');
});

test('tts-zeroshot: parseArgs rejects invalid quality', () => {
  assert.throws(() => parseTtsArgs(['--quality', '0']), /between 1 and 40/);
  assert.throws(() => parseTtsArgs(['--quality', '41']), /between 1 and 40/);
});

test('tts-zeroshot: AVAILABLE_VOICES contains expected voices', () => {
  assert.ok(AVAILABLE_VOICES.length >= 11);
  assert.ok(AVAILABLE_VOICES.includes('Magpie-ZeroShot.Female-1'));
  assert.ok(AVAILABLE_VOICES.includes('Magpie-ZeroShot.Male-1'));
  assert.ok(AVAILABLE_VOICES.includes('Magpie-ZeroShot.Female-Happy'));
});

// --- VoiceChat Tests ---

test('voicechat: MODEL_ID is correct', () => {
  assert.equal(MODEL_ID, 'nvidia/nemotron-voicechat');
});

test('voicechat: NVCF_FUNCTION_ID is set', () => {
  assert.equal(NVCF_FUNCTION_ID, '42c86b5f-545a-4b2f-a83b-90fd71da9912');
});

test('voicechat: CLOUD_WS_URL contains function ID', () => {
  assert.ok(CLOUD_WS_URL.includes(NVCF_FUNCTION_ID));
  assert.ok(CLOUD_WS_URL.startsWith('wss://'));
});

test('voicechat: parseArgs keeps defaults', () => {
  const options = parseVoicechatArgs([]);
  assert.equal(options.mode, 'health');
  assert.equal(options.timeoutMs, 30_000);
  assert.equal(options.audioFile, null);
  assert.equal(options.persona, null);
});

test('voicechat: parseArgs accepts overrides', () => {
  const options = parseVoicechatArgs([
    '--mode', 'info',
    '--base-url', 'ws://localhost:9000/v1/realtime',
    '--timeout-ms', '10000',
    '--persona', 'You are a helpful assistant.',
  ]);
  assert.equal(options.mode, 'info');
  assert.equal(options.wsURL, 'ws://localhost:9000/v1/realtime');
  assert.equal(options.timeoutMs, 10000);
  assert.equal(options.persona, 'You are a helpful assistant.');
});

test('voicechat: parseArgs rejects invalid mode', () => {
  assert.throws(() => parseVoicechatArgs(['--mode', 'invalid']), /must be one of/);
});

// --- VoiceChat Interactive Tests ---

const {
  SAMPLE_RATE,
  FRAME_BYTES,
  JITTER_THRESHOLD_BYTES,
  parseArgs: parseInteractiveArgs,
} = require('../nvidia-voicechat-interactive');

test('voicechat-interactive: parseArgs keeps defaults', () => {
  const options = parseInteractiveArgs([]);
  assert.equal(options.timeoutMs, 30_000);
  assert.equal(options.inputDevice, -1);
  assert.equal(options.outputDevice, -1);
  assert.equal(options.listDevices, false);
  assert.ok(options.persona.length > 0);
});

test('voicechat-interactive: parseArgs accepts overrides', () => {
  const options = parseInteractiveArgs([
    '--input-device', '3',
    '--output-device', '5',
    '--persona', 'You are a pirate',
    '--timeout-ms', '15000',
    '--base-url', 'ws://localhost:9000/v1/realtime',
  ]);
  assert.equal(options.inputDevice, 3);
  assert.equal(options.outputDevice, 5);
  assert.equal(options.persona, 'You are a pirate');
  assert.equal(options.timeoutMs, 15000);
  assert.equal(options.wsURL, 'ws://localhost:9000/v1/realtime');
});

test('voicechat-interactive: parseArgs sets listDevices flag', () => {
  const options = parseInteractiveArgs(['--list-devices']);
  assert.equal(options.listDevices, true);
});

test('voicechat-interactive: audio constants are correct', () => {
  assert.equal(SAMPLE_RATE, 24000);
  // 24kHz * 1ch * 2bytes * 240ms = 11520
  assert.equal(FRAME_BYTES, 11520);
  assert.ok(JITTER_THRESHOLD_BYTES > 0);
});

test('voicechat-interactive: parseArgs rejects unknown argument', () => {
  assert.throws(() => parseInteractiveArgs(['--bogus']), /Unknown argument/);
});
