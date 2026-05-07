const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildSummary,
  parseArgs,
  TimeoutError,
  withTimeout,
} = require('../nvidia-model-benchmark');

test('parseArgs keeps benchmark defaults and accepts overrides', () => {
  const options = parseArgs([
    '--runs',
    '2',
    '--timeout-ms',
    '5000',
    '--prompt',
    'ping',
    '--max-tokens',
    '8',
    '--delay-ms',
    '100',
    '--output',
    'out.json',
    '--limit',
    '3',
  ]);

  assert.equal(options.runs, 2);
  assert.equal(options.timeoutMs, 5000);
  assert.equal(options.prompt, 'ping');
  assert.equal(options.maxTokens, 8);
  assert.equal(options.delayMs, 100);
  assert.equal(options.output, 'out.json');
  assert.equal(options.limit, 3);
});

test('buildSummary aggregates successful attempts and keeps failure count', () => {
  const summary = buildSummary([
    { model: 'a', ok: true, firstTokenMs: 100, totalMs: 500 },
    { model: 'a', ok: true, firstTokenMs: 150, totalMs: 650 },
    { model: 'a', ok: false, error: 'timeout' },
    { model: 'b', ok: false, error: 'bad request' },
  ]);

  assert.deepEqual(summary, [
    {
      model: 'a',
      attempts: 3,
      successes: 2,
      failures: 1,
      avgFirstTokenMs: 125,
      avgTotalMs: 575,
      minTotalMs: 500,
      maxTotalMs: 650,
    },
    {
      model: 'b',
      attempts: 1,
      successes: 0,
      failures: 1,
      avgFirstTokenMs: null,
      avgTotalMs: null,
      minTotalMs: null,
      maxTotalMs: null,
    },
  ]);
});

test('withTimeout rejects slow work with TimeoutError', async () => {
  await assert.rejects(
    () => withTimeout(() => new Promise((resolve) => setTimeout(resolve, 50)), 5, 'too slow'),
    TimeoutError,
  );
});
