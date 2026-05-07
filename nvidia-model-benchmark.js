#!/usr/bin/env node

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const DEFAULT_BASE_URL = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_PROMPT = '请用一句中文回答：今天适合做什么？';

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
    if (!line || line.startsWith('#')) {
      continue;
    }

    const normalizedLine = line.startsWith('export ') ? line.slice('export '.length).trim() : line;
    const separatorIndex = normalizedLine.indexOf('=');
    if (separatorIndex === -1) {
      continue;
    }

    const key = normalizedLine.slice(0, separatorIndex).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || Object.prototype.hasOwnProperty.call(env, key)) {
      continue;
    }

    const value = normalizedLine.slice(separatorIndex + 1);
    env[key] = stripOptionalQuotes(value);
  }

  return true;
}

class TimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'TimeoutError';
  }
}

function parsePositiveInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function parseNonNegativeInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return parsed;
}

function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    apiKey: process.env.NVIDIA_API_KEY,
    baseURL: process.env.NVIDIA_BASE_URL || DEFAULT_BASE_URL,
    delayMs: 1500,
    limit: null,
    maxTokens: 128,
    output: `nvidia-model-benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
    prompt: DEFAULT_PROMPT,
    runs: 3,
    timeoutMs: 60_000,
    temperature: 0,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) {
        throw new Error(`Missing value for ${arg}.`);
      }
      return argv[index];
    };

    if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--api-key') {
      options.apiKey = next();
    } else if (arg === '--base-url') {
      options.baseURL = next();
    } else if (arg === '--delay-ms') {
      options.delayMs = parseNonNegativeInt(next(), '--delay-ms');
    } else if (arg === '--limit') {
      options.limit = parsePositiveInt(next(), '--limit');
    } else if (arg === '--max-tokens') {
      options.maxTokens = parsePositiveInt(next(), '--max-tokens');
    } else if (arg === '--output' || arg === '-o') {
      options.output = next();
    } else if (arg === '--prompt') {
      options.prompt = next();
    } else if (arg === '--runs') {
      options.runs = parsePositiveInt(next(), '--runs');
    } else if (arg === '--temperature') {
      options.temperature = Number(next());
      if (!Number.isFinite(options.temperature)) {
        throw new Error('--temperature must be a number.');
      }
    } else if (arg === '--timeout-ms') {
      options.timeoutMs = parsePositiveInt(next(), '--timeout-ms');
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  console.log(`Usage: node nvidia-model-benchmark.js [options]

Benchmarks every model returned by NVIDIA's OpenAI-compatible model list.

Options:
  --api-key <key>       NVIDIA API key. Defaults to NVIDIA_API_KEY.
  --base-url <url>      API base URL. Defaults to NVIDIA_BASE_URL or ${DEFAULT_BASE_URL}.
  --prompt <text>       Prompt used for every test call.
  --runs <n>            Attempts per model. Default: 3.
  --timeout-ms <n>      Maximum wait per attempt. Default: 60000.
  --max-tokens <n>      Completion max tokens. Default: 128.
  --delay-ms <n>        Delay between attempts. Default: 1500.
  --output, -o <path>   JSON output path.
  --limit <n>           Optional model count limit for smoke tests.
  --help, -h            Show this help.

Example:
  NVIDIA_API_KEY=nvapi-... node nvidia-model-benchmark.js --output results.json
`);
}

function round(value) {
  return Math.round(value);
}

function avg(values) {
  if (values.length === 0) {
    return null;
  }
  return round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function buildSummary(results) {
  const grouped = new Map();

  for (const result of results) {
    if (!grouped.has(result.model)) {
      grouped.set(result.model, []);
    }
    grouped.get(result.model).push(result);
  }

  return Array.from(grouped.entries()).map(([model, attempts]) => {
    const successes = attempts.filter((attempt) => attempt.ok);
    const totalTimes = successes.map((attempt) => attempt.totalMs);
    const firstTokenTimes = successes
      .map((attempt) => attempt.firstTokenMs)
      .filter((value) => value !== null && value !== undefined);

    return {
      model,
      attempts: attempts.length,
      successes: successes.length,
      failures: attempts.length - successes.length,
      avgFirstTokenMs: avg(firstTokenTimes),
      avgTotalMs: avg(totalTimes),
      minTotalMs: totalTimes.length > 0 ? Math.min(...totalTimes) : null,
      maxTotalMs: totalTimes.length > 0 ? Math.max(...totalTimes) : null,
    };
  });
}

function formatError(error) {
  return {
    name: error.name,
    message: error.message,
    code: error.code,
    status: error.status,
  };
}

function withTimeout(work, timeoutMs, label = 'operation') {
  const controller = new AbortController();

  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      controller.abort();
      reject(new TimeoutError(`${label} timed out after ${timeoutMs}ms.`));
    }, timeoutMs);

    Promise.resolve()
      .then(() => work(controller.signal))
      .then((value) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        reject(error);
      });
  });
}

async function listModels(client) {
  const response = await client.models.list();
  const models = response.data || [];
  return models.map((model) => model.id).filter(Boolean);
}

async function benchmarkAttempt(client, model, attempt, options) {
  const startedAt = performance.now();
  let firstTokenMs = null;
  let responseText = '';

  try {
    await withTimeout(async (signal) => {
      const stream = await client.chat.completions.create(
        {
          model,
          messages: [{ role: 'user', content: options.prompt }],
          stream: true,
          max_tokens: options.maxTokens,
          temperature: options.temperature,
        },
        { signal },
      );

      for await (const chunk of stream) {
        const content = chunk.choices?.[0]?.delta?.content || '';
        if (content && firstTokenMs === null) {
          firstTokenMs = round(performance.now() - startedAt);
        }
        responseText += content;
      }
    }, options.timeoutMs, `${model} attempt ${attempt}`);

    return {
      model,
      attempt,
      ok: true,
      firstTokenMs,
      totalMs: round(performance.now() - startedAt),
      responseChars: responseText.length,
      responsePreview: responseText.slice(0, 200),
    };
  } catch (error) {
    return {
      model,
      attempt,
      ok: false,
      firstTokenMs,
      totalMs: round(performance.now() - startedAt),
      responseChars: responseText.length,
      responsePreview: responseText.slice(0, 200),
      error: formatError(error),
    };
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runBenchmark(client, options) {
  const allModels = await listModels(client);
  const models = options.limit ? allModels.slice(0, options.limit) : allModels;
  const results = [];

  console.log(`Found ${allModels.length} models. Benchmarking ${models.length} model(s).`);
  console.log(`Runs per model: ${options.runs}; timeout per attempt: ${options.timeoutMs}ms.`);
  console.log('');

  for (const [modelIndex, model] of models.entries()) {
    console.log(`[${modelIndex + 1}/${models.length}] ${model}`);

    for (let attempt = 1; attempt <= options.runs; attempt += 1) {
      const result = await benchmarkAttempt(client, model, attempt, options);
      results.push(result);

      if (result.ok) {
        console.log(
          `  attempt ${attempt}: ok, first token ${result.firstTokenMs ?? 'n/a'}ms, total ${result.totalMs}ms, chars ${result.responseChars}`,
        );
      } else {
        console.log(
          `  attempt ${attempt}: failed after ${result.totalMs}ms, ${result.error.name}: ${result.error.message}`,
        );
      }

      const isLastAttempt = modelIndex === models.length - 1 && attempt === options.runs;
      if (!isLastAttempt && options.delayMs > 0) {
        await sleep(options.delayMs);
      }
    }
  }

  const summary = buildSummary(results);
  return {
    metadata: {
      baseURL: options.baseURL,
      prompt: options.prompt,
      runs: options.runs,
      timeoutMs: options.timeoutMs,
      maxTokens: options.maxTokens,
      delayMs: options.delayMs,
      totalModelsReturned: allModels.length,
      totalModelsBenchmarked: models.length,
      startedAt: new Date().toISOString(),
    },
    results,
    summary,
  };
}

async function main(extraClientOptions = {}) {
  loadEnvFile();
  const options = parseArgs();

  if (options.help) {
    printHelp();
    return;
  }

  if (!options.apiKey) {
    throw new Error('Missing NVIDIA API key. Set NVIDIA_API_KEY or pass --api-key.');
  }

  const OpenAI = require('openai');
  const client = new OpenAI({
    apiKey: options.apiKey,
    baseURL: options.baseURL,
    ...extraClientOptions,
  });

  const report = await runBenchmark(client, options);
  const outputPath = path.resolve(options.output);
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);

  console.log('');
  console.log(`Wrote benchmark report to ${outputPath}`);
  console.table(report.summary);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = {
  DEFAULT_BASE_URL,
  TimeoutError,
  benchmarkAttempt,
  buildSummary,
  loadEnvFile,
  main,
  parseArgs,
  runBenchmark,
  withTimeout,
};
