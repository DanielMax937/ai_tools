#!/usr/bin/env node

const DEFAULT_PROXY_URL = 'http://127.0.0.1:1087';

function createProxyFetch(proxyUrl = DEFAULT_PROXY_URL) {
  const { ProxyAgent } = require('undici');
  const dispatcher = new ProxyAgent(proxyUrl);

  return (input, init = {}) => fetch(input, {
    ...init,
    dispatcher,
  });
}

async function main() {
  const {
    loadEnvFile,
    main: runBenchmarkCli,
  } = require('./nvidia-model-benchmark');

  loadEnvFile();

  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    await runBenchmarkCli();
    return;
  }

  const proxyUrl = process.env.NVIDIA_BENCHMARK_PROXY_URL || DEFAULT_PROXY_URL;
  console.log(`Using proxy: ${proxyUrl}`);
  await runBenchmarkCli({
    fetch: createProxyFetch(proxyUrl),
  });
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = {
  DEFAULT_PROXY_URL,
  createProxyFetch,
};
