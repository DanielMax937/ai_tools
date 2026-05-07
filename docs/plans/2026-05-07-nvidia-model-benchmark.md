# NVIDIA Model Benchmark Plan

## Goal

Add a standalone Node.js script that lists NVIDIA models through the OpenAI-compatible API and benchmarks each returned model three times.

## Success Criteria

- Fetch every model from `client.models.list()`.
- Call each model sequentially three times by default.
- Enforce a maximum one-minute timeout for each single model call.
- Record time to first content token and total response time.
- Preserve useful failure details without stopping the whole run.
- Output machine-readable JSON plus readable console progress.

## Approach

1. Add `nvidia-model-benchmark.js`.
2. Use `NVIDIA_API_KEY` and default base URL `https://integrate.api.nvidia.com/v1`.
3. Implement argument parsing for prompt, runs, timeout, max tokens, delay, output path, and optional model limit.
4. Stream responses to measure first-token latency and full completion latency.
5. Add pure helper exports so tests can validate timeout handling, summary math, and argument parsing without real network calls.
6. Add `test/nvidia-model-benchmark.test.js` using Node's built-in test runner.
7. Add npm scripts for the benchmark and unit test.
8. Run tests and a help command as verification.

## Steps

1. Create a failing unit test for argument parsing and summary generation.
2. Add the benchmark script with exported helpers.
3. Wire package scripts.
4. Run the unit test and fix failures.
5. Run the script help command to confirm CLI usability.
