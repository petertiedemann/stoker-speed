# stoker-speed

LLM inference benchmark using Bram Stoker's Dracula.  Measures prefill
rate, decode rate, and TTFT against any OpenAI-compatible HTTP endpoint.

## Why not llama-bench / vllm bench / llama-benchy?

| Tool | HTTP? | MTP-aware? | Real prompt? | Real output? |
|------|-------|------------|-------------|--------------|
| `llama-bench` | No (llama.cpp C++ only) | — | Random tokens | Token continuations |
| `vllm bench serve` | Yes | Partial¹ | Random tokens | Token continuations |
| [llama-benchy](https://github.com/eugr/llama-benchy) | Yes | Yes² | Gutenberg book text | Book text continuations |
| **stoker-speed** | Yes | Yes | Dracula (real lit.) | **German translation** |

¹ vLLM bench measures TTFT as time-to-first-chunk, not time-to-first-content-token.  
² llama-benchy is excellent — sweeps, concurrency, prefix-cache measurement. If you
need those features use it. stoker-speed is for quick A/B tests with a realistic
translation workload.

**The unique bit:** stoker-speed asks the model to *translate* Dracula into German.
That means both input and output are real-world text — not random tokens and not
simple next-token prediction.  This matters for engines where output quality
affects generation patterns (MTP/speculative decoding).

## Install

Clone and run.  The only dependency is `requests`.

```bash
git clone https://github.com/petertiedemann/stoker-speed.git
pip install requests   # or let the script auto-install on first run
```

## Usage

```bash
# Default: 5 benchmark runs + 1 warmup, 8192 ctx → 4096 out
python translate_bench.py

# Quick sanity check (single run, no warmup)
python translate_bench.py -n 1 -w 0

# Long context, custom endpoint
python translate_bench.py -c 32000 -o 8192 http://10.0.0.1:8000 qwen3.6-27b

# JSON output for scripting
python translate_bench.py --json -n 10 > results.json

# Reproducible runs
python translate_bench.py --seed 42
```

### Full options

```
python translate_bench.py [-h] [-c CONTEXT] [-o MAX_OUTPUT] [-n NUM_RUNS]
                          [-w WARMUP] [-q] [--json] [--no-random-offset]
                          [--seed SEED]
                          [model] [url]

positional:
  model        Model name sent to the server (default: qwen3.6-27b)
  url          Server base URL (default: http://localhost:8000)

options:
  -c, --context     Approx. context length in tokens (default: 8192)
  -o, --max-output  Max output tokens (default: 4096)
  -n, --num-runs    Benchmark runs in statistics (default: 5)
  -w, --warmup      Warmup runs excluded from stats (default: 1)
  -q, --quiet       Suppress per-run output
  --json            Emit results as JSON
  --no-random-offset  Start Dracula at byte 0 every time
  --seed SEED       Random seed for reproducibility
```

### Example output

```
Model:         qwen3.6-27b
URL:           http://localhost:8000
Context:       ~8192 tokens
Max output:    4096 tokens
Runs:          5  (+ 1 warmup)

  [WARM] run W1: TTFT=0.523s  prefill=16000 t/s  output=387.2 t/s  (4096 tok)
  [BENCH] run R1: TTFT=0.512s  prefill=16350 t/s  output=391.1 t/s  (4100 tok)
  [BENCH] run R2: TTFT=0.541s  prefill=15470 t/s  output=379.8 t/s  (4080 tok)
  [BENCH] run R3: TTFT=0.505s  prefill=16580 t/s  output=394.2 t/s  (4110 tok)
  [BENCH] run R4: TTFT=0.528s  prefill=15850 t/s  output=385.5 t/s  (4070 tok)
  [BENCH] run R5: TTFT=0.518s  prefill=16140 t/s  output=388.9 t/s  (4090 tok)

============================================================
  Benchmark Results  (n=5)
============================================================
  TTFT (s)              : 0.521  (±0.015, med=0.518, min=0.505, max=0.541)
  Gen time (s)          : 10.30  (±0.18, med=10.28, min=10.10, max=10.60)
  Total time (s)        : 10.82  (±0.19, med=10.80, min=10.61, max=11.14)
  Prompt tokens         : 8230  (±85, med=8220, min=8100, max=8400)
  Output tokens         : 4089  (±16, med=4090, min=4070, max=4110)
  Prefill rate (t/s)    : 15866  (±426, med=15910, min=15500, max=16400)
  Output rate (t/s)     : 387.0  (±5.2, med=388.9, min=379.8, max=394.2)
```

## How it works

Each run:
1. Picks a random window of Dracula text (defeats server-side prefix caching)
2. Prepends a random ~50-token noise prefix + a randomly-chosen instruction
3. Sends the prompt to `/v1/chat/completions` with `stream: true` and `include_usage: true`
4. Counts tokens from the server's `usage.completion_tokens` (correct for MTP — multiple
   tokens per decode step are counted individually, not as one chunk)

## Token counting & MTP

stoker-speed uses `usage.completion_tokens` from the server's streaming response.
This is the authoritative token count regardless of how many tokens arrive per
SSE chunk — MTP, speculative decoding, and block streaming are all counted
correctly.  If a server doesn't return `usage` the script falls back to a
character-count estimate and prints a warning.
