#!/usr/bin/env python3
"""
stoker-speed — LLM inference benchmark using Bram Stoker's Dracula.

Measures prefill rate, decode rate, and TTFT via the OpenAI-compatible
/v1/chat/completions streaming endpoint.  Properly counts tokens for
engines that produce multiple tokens per decode step (MTP/speculative).
"""
import argparse
import json
import math
import os
import random
import statistics
import string
import sys
import time

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


# ── Prompts (multiple variants to bust KV-cache of the instruction itself) ──

_INSTRUCTION_VARIANTS = [
    "Translate this text to German:",
    "Please translate the following to German:",
    "Translate the text below into German:",
    "Render this passage in German:",
    "Do a German translation of this:",
]

_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the following text from "
    "English to German accurately and completely. Maintain the original "
    "meaning, tone, and style."
)


# ── KV-cache busting ────────────────────────────────────────────────────

_WORDS = [
    "apple", "banana", "cherry", "dragon", "eagle", "falcon", "grape",
    "house", "igloo", "jelly", "kite", "lion", "mango", "noble",
    "ocean", "piano", "queen", "river", "stone", "tiger", "umbra",
    "violet", "waltz", "xenon", "yacht", "zebra",
]


def random_buster(num_words: int = 12, suffix_len: int = 8) -> str:
    """A short random prefix changed on every request — defeats prefix caching.

    Returns ~50 tokens of random text that is prepended *after* the
    instruction.  Combined with a random Dracula offset, the entire
    prompt is unique per run — no prefix or suffix cache can help.
    """
    words = random.choices(_WORDS, k=random.randint(num_words - 3, num_words + 3))
    suffix = "".join(random.choices(string.ascii_lowercase, k=suffix_len))
    return " ".join(words) + ". " + suffix + "\n\n"


# ── Dracula loading ─────────────────────────────────────────────────────

def load_dracula(script_dir: str) -> str:
    """Find and load dracula.txt, stripping Project Gutenberg boilerplate."""
    candidates = [
        os.path.join(script_dir, "dracula.txt"),
        os.path.join(script_dir, "data", "dracula.txt"),
        os.path.join(script_dir, "..", "data", "dracula.txt"),
        "/tmp/dracula.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            start = "*** START OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
            end = "*** END OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
            if start in text:
                text = text.split(start, 1)[1]
            if end in text:
                text = text.split(end, 1)[0]
            return text.strip()
    raise FileNotFoundError(
        "dracula.txt not found.  Place it next to translate_bench.py "
        "or at /tmp/dracula.txt"
    )


def pick_context(dracula: str, approx_tokens: int,
                 chars_per_token: float = 4.0,
                 offset: int | None = None) -> tuple[str, int]:
    """Return a window of *dracula* starting at *offset* (random if None).

    Randomising the offset defeats suffix-based / content-addressed
    prefix caches that can match token blocks regardless of position.
    """
    char_window = int(approx_tokens * chars_per_token)
    max_offset = max(0, len(dracula) - char_window - 500)
    if offset is None:
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
    else:
        offset = max(0, min(offset, max_offset))
    end = min(offset + char_window, len(dracula))
    return dracula[offset:end], offset


# ── Single-run measurement ──────────────────────────────────────────────

def run_one(model: str, base_url: str, context_tokens: int,
            max_output: int, dracula: str, run_label: str,
            offset: int | None = None) -> dict:
    """Execute one benchmark request and return timing / token metrics.

    Token counts are taken from the server's ``usage`` payload
    (streaming with ``include_usage``), which correctly reports the
    true number of generated tokens — essential for MTP where a single
    SSE chunk can represent multiple tokens.
    """
    context_text, actual_offset = pick_context(dracula, context_tokens,
                                                offset=offset)
    buster = random_buster()
    instruction = random.choice(_INSTRUCTION_VARIANTS)

    user_prompt = f"{instruction}\n\n{buster}{context_text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output,
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    first_token_time = None
    usage = None
    response_text = ""

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload, stream=True, timeout=900,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Usage may arrive in its own chunk or the final chunk
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            # First token timing — triggered by reasoning OR content,
            # whichever arrives first.
            has_output = (
                delta.get("reasoning")
                or delta.get("reasoning_content")
                or delta.get("content")
            )
            if has_output and first_token_time is None:
                first_token_time = time.perf_counter()

            if delta.get("content"):
                response_text += delta["content"]

    t_end = time.perf_counter()

    # ── Metrics ─────────────────────────────────────────────────────
    total_time = t_end - t_start
    ttft = (first_token_time - t_start) if first_token_time else None
    gen_time = (t_end - first_token_time) if first_token_time else total_time

    # Authoritative token counts from the server (includes MTP accounting).
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        token_source = "server"
    else:
        # Fallback: rough estimate from character count.
        # Much better than counting SSE chunks, but still approximate.
        prompt_tokens = context_tokens  # user-requested value is our best guess
        completion_tokens = max(1, len(response_text) // 4)
        token_source = "estimated"

    # Rates
    prefill_rate = None
    if ttft and ttft > 0 and prompt_tokens:
        prefill_rate = prompt_tokens / ttft

    output_rate = None
    if gen_time > 0 and completion_tokens:
        output_rate = completion_tokens / gen_time

    return {
        "run": run_label,
        "token_source": token_source,
        "ttft": ttft,
        "gen_time": gen_time,
        "total_time": total_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_rate": prefill_rate,
        "output_rate": output_rate,
        "response_chars": len(response_text),
        "dracula_offset": actual_offset,
    }


# ── Statistics ──────────────────────────────────────────────────────────

STAT_COLUMNS = [
    ("TTFT (s)",           "ttft",            ".3f"),
    ("Gen time (s)",       "gen_time",        ".2f"),
    ("Total time (s)",     "total_time",      ".2f"),
    ("Prompt tokens",      "prompt_tokens",   ".0f"),
    ("Output tokens",      "completion_tokens", ".0f"),
    ("Prefill rate (t/s)", "prefill_rate",    ".0f"),
    ("Output rate (t/s)",  "output_rate",     ".1f"),
]


def print_stats(results: list[dict], label: str) -> None:
    """Pretty-print mean / stddev / median / min / max for each metric."""
    n = len(results)
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {label}  (n={n})")
    print(f"{bar}")
    for col_name, key, fmt in STAT_COLUMNS:
        values = [r[key] for r in results if r.get(key) is not None]
        if not values:
            print(f"  {col_name:<22}:  (no data)")
            continue
        if n >= 3:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            med = statistics.median(values)
            mn, mx = min(values), max(values)
            print(
                f"  {col_name:<22}: {mean:{fmt}}  "
                f"(±{stdev:{fmt}}, med={med:{fmt}}, "
                f"min={mn:{fmt}}, max={mx:{fmt}})"
            )
        else:
            mean = statistics.mean(values)
            print(f"  {col_name:<22}: {mean:{fmt}}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="stoker-speed — LLM inference benchmark (Dracula translation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python translate_bench.py                              # defaults: 5 runs + 1 warmup
  python translate_bench.py -n 1 -w 0                    # single quick run, no warmup
  python translate_bench.py qwen3.6-27b http://10.0.0.1:8000 -c 32000 -o 8192
  python translate_bench.py -n 20 --json > results.json  # JSON for analysis
        """,
    )
    parser.add_argument(
        "model", nargs="?", default="qwen3.6-27b",
        help="Model name sent to the server (default: qwen3.6-27b)")
    parser.add_argument(
        "url", nargs="?", default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)")
    parser.add_argument(
        "-c", "--context", type=int, default=8192,
        help="Approximate context length in tokens (default: 8192)")
    parser.add_argument(
        "-o", "--max-output", type=int, default=4096,
        help="Max output tokens (default: 4096)")
    parser.add_argument(
        "-n", "--num-runs", type=int, default=5,
        help="Number of benchmark runs included in statistics (default: 5)")
    parser.add_argument(
        "-w", "--warmup", type=int, default=1,
        help="Number of warmup runs excluded from statistics (default: 1)")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-run output; print only summary")
    parser.add_argument(
        "--json", action="store_true",
        help="Emit benchmark results as JSON (implies --quiet)")
    parser.add_argument(
        "--no-random-offset", action="store_true",
        help="Always start Dracula at the beginning (weaker cache busting)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dracula = load_dracula(os.path.dirname(os.path.abspath(__file__)))

    if not args.quiet and not args.json:
        print(f"Model:         {args.model}")
        print(f"URL:           {args.url}")
        print(f"Context:       ~{args.context} tokens")
        print(f"Max output:    {args.max_output} tokens")
        print(f"Runs:          {args.num_runs}  (+ {args.warmup} warmup)")
        print(f"Dracula:       {len(dracula)} chars available")
        print()

    all_results: list[dict] = []
    total = args.warmup + args.num_runs

    for i in range(1, total + 1):
        is_warmup = i <= args.warmup
        run_label = f"W{i}" if is_warmup else f"R{i - args.warmup}"
        offset = None if args.no_random_offset else None  # random inside run_one

        result = run_one(
            args.model, args.url, args.context, args.max_output,
            dracula, run_label, offset=offset,
        )
        result["warmup"] = is_warmup
        all_results.append(result)

        if not args.quiet and not args.json:
            tag = "WARM" if is_warmup else "BENCH"
            ttft_s = f"{result['ttft']:.3f}s" if result["ttft"] else "N/A"
            p_s = f"{result['prefill_rate']:.0f} t/s" if result["prefill_rate"] else "N/A"
            o_s = f"{result['output_rate']:.1f} t/s" if result["output_rate"] else "N/A"
            src = "" if result["token_source"] == "server" else " [est]"
            print(
                f"  [{tag}] run {run_label}: "
                f"TTFT={ttft_s}  prefill={p_s}  output={o_s}"
                f"  ({result['completion_tokens']} tok{src})"
            )

    # ── Summarise ───────────────────────────────────────────────────
    bench = [r for r in all_results if not r["warmup"]]
    warm = [r for r in all_results if r["warmup"]]

    if args.json:
        # Don't include internal keys in JSON output
        json_out = [{k: v for k, v in r.items() if k != "warmup"}
                     for r in bench]
        print(json.dumps(json_out, indent=2))
    else:
        if warm:
            print_stats(warm, "Warmup  (excluded from results)")
        print_stats(bench, "Benchmark Results")
        print()

        # Warn if token counts were estimated
        if any(r["token_source"] == "estimated" for r in bench):
            print("⚠  Token counts were ESTIMATED (server did not return usage).")
            print("   Output rate may be inaccurate for engines with MTP.")
            print()


if __name__ == "__main__":
    main()
