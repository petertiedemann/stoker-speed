#!/usr/bin/env python3
"""Translation benchmark: prefill and decode using public-domain text (Dracula).
   Context = first K tokens of Dracula, task = translate to German.
   A random prefix is prepended to bust any prefix caching.
"""
import json, os, sys, time, math, random, string

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-27b"
BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
CONTEXT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
MAX_OUTPUT_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

def random_buster():
    """Random prefix to bust prefix caching. ~50 tokens."""
    words = random.choices([
        "apple", "banana", "cherry", "dragon", "eagle", "falcon", "grape",
        "house", "igloo", "jelly", "kite", "lion", "mango", "noble",
        "ocean", "piano", "queen", "river", "stone", "tiger", "umbra",
        "violet", "waltz", "xenon", "yacht", "zebra"
    ], k=random.randint(10, 15))
    return " ".join(words) + ". " + "".join(random.choices(string.ascii_lowercase, k=8)) + "\n\n"

DETAILED = os.environ.get("DETAILED", "").lower() in ("1", "yes", "true")

# Load Dracula text
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRACULA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "dracula.txt")
if not os.path.exists(DRACULA_PATH):
    DRACULA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "dracula.txt")
if not os.path.exists(DRACULA_PATH):
    DRACULA_PATH = "/tmp/dracula.txt"
with open(DRACULA_PATH, "r", encoding="utf-8") as f:
    dracula_text = f.read()

# Strip Gutenberg headers/footer
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK DRACULA ***"
if START_MARKER in dracula_text:
    dracula_text = dracula_text.split(START_MARKER, 1)[1]
if END_MARKER in dracula_text:
    dracula_text = dracula_text.split(END_MARKER, 1)[0]
dracula_text = dracula_text.strip()

# Rough token estimate: ~4 chars per token for English
chars_per_token = 4.0
context_chars = min(int(CONTEXT_TOKENS * chars_per_token), len(dracula_text))
context_text = dracula_text[:context_chars]

# Random prefix to bust any prefix caching
buster = random_buster()

system_prompt = f"You are a professional translator. Translate the following text from English to German accurately and completely. Maintain the original meaning, tone, and style."

user_prompt = f"Translate this text to German:\n\n{buster}{context_text}"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "max_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.3,
    "stream": True,
    "stream_options": {"include_usage": True},
}

print(f"Model: {MODEL}")
print(f"URL: {BASE_URL}")
print(f"Context: ~{CONTEXT_TOKENS} tokens ({len(context_text)} chars)")
print(f"Max output: {MAX_OUTPUT_TOKENS} tokens")
print(f"Source: Dracula, {len(dracula_text)} chars total")
print()

t_start = time.perf_counter()
first_token_time = None
reasoning_chunks = 0
content_chunks = 0
usage = None
response_text = ""

with requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=900) as resp:
    resp.raise_for_status()
    for raw_line in resp.iter_lines():
        if not raw_line: continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "): continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]": break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices: continue
        delta = choices[0].get("delta", {})

        if delta.get("reasoning") or delta.get("reasoning_content"):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            reasoning_chunks += 1

        if delta.get("content"):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            content_chunks += 1
            response_text += delta["content"]

t_end = time.perf_counter()
total_time = t_end - t_start
ttft = (first_token_time - t_start) if first_token_time else None
gen_time = (t_end - first_token_time) if first_token_time else total_time

server_tokens = usage.get("completion_tokens") if usage else None
server_prompt = usage.get("prompt_tokens") if usage else None
chunk_total = reasoning_chunks + content_chunks

tokens = server_tokens if server_tokens is not None else chunk_total
tps = tokens / gen_time if gen_time > 0 and tokens > 0 else 0
prompt_tokens_used = server_prompt if server_prompt else CONTEXT_TOKENS

# Prefill rate: prompt_tokens / ttft
if ttft and ttft > 0:
    prefill_rate = prompt_tokens_used / ttft
else:
    prefill_rate = None

print(f"TTFT:           {ttft:.2f}s" if ttft else "TTFT:           N/A")
print(f"Gen time:       {gen_time:.2f}s")
print(f"Total time:     {total_time:.2f}s")
print(f"Prompt tokens:  {prompt_tokens_used}")
print(f"Output tokens:  {tokens}")
print(f"Avg prefill:    {prefill_rate:.1f} tok/s" if prefill_rate else "Avg prefill:    N/A")
print(f"Output rate:    {tps:.1f} tok/s")
print(f"Response len:   {len(response_text)} chars")
