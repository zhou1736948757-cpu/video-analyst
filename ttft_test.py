#!/usr/bin/env python3
"""
Measure TTFT (Time to First Token) and tokens/s for Anthropic Claude API (streaming).
Works with the official API and any Anthropic-format relay/proxy.

Usage:
  # 官方 API
  export ANTHROPIC_API_KEY=sk-ant-...
  ./ttft_test.py

  # 中转站 / 反代 (--api-key 或 PROXY_API_KEY 环境变量)
  # 路径 /v1/messages 会自动补全，也可以写完整 URL
  ./ttft_test.py --base-url https://minnimax.chat --api-key gw-xxx
  export PROXY_API_KEY=gw-xxx
  ./ttft_test.py --base-url https://minnimax.chat

  ./ttft_test.py --model claude-opus-4-8 --iterations 3
  ./ttft_test.py --prompt "你好" --max-tokens 200 --iterations 5

Notes:
  - Python 3.7+ (uses only stdlib; no pip install needed)
  - Output tokens count comes from message_delta.usage.output_tokens (cumulative)
  - TTFT is measured from request send to first message_start event
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime


def stream_request(
    model, prompt, api_key, base_url, auth_header="x-api-key", max_tokens=1024, timeout=60
):
    """Send a streaming request; return (ttft, input_tokens, output_tokens, elapsed, text)."""
    # Auto-append /v1/messages if base_url is just the host
    url = base_url.rstrip("/")
    if not url.endswith("/v1/messages"):
        url = url + "/v1/messages"

    if auth_header == "bearer":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers
    )

    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    input_tokens = 0
    full_text_parts = []

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("content-type", "")
            raw_lines = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                raw_lines.append(line)
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "message_start":
                    # First event back from server → TTFT
                    ttft = time.perf_counter() - start
                    usage = event.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)

                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        full_text_parts.append(delta.get("text", ""))

                elif etype == "message_delta":
                    # usage.output_tokens is cumulative
                    usage = event.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                elif etype == "message_stop":
                    break

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from None
    except TimeoutError:
        raise RuntimeError(f"Timeout after {timeout}s") from None

    # If we never got message_start, dump what we did receive for debugging
    if ttft is None:
        sample = "\n".join(raw_lines[:20])
        raise RuntimeError(
            f"No message_start event received (status={status}, "
            f"content-type={content_type!r}, {len(raw_lines)} raw lines).\n"
            f"First lines:\n{sample}"
        )

    elapsed = time.perf_counter() - start
    return ttft, input_tokens, output_tokens, elapsed, "".join(full_text_parts)


def main():
    p = argparse.ArgumentParser(
        description="Measure Claude API TTFT and tokens/s (streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default="MiniMax-M3",
        help="Model ID (default: MiniMax-M3)",
    )
    p.add_argument(
        "--prompt",
        default=(
            "请用中文详细解释一下 Transformer 架构的工作原理，"
            "包括自注意力机制、多头注意力和位置编码，字数控制在 500 字左右。"
        ),
        help="Test prompt",
    )
    p.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    p.add_argument(
        "--base-url",
        default="https://minnimax.chat",
        help="API endpoint URL (default: proxy)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key (else: PROXY_API_KEY or ANTHROPIC_API_KEY env var)",
    )
    p.add_argument(
        "--auth-header",
        default="x-api-key",
        choices=["x-api-key", "bearer"],
        help="Header style: 'x-api-key' (Anthropic) or 'bearer' (OpenAI-style)",
    )
    p.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Number of test runs (median reported when >1)",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="Run one un-timed warm-up before iterations (recommended for >1)",
    )
    args = p.parse_args()

    # Resolve API key from (in priority order): --api-key > PROXY_API_KEY > ANTHROPIC_API_KEY
    api_key = (
        args.api_key
        or os.environ.get("PROXY_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print(
            "Error: need --api-key, or PROXY_API_KEY / ANTHROPIC_API_KEY env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1/messages"):
        endpoint = endpoint + "/v1/messages"

    print(f"Model:      {args.model}")
    print(f"Endpoint:   {endpoint}")
    print(f"Auth:       {args.auth_header}")
    key_preview = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
    print(f"API key:    {key_preview}")
    print(f"Iterations: {args.iterations}")
    prompt_preview = args.prompt if len(args.prompt) <= 70 else args.prompt[:67] + "..."
    print(f"Prompt:     {prompt_preview}")
    print(f"Started:    {datetime.now().isoformat()}")
    print()

    if args.warmup:
        print("--- Warm-up ---")
        try:
            stream_request(
                args.model, args.prompt, api_key, args.base_url, args.auth_header, args.max_tokens
            )
        except Exception as e:
            print(f"  warm-up failed: {e}")
        print()

    results = []  # list of dicts

    for i in range(args.iterations):
        print(f"--- Run {i + 1}/{args.iterations} ---")
        try:
            ttft, in_tok, out_tok, elapsed, _ = stream_request(
                args.model, args.prompt, api_key, args.base_url, args.auth_header, args.max_tokens
            )
        except Exception as e:
            print(f"  error: {e}")
            continue

        wall = out_tok / elapsed if elapsed > 0 else 0.0
        gen_time = max(elapsed - ttft, 1e-9)
        steady = out_tok / gen_time

        print(
            f"  TTFT:     {ttft * 1000:7.1f} ms\n"
            f"  Tokens:   {out_tok} (input {in_tok}) in {elapsed:.2f}s\n"
            f"  Wall:     {wall:5.1f} tok/s (incl. TTFT)\n"
            f"  Steady:   {steady:5.1f} tok/s (excl. TTFT)"
        )
        results.append(
            dict(ttft=ttft, out_tok=out_tok, elapsed=elapsed, wall=wall, steady=steady)
        )

    if not results:
        print("\nNo successful runs.")
        sys.exit(1)

    if len(results) > 1:
        print("\n=== Aggregate (median) ===")
        med_ttft = statistics.median(r["ttft"] for r in results) * 1000
        med_wall = statistics.median(r["wall"] for r in results)
        med_steady = statistics.median(r["steady"] for r in results)
        print(f"  TTFT:       {med_ttft:.1f} ms")
        print(f"  Throughput: {med_wall:.1f} tok/s (wall)")
        print(f"  Steady:     {med_steady:.1f} tok/s (excl. TTFT)")

    print()


if __name__ == "__main__":
    main()