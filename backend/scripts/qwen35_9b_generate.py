#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local Qwen3.5-9B MLX-VLM prompt with thinking disabled."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-MLX-4bit",
        help="MLX-VLM model repo or local path.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system message.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Lower this if you hit unified-memory pressure.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode. Disabled by default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model and processor.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from mlx_vlm import load
        from mlx_vlm.generate import generate
    except ImportError as exc:
        raise SystemExit(
            'Missing dependencies. Install with: python3 -m pip install -U "mlx-vlm[torch]"'
        ) from exc

    model, processor = load(args.model, trust_remote_code=args.trust_remote_code)

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )

    generate_kwargs: dict[str, object] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "verbose": False,
    }
    if args.prefill_step_size is not None:
        generate_kwargs["prefill_step_size"] = args.prefill_step_size

    result = generate(
        model,
        processor,
        prompt,
        **generate_kwargs,
    )

    sys.stdout.write(result.text)
    if not result.text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
