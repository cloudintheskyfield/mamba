#!/usr/bin/env python3
"""Run text generation with a local or Hub-hosted Mamba HF checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a Mamba HF checkpoint")
    parser.add_argument(
        "--model-path",
        type=str,
        default="state-spaces/mamba-2.8b-hf",
        help="Local model directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="My cat wrote all this CUDA code for a new language model and",
        help="Prompt to continue",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default="float16",
        help="Torch dtype used to load the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single device to place the model on when --device-map is disabled",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help='Optional device map, e.g. "auto". When set, transformers handles placement.',
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save prompt, output, and timing metadata as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = DTYPE_MAP[args.dtype]
    load_kwargs = {"torch_dtype": torch_dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if args.device_map is None:
        model = model.to(args.device)
    model.eval()
    load_seconds = time.time() - start

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if args.device_map is None:
        inputs = {key: value.to(args.device) for key, value in inputs.items()}

    with torch.inference_mode():
        gen_start = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
        generate_seconds = time.time() - gen_start

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    result = {
        "model_path": args.model_path,
        "device": args.device,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": round(load_seconds, 3),
        "generate_seconds": round(generate_seconds, 3),
        "prompt": args.prompt,
        "generated_text": generated_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
