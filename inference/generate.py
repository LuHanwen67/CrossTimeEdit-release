#!/usr/bin/env python3
"""Generate CrossTimeEdit outputs from a manifest with one process per GPU."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=90)
    return parser.parse_args()


def load_pipe(args):
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    base = args.base_model
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path=[
                str(base / "text_encoder/model-00001-of-00002.safetensors"),
                str(base / "text_encoder/model-00002-of-00002.safetensors"),
            ]),
            ModelConfig(path=str(base / "transformer/diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(base / "vae/diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    pipe.load_lora(pipe.dit, str(args.lora))
    return pipe


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from PIL import Image

    rows = [
        row for row in json.loads(args.manifest.read_text(encoding="utf-8"))
        if row.get("type", "change") == "change"
    ][args.rank::args.world_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipe(args)
    started = time.time()
    completed = 0
    for index, sample in enumerate(rows, 1):
        target = args.output_dir / f"{sample['folderID']}.jpg"
        if target.exists():
            completed += 1
            continue
        source = args.data_root / sample["t2_streetview_relative_path"]
        with Image.open(source) as image:
            result = pipe(
                prompt=sample["instruction"],
                edit_image=[image.convert("RGB")],
                height=512,
                width=1024,
                num_inference_steps=args.steps,
                cfg_scale=4.0,
                embedded_guidance=3.5,
                seed=args.seed,
                rand_device="cpu",
            )
        result.save(target, quality=args.quality)
        completed += 1
        if index % 20 == 0:
            print(f"GPU {args.gpu}: {index}/{len(rows)}", flush=True)

    marker = args.output_dir / f"completed_rank{args.rank}.json"
    marker.write_text(
        json.dumps(
            {
                "count": completed,
                "gpu": args.gpu,
                "seed": args.seed,
                "steps": args.steps,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
