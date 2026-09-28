#!/usr/bin/env python3
"""Evaluate IA, BP, and QP for one directory of generated street views."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "prompts/evaluation/streetview_vlm_protocol.py",
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_protocol(path: Path):
    spec = importlib.util.spec_from_file_location("streetview_vlm_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_image(directory: Path, sample_id: str):
    for suffix in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def encode_image(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else None


def extract_score(parsed, dimension):
    if not isinstance(parsed, dict):
        return None
    if dimension == "Instruction_Alignment":
        return parsed.get("score")
    nested = parsed.get(dimension)
    return nested.get("score") if isinstance(nested, dict) else None


def request_score(args, prompt, image, dimension, api_key, base_url):
    url = f"{base_url.rstrip('/')}/v1beta/models/{args.model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": encode_image(image)}},
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
    }
    last_error = "unknown"
    for round_index in range(1, args.rounds + 1):
        for attempt in range(1, args.retries + 1):
            try:
                response = httpx.post(
                    url,
                    json=body,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    timeout=args.timeout,
                )
                response.raise_for_status()
                parts = (response.json().get("candidates") or [{}])[0].get(
                    "content", {}
                ).get("parts", [])
                parsed = parse_json("".join(part.get("text", "") for part in parts))
                score = extract_score(parsed, dimension)
                if isinstance(score, (int, float)) and 0 <= float(score) <= 10:
                    return float(score), "ok", round_index, attempt
                raise ValueError("invalid score response")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2 ** (attempt - 1) + random.random())
    return -1.0, last_error, args.rounds, args.retries


def load_existing(path: Path):
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["folderID"]] = record
    return records


def complete(record, dimensions):
    return bool(record) and all(record.get(f"{name}_status") == "ok" for name, _, _ in dimensions)


def main():
    args = parse_args()
    random.seed(args.seed)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before evaluation")
    base_url = os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
    )
    protocol = load_protocol(args.protocol)
    dimensions = (
        ("Instruction_Alignment", protocol.PROMPT_IA_WITH_GT, "ia"),
        ("Background_Preservation", protocol.PROMPT_BP, "pair"),
        ("Image_Quality_and_Physics", protocol.PROMPT_QP, "pair"),
    )
    rows = [
        row for row in json.loads(args.manifest.read_text(encoding="utf-8"))
        if row.get("type", "change") == "change"
    ]
    existing = load_existing(args.output)
    lock = threading.Lock()

    def evaluate(row):
        sample_id = row["folderID"]
        record = dict(existing.get(sample_id, {}))
        if complete(record, dimensions):
            return record
        generated_path = resolve_image(args.image_dir, sample_id)
        if generated_path is None:
            raise FileNotFoundError(f"Missing generated image: {sample_id}")
        with Image.open(args.data_root / row["t2_streetview_relative_path"]) as image:
            input_image = image.convert("RGB")
        with Image.open(args.data_root / row["t1_streetview_relative_path"]) as image:
            target_image = image.convert("RGB")
        with Image.open(generated_path) as image:
            output_image = image.convert("RGB")
        pair = protocol.create_pair_comparison(input_image, output_image)
        ia = protocol.create_ia_gt_comparison(input_image, target_image, output_image)
        record.update({"folderID": sample_id, "model": args.name})
        for dimension, template, mode in dimensions:
            if record.get(f"{dimension}_status") == "ok":
                continue
            prompt = template.format(instruction=row["instruction"])
            score, status, rounds, attempts = request_score(
                args, prompt, ia if mode == "ia" else pair, dimension, api_key, base_url
            )
            record[f"{dimension}_score"] = score
            record[f"{dimension}_status"] = status
            record[f"{dimension}_rounds"] = rounds
            record[f"{dimension}_attempts"] = attempts
        with lock:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    pending = [row for row in rows if not complete(existing.get(row["folderID"]), dimensions)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 20 == 0 or index == len(futures):
                print(f"completed={index}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
