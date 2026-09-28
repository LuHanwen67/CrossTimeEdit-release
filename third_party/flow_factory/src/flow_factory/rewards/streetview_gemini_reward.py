"""Pointwise Gemini rewards for the third Streetview Flow-GRPO run.

The three dimensions are intentionally scored with independent API calls.
Prompt text is loaded from the existing evaluation scripts so training and
offline evaluation do not silently diverge.
"""
from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List

import httpx
import torch
from PIL import Image, ImageDraw, ImageFont

from .abc import PointwiseRewardModel, RewardModelOutput

LOGGER = logging.getLogger(__name__)
_PROMPT_CACHE = {}
_PROMPT_LOCK = threading.Lock()


def _load_module(path: str, name: str):
    key = (path, name)
    with _PROMPT_LOCK:
        if key in _PROMPT_CACHE:
            return _PROMPT_CACHE[key]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load prompt source: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PROMPT_CACHE[key] = module
        return module


def _pil_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _font(size: int):
    for path in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _label(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    f = _font(max(14, image.height // 28))
    box = draw.textbbox((0, 0), text, font=f)
    draw.rectangle((0, 0, box[2] - box[0] + 14, box[3] - box[1] + 10), fill=(0, 0, 0))
    draw.text((7, 5), text, fill=(255, 255, 255), font=f)
    return image


class StreetviewGeminiRewardModel(PointwiseRewardModel):
    """One independent Gemini score per generated image.

    ``dimension`` is one of ``ia_gt``, ``bp`` or ``qp``.  For ``ia_gt`` the
    JSON metadata must contain a valid ``gt_image`` path.
    """

    required_fields = ("prompt", "image", "condition_images", "metadata")
    use_tensor_inputs = False

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        extra = getattr(config, "extra_kwargs", {}) or {}
        self.dimension = str(extra.get("dimension", "bp")).lower()
        if self.dimension not in {"ia_gt", "bp", "qp"}:
            raise ValueError(f"Unsupported Gemini reward dimension: {self.dimension}")
        self.model_name = extra.get("model", "gemini-3.1-flash-lite")
        self.prompt_root = Path(extra.get("prompt_root", "prompts/evaluation"))
        self.base_url = extra.get(
            "base_url",
            os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
        ).rstrip("/")
        self.api_key = extra.get("api_key", os.environ.get("GEMINI_API_KEY"))
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini reward")
        self.timeout = float(extra.get("timeout", 180))
        # ``retries`` counts retries after the initial request.  The default
        # therefore allows up to four attempts before the neutral score 5.0.
        self.retries = int(extra.get("retries", 3))
        self.max_output_tokens = int(extra.get("max_output_tokens", 256))
        self._calls = 0; self._errors = 0; self._max_tokens = 0; self._prompt_tokens = 0; self._candidate_tokens = 0
        self._samples = 0; self._successes = 0; self._final_failures = 0; self._retry_requests = 0
        self._lock = threading.Lock()

    def _prompt(self, instruction: str) -> str:
        protocol = _load_module(str(self.prompt_root / "streetview_vlm_protocol.py"), "streetview_vlm_protocol")
        if self.dimension == "ia_gt":
            return protocol.PROMPT_IA_WITH_GT.format(instruction=instruction)
        prompt = protocol.PROMPT_BP if self.dimension == "bp" else protocol.PROMPT_QP
        return prompt.format(instruction=instruction)

    def _request(self, image: Image.Image, prompt: str) -> tuple[float, dict]:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        url = f"{root}/v1beta/models/{self.model_name}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": __import__('base64').b64encode(_pil_bytes(image)).decode()}},
            ]}],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens, "temperature": 0},
        }
        last = "unknown"
        for attempt in range(self.retries + 1):
            if attempt:
                with self._lock:
                    self._retry_requests += 1
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=body, headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
                data = response.json()
                candidates = data.get("candidates") or []
                usage = data.get("usageMetadata") or {}
                finish = candidates[0].get("finishReason") if candidates else None
                text = "".join(p.get("text", "") for p in (candidates[0].get("content", {}).get("parts", []) if candidates else []))
                with self._lock:
                    self._calls += 1
                    self._prompt_tokens += int(usage.get("promptTokenCount", 0) or 0)
                    self._candidate_tokens += int(usage.get("candidatesTokenCount", 0) or 0)
                    if finish == "MAX_TOKENS": self._max_tokens += 1
                if response.status_code != 200 or not text:
                    last = f"http={response.status_code},finish={finish}"; raise RuntimeError(last)
                try:
                    parsed = self._parse(text)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    last = f"parse_error={exc}"
                    if attempt < self.retries:
                        time.sleep(2 ** attempt)
                        continue
                    with self._lock:
                        self._errors += 1
                    LOGGER.warning("Gemini score parse failed after %d retries dimension=%s: %s; using neutral 5.0", self.retries, self.dimension, exc)
                    return 5.0, {"finish": finish, "usage": usage, "parse_error": str(exc), "final_failure": True}
                with self._lock:
                    self._successes += 1
                return float(parsed), {"finish": finish, "usage": usage}
            except Exception as exc:
                last = str(exc)
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        with self._lock:
            self._errors += 1
        LOGGER.warning("Gemini request failed after %d retries dimension=%s: %s; using neutral 5.0", self.retries, self.dimension, last)
        return 5.0, {"error": last, "final_failure": True}

    @staticmethod
    def _parse(text: str) -> float:
        text = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start: raise ValueError("JSON parse failed")
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("JSON root is not an object")

        # IA-with-GT returns a top-level score, while BP/QP return a
        # dimension-named object containing score.  Accept both forms without
        # depending on the exact dimension label used by the prompt.
        score = data.get("score")
        if score is None:
            candidates = []
            def collect_scores(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if str(key).strip().lower() == "score":
                            candidates.append(item)
                        else:
                            collect_scores(item)
                elif isinstance(value, list):
                    for item in value:
                        collect_scores(item)
            collect_scores(data)
            if len(candidates) == 1:
                score = candidates[0]
            elif not candidates:
                raise ValueError("score field not found")
            else:
                raise ValueError("multiple score fields found")
        if isinstance(score, dict): score = score.get("score")
        if isinstance(score, list): score = score[0]
        score = float(score)
        if not 0 <= score <= 10: raise ValueError(f"score out of range: {score}")
        return score

    def __call__(self, prompt: List[str], image=None, condition_images=None, metadata=None, **kwargs):
        scores=[]
        for i, (instruction, generated) in enumerate(zip(prompt, image or [])):
            try:
                original = condition_images[i][0] if condition_images and condition_images[i] else generated
                protocol = _load_module(str(self.prompt_root / "streetview_vlm_protocol.py"), "streetview_vlm_protocol")
                if self.dimension == "ia_gt":
                    raw_meta = metadata[i] if metadata and i < len(metadata) else {}
                    if isinstance(raw_meta, str):
                        try:
                            raw_meta = json.loads(raw_meta)
                        except json.JSONDecodeError:
                            raw_meta = {}
                    meta = raw_meta if isinstance(raw_meta, dict) else {}
                    gt_path = meta.get("gt_image")
                    extra_gt = kwargs.get("gt_image")
                    if not gt_path and isinstance(extra_gt, (list, tuple)) and i < len(extra_gt):
                        gt_path = extra_gt[i]
                    if not gt_path or not Path(gt_path).is_file():
                        raise FileNotFoundError(f"missing gt_image for sample {i}: {gt_path}")
                    with Image.open(gt_path) as gt:
                        view = protocol.create_ia_gt_comparison(original, gt, generated)
                else:
                    view = protocol.create_pair_comparison(original, generated)
                score, info = self._request(view, self._prompt(instruction))
                scores.append(score)
                with self._lock:
                    self._samples += 1
                    if info.get("final_failure"):
                        self._final_failures += 1
                    samples = self._samples
                    successes = self._successes
                    failures = self._final_failures
                    retries = self._retry_requests
                if samples % 100 == 0:
                    LOGGER.info("Gemini reward stats dimension=%s samples=%d success=%d final_fail=%d success_rate=%.2f%% retries=%d", self.dimension, samples, successes, failures, 100.0 * successes / max(samples, 1), retries)
            except Exception as exc:
                # Neutral fallback keeps a transient API failure from creating
                # a deterministic preference.  The error is explicitly logged.
                LOGGER.warning("Gemini reward failed dimension=%s index=%d: %s", self.dimension, i, exc)
                scores.append(5.0)
                with self._lock:
                    self._samples += 1
                    self._final_failures += 1
        return RewardModelOutput(rewards=torch.tensor(scores, dtype=torch.float32, device=self.accelerator.device), extra_info={})
