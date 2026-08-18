from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from .images import ImageInput, normalize_image
from .types import MaskFrame, MaskPrediction


def _validate_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _validate_prompts(prompts: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(prompts, str):
        prompts = (prompts,)
    else:
        prompts = tuple(prompts)
    if not prompts:
        raise ValueError("at least one prompt is required")
    normalized = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError("every prompt must be a string")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompts must not be empty")
        normalized.append(prompt)
    return tuple(normalized)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16" and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class Sam3MaskGenerator:
    """Reusable SAM 3 image segmenter with cached vision features per image."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: Any,
        dtype: Any,
        torch_module: Any,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self._torch = torch_module

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "Sam3MaskGenerator":
        """Load a fully local Transformers SAM 3 model bundle."""

        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 masking dependencies are not installed; install "
                "requirements.sam3.txt in the dedicated masking environment"
            ) from exc

        model_dir = Path(model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"SAM 3 model directory does not exist: {model_dir}"
            )
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"SAM 3 model directory has no model.safetensors: {model_dir}"
            )

        resolved_device = cls._resolve_device(torch, device)
        resolved_dtype = cls._resolve_dtype(torch, resolved_device, dtype)
        processor = Sam3Processor.from_pretrained(str(model_dir), local_files_only=True)
        model = Sam3Model.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=resolved_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(resolved_device)
        model.eval()
        return cls(
            model,
            processor,
            device=resolved_device,
            dtype=resolved_dtype,
            torch_module=torch,
        )

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        requested = requested.lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if not (requested == "cpu" or requested.startswith("cuda")):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
        return torch.device(requested)

    @staticmethod
    def _resolve_dtype(torch: Any, device: Any, requested: str) -> Any:
        requested = requested.lower()
        allowed = {"auto", "bf16", "fp16", "fp32"}
        if requested not in allowed:
            raise ValueError(f"dtype must be one of {sorted(allowed)}")
        if requested == "auto":
            if device.type == "cpu":
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[requested]

    def segment(
        self,
        image: ImageInput,
        prompts: Union[str, Sequence[str]],
        *,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        source_id: Optional[str] = None,
        synonym_to_canonical: Optional[Mapping[str, str]] = None,
    ) -> MaskFrame:
        """Segment all instances matching each prompt in one image."""

        prompts = _validate_prompts(prompts)
        score_threshold = _validate_threshold(score_threshold, "score_threshold")
        mask_threshold = _validate_threshold(mask_threshold, "mask_threshold")
        pil_image = normalize_image(image)
        width, height = pil_image.size

        image_inputs = self.processor(images=pil_image, return_tensors="pt")
        if "pixel_values" not in image_inputs:
            raise RuntimeError("SAM 3 processor did not return pixel_values")
        original_sizes = image_inputs.get("original_sizes")
        target_sizes = (
            _as_numpy(original_sizes).tolist()
            if original_sizes is not None
            else [[height, width]]
        )
        pixel_values = image_inputs["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )

        predictions = []
        with self._torch.inference_mode():
            vision_embeds = self.model.get_vision_features(pixel_values=pixel_values)
            for prompt_index, prompt in enumerate(prompts):
                text_inputs = self.processor(text=prompt, return_tensors="pt")
                text_inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in text_inputs.items()
                }
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
                processed = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=target_sizes,
                )
                if len(processed) != 1:
                    raise RuntimeError(
                        "SAM 3 post-processing returned an unexpected batch size"
                    )
                prompt_predictions = self._convert_prompt_results(
                    processed[0],
                    prompt=prompt,
                    prompt_index=prompt_index,
                    expected_shape=(height, width),
                )
                predictions.extend(prompt_predictions)

        predictions = self._canonicalize_predictions(
            predictions, synonym_to_canonical or {}
        )
        predictions.sort(
            key=lambda prediction: (
                -prediction.score,
                -int(np.count_nonzero(prediction.mask)),
                prediction.id,
            )
        )

        return MaskFrame(
            width=width,
            height=height,
            predictions=tuple(predictions),
            source_id=source_id,
        )

    @staticmethod
    def _canonicalize_predictions(
        predictions: Sequence[MaskPrediction],
        synonym_to_canonical: Mapping[str, str],
    ) -> list[MaskPrediction]:
        canonicalized = [
            replace(
                prediction,
                prompt=synonym_to_canonical.get(prediction.prompt, prediction.prompt),
                query_prompt=prediction.prompt,
            )
            for prediction in predictions
        ]
        ranked = sorted(
            canonicalized,
            key=lambda prediction: (
                -prediction.score,
                -int(np.count_nonzero(prediction.mask)),
                prediction.id,
            ),
        )
        kept: list[MaskPrediction] = []
        for candidate in ranked:
            duplicate = False
            for prior in kept:
                if candidate.prompt.casefold() != prior.prompt.casefold():
                    continue
                intersection = int(np.count_nonzero(candidate.mask & prior.mask))
                if not intersection:
                    continue
                first_area = int(np.count_nonzero(candidate.mask))
                second_area = int(np.count_nonzero(prior.mask))
                union = first_area + second_area - intersection
                iou = intersection / max(union, 1)
                containment = intersection / max(min(first_area, second_area), 1)
                if iou >= 0.80 or containment >= 0.92:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _convert_prompt_results(
        result: Mapping[str, Any],
        *,
        prompt: str,
        prompt_index: int,
        expected_shape: tuple[int, int],
    ) -> tuple[MaskPrediction, ...]:
        missing = {"masks", "boxes", "scores"}.difference(result)
        if missing:
            raise RuntimeError(
                f"SAM 3 result is missing fields: {', '.join(sorted(missing))}"
            )
        masks = _as_numpy(result["masks"])
        boxes = _as_numpy(result["boxes"])
        scores = _as_numpy(result["scores"]).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.size == 0:
            return ()
        if masks.ndim != 3:
            raise ValueError(f"SAM 3 masks have unexpected shape {masks.shape}")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"SAM 3 boxes have unexpected shape {boxes.shape}")
        if len(masks) != len(boxes) or len(masks) != len(scores):
            raise ValueError("SAM 3 mask, box, and score counts do not agree")

        candidates = []
        for mask, box, score in zip(masks, boxes, scores):
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"SAM 3 mask has shape {mask.shape}; expected {expected_shape}"
                )
            candidates.append((float(score), tuple(float(v) for v in box), mask))
        candidates.sort(key=lambda item: item[0], reverse=True)

        return tuple(
            MaskPrediction(
                id=f"p{prompt_index:03d}-i{instance_index:03d}",
                prompt=prompt,
                score=score,
                box_xyxy=box,
                mask=mask,
            )
            for instance_index, (score, box, mask) in enumerate(candidates)
        )

    def close(self) -> None:
        """Best-effort release for direct API users; CLIs release by process exit."""

        if self.model is None:
            return
        if hasattr(self.model, "to"):
            self.model.to("cpu")
        self.model = None
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()

    def __enter__(self) -> "Sam3MaskGenerator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
