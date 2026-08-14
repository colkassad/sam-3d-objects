from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sam3_masking.generator import Sam3MaskGenerator, _as_numpy


class FakeModel:
    def __init__(self):
        self.vision_calls = 0
        self.prompt_calls = []

    def get_vision_features(self, *, pixel_values):
        self.vision_calls += 1
        assert pixel_values.dtype == torch.float32
        return torch.ones((1, 2), dtype=torch.float32)

    def __call__(self, *, vision_embeds, input_ids):
        assert vision_embeds.shape == (1, 2)
        prompt_id = int(input_ids.item())
        self.prompt_calls.append(prompt_id)
        return SimpleNamespace(prompt_id=prompt_id)


class FakeProcessor:
    def __init__(self):
        self.prompt_ids = {"parked car": 0, "traffic cone": 1}
        self.target_sizes = []

    def __call__(self, *, images=None, text=None, return_tensors=None):
        assert return_tensors == "pt"
        if images is not None:
            return {
                "pixel_values": torch.zeros((1, 3, 8, 8)),
                "original_sizes": torch.tensor([[images.height, images.width]]),
            }
        return {"input_ids": torch.tensor([[self.prompt_ids[text]]])}

    def post_process_instance_segmentation(
        self, outputs, *, threshold, mask_threshold, target_sizes
    ):
        assert threshold == 0.5
        assert mask_threshold == 0.5
        self.target_sizes.append(target_sizes)
        if outputs.prompt_id == 1:
            return [
                {
                    "masks": torch.zeros((0, 4, 5), dtype=torch.bool),
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                }
            ]
        first = torch.zeros((4, 5), dtype=torch.bool)
        first[0:2, 0:2] = True
        second = torch.zeros((4, 5), dtype=torch.bool)
        second[2:4, 3:5] = True
        return [
            {
                "masks": torch.stack((first, second)),
                "boxes": torch.tensor([[0, 0, 2, 2], [3, 2, 5, 4]]),
                "scores": torch.tensor([0.25, 0.9]),
            }
        ]


def test_as_numpy_promotes_bfloat16_tensors_to_float32():
    tensor = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)

    converted = _as_numpy(tensor)

    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, [0.25, 0.75])


def test_multi_prompt_inference_encodes_image_once_and_sorts_instances():
    model = FakeModel()
    processor = FakeProcessor()
    generator = Sam3MaskGenerator(
        model,
        processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    image = Image.new("RGBA", (5, 4), (10, 20, 30, 40))
    frame = generator.segment(
        image, ["parked car", "traffic cone"], source_id="frame-1"
    )

    assert model.vision_calls == 1
    assert model.prompt_calls == [0, 1]
    assert processor.target_sizes == [[[4, 5]], [[4, 5]]]
    assert [item.score for item in frame.predictions] == pytest.approx([0.9, 0.25])
    assert [item.id for item in frame.predictions] == ["p000-i000", "p000-i001"]
    assert all(item.mask.shape == (4, 5) for item in frame.predictions)
    assert all(item.mask.dtype == np.bool_ for item in frame.predictions)
    assert frame.source_id == "frame-1"


@pytest.mark.parametrize("prompts", [[], [""], ["   "]])
def test_segment_rejects_empty_prompts(prompts):
    generator = Sam3MaskGenerator(
        FakeModel(),
        FakeProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    with pytest.raises(ValueError):
        generator.segment(Image.new("RGB", (5, 4)), prompts)


def test_segment_rejects_invalid_thresholds():
    generator = Sam3MaskGenerator(
        FakeModel(),
        FakeProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    with pytest.raises(ValueError, match="score_threshold"):
        generator.segment(Image.new("RGB", (5, 4)), "parked car", score_threshold=2)


def test_results_are_sorted_globally_across_prompts():
    class MultiPromptProcessor(FakeProcessor):
        def post_process_instance_segmentation(
            self, outputs, *, threshold, mask_threshold, target_sizes
        ):
            if outputs.prompt_id == 0:
                return super().post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=target_sizes,
                )
            self.target_sizes.append(target_sizes)
            mask = torch.zeros((4, 5), dtype=torch.bool)
            mask[1:3, 1:4] = True
            return [
                {
                    "masks": mask.unsqueeze(0),
                    "boxes": torch.tensor([[1, 1, 4, 3]]),
                    "scores": torch.tensor([0.95]),
                }
            ]

    generator = Sam3MaskGenerator(
        FakeModel(),
        MultiPromptProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        torch_module=torch,
    )
    frame = generator.segment(Image.new("RGB", (5, 4)), ["parked car", "traffic cone"])
    assert [prediction.prompt for prediction in frame.predictions] == [
        "traffic cone",
        "parked car",
        "parked car",
    ]
