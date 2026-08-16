import json
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_masking.types import MaskFrame, MaskPrediction
from sam3_route.artifacts import ROUTE_MANIFEST_SCHEMA, atomic_write_json
from sam3_route.batch_segment import batch_segment_route


def test_batch_segment_loads_model_once_and_updates_all_keyframes(tmp_path):
    run_dir = tmp_path / "run"
    keyframes = []
    for index in (1, 2):
        frame_id = f"frame-{index:06d}"
        image_path = run_dir / "frames" / frame_id / "rgb.png"
        image_path.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image_path)
        keyframes.append(
            {
                "id": frame_id,
                "scan_index": index,
                "timestamp_ns": index,
                "reference_column": 2,
                "rgb": image_path.relative_to(run_dir).as_posix(),
                "geometry": f"frames/{frame_id}/geometry.npz",
                "mask_manifest": None,
            }
        )
    atomic_write_json(
        run_dir / "route-manifest.json",
        {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "stages": {"extract": {"status": "complete"}},
            "keyframes": keyframes,
            "prompts": [],
        },
    )
    events = []

    class FakeGenerator:
        device = type("Device", (), {"type": "cpu"})()

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def segment(self, image, prompts, **kwargs):
            events.append((Path(image).name, tuple(prompts), kwargs["source_id"]))
            if kwargs["source_id"] == "frame-000002":
                return MaskFrame(5, 4, (), source_id=kwargs["source_id"])
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:4] = True
            prediction = MaskPrediction(
                id="p000-i000",
                prompt="car",
                score=0.9,
                box_xyxy=(2, 1, 4, 3),
                mask=mask,
            )
            return MaskFrame(5, 4, (prediction,), source_id=kwargs["source_id"])

    factory_calls = []

    def factory(model_dir, **kwargs):
        factory_calls.append((model_dir, kwargs))
        return FakeGenerator()

    count = batch_segment_route(
        run_dir,
        model_dir=tmp_path / "model",
        prompts=["car"],
        generator_factory=factory,
    )

    assert count == 1
    assert len(factory_calls) == 1
    assert events[0] == "enter" and events[-1] == "exit"
    manifest = json.loads((run_dir / "route-manifest.json").read_text())
    assert all(frame["mask_manifest"] for frame in manifest["keyframes"])
    second_manifest = json.loads(
        (run_dir / manifest["keyframes"][1]["mask_manifest"]).read_text()
    )
    assert second_manifest["predictions"] == []
