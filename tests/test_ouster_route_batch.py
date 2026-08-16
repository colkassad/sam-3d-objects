import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_masking.types import MaskFrame, MaskPrediction
from sam3_route.artifacts import ROUTE_MANIFEST_SCHEMA, atomic_write_json
from sam3_route.batch_segment import batch_segment_route


def test_importing_route_package_does_not_eagerly_load_surface_dependencies():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import sam3_route; "
                "assert 'sam3_route.surface' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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


def test_surface_batch_uses_independent_artifacts_and_preserves_literal_prompts(
    tmp_path,
):
    run_dir = tmp_path / "run"
    image_path = run_dir / "frames" / "frame-000001" / "rgb.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
    object_manifest = "frames/frame-000001/segmentation/manifest.json"
    atomic_write_json(
        run_dir / "route-manifest.json",
        {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "stages": {"extract": {"status": "complete"}},
            "keyframes": [
                {
                    "id": "frame-000001",
                    "rgb": image_path.relative_to(run_dir).as_posix(),
                    "mask_manifest": object_manifest,
                    "surface_mask_manifest": None,
                }
            ],
            "prompts": ["car"],
            "surface_prompts": [],
        },
    )

    class FakeGenerator:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def segment(self, image, prompts, **kwargs):
            assert tuple(prompts) == ("dirt track", "gravel carriageway")
            mask = np.ones((3, 4), dtype=bool)
            return MaskFrame(
                4,
                3,
                (
                    MaskPrediction(
                        id="p000-i000",
                        prompt=prompts[0],
                        score=0.9,
                        box_xyxy=(0, 0, 4, 3),
                        mask=mask,
                    ),
                ),
                source_id=kwargs["source_id"],
            )

    batch_segment_route(
        run_dir,
        model_dir=tmp_path / "model",
        prompts=["dirt track", "gravel carriageway"],
        artifact_set="surface",
        generator_factory=lambda *args, **kwargs: FakeGenerator(),
    )

    manifest = json.loads((run_dir / "route-manifest.json").read_text())
    frame = manifest["keyframes"][0]
    assert frame["mask_manifest"] == object_manifest
    assert "surface-segmentation" in frame["surface_mask_manifest"]
    assert manifest["prompts"] == ["car"]
    assert manifest["surface_prompts"] == ["dirt track", "gravel carriageway"]
