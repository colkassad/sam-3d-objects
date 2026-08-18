from types import SimpleNamespace

import numpy as np

from sam3_masking.artifacts import read_manifest_document, write_mask_manifest
from sam3_masking.mesh_bridge import build_parser, reconstruct_manifest, run
from sam3_masking.types import MaskFrame, MaskPrediction


def make_frame(count=2):
    predictions = []
    for index in range(count):
        mask = np.zeros((4, 5), dtype=np.bool_)
        mask[index : index + 2, index : index + 2] = True
        predictions.append(
            MaskPrediction(
                id=f"p000-i{index:03d}",
                prompt="car",
                score=0.9 - index * 0.1,
                box_xyxy=(index, index, index + 2, index + 2),
                mask=mask,
            )
        )
    return MaskFrame(width=5, height=4, predictions=tuple(predictions))


class FakeGlb:
    def export(self, path):
        from pathlib import Path

        Path(path).write_bytes(b"glTF")


def test_reconstruct_manifest_reuses_one_inference_instance(tmp_path):
    manifest = write_mask_manifest(make_frame(), tmp_path / "segmentation")
    events = []

    class FakeInference:
        def __init__(self, config, **kwargs):
            events.append(("init", config, kwargs))

        def __call__(self, image, mask, **kwargs):
            events.append(("infer", int(mask.sum()), kwargs))
            return {"glb": FakeGlb()}

    failures = reconstruct_manifest(
        manifest,
        image_path=tmp_path / "image.png",
        output_dir=tmp_path,
        sam3d_config=tmp_path / "pipeline.yaml",
        inference_factory=FakeInference,
        image_loader=lambda _: np.zeros((4, 5, 3), dtype=np.uint8),
    )

    assert failures == 0
    assert [event[0] for event in events] == ["init", "infer", "infer"]
    document = read_manifest_document(manifest)
    assert all(record["mesh"]["status"] == "ok" for record in document["predictions"])
    assert all(
        record["mesh"]["path"].startswith("../meshes/")
        for record in document["predictions"]
    )


def test_reconstruct_manifest_records_partial_failures(tmp_path):
    manifest = write_mask_manifest(make_frame(), tmp_path / "segmentation")

    class FailingInference:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("bad mesh")
            return {"glb": FakeGlb()}

    failures = reconstruct_manifest(
        manifest,
        image_path=tmp_path / "image.png",
        output_dir=tmp_path,
        sam3d_config=tmp_path / "pipeline.yaml",
        inference_factory=FailingInference,
        image_loader=lambda _: np.zeros((4, 5, 3), dtype=np.uint8),
    )
    assert failures == 1
    statuses = [
        item["mesh"]["status"]
        for item in read_manifest_document(manifest)["predictions"]
    ]
    assert statuses == ["failed", "ok"]


def test_bridge_waits_for_segmenter_before_reconstruction(tmp_path, monkeypatch):
    events = []
    output_dir = tmp_path / "output"
    repo_root = tmp_path / "repo"
    (repo_root / "sam3d_objects").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")

    def fake_subprocess(command, check):
        model_argument = command[command.index("--model-dir") + 1]
        assert model_argument == str((repo_root / "relative-model").resolve())
        image_argument = command[command.index("--image") + 1]
        assert image_argument == str(image_path.resolve())
        events.extend(("segment-start", "segment-exit"))
        write_mask_manifest(
            make_frame(0), output_dir / "segmentation", image_path=image_path
        )
        return SimpleNamespace(returncode=0)

    def fake_reconstruct(*args, **kwargs):
        assert events[-1] == "segment-exit"
        events.append("reconstruct")
        return 0

    monkeypatch.setattr(
        "sam3_masking.mesh_bridge.reconstruct_manifest", fake_reconstruct
    )
    args = build_parser().parse_args(
        [
            "--image",
            str(image_path),
            "--prompts",
            "car",
            "--output-dir",
            str(output_dir),
            "--sam3-model-dir",
            "relative-model",
            "--repo-root",
            str(repo_root),
        ]
    )
    assert run(args, subprocess_run=fake_subprocess) == 0
    assert events == ["segment-start", "segment-exit", "reconstruct"]
