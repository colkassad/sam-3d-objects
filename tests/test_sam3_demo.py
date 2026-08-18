import json
from pathlib import Path

import pytest

from scripts import demo_prompt_to_mesh as demo


def make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def make_inputs(tmp_path, monkeypatch):
    image = tmp_path / "road scene.jpg"
    image.write_bytes(b"image")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    config = tmp_path / "pipeline.yaml"
    config.write_text("config")
    executable = make_executable(tmp_path / "bin" / "sam3-mask")
    monkeypatch.setenv("SAM3_MASK_EXECUTABLE", str(executable))
    return image, model, config, executable


def test_discover_executable_precedence(tmp_path, monkeypatch):
    explicit = make_executable(tmp_path / "explicit" / "sam3-mask")
    configured = make_executable(tmp_path / "configured" / "sam3-mask")
    sibling = make_executable(tmp_path / "envs/sam3-masking/bin/sam3-mask")
    path_executable = make_executable(tmp_path / "path" / "sam3-mask")
    monkeypatch.setenv("SAM3_MASK_EXECUTABLE", str(configured))
    monkeypatch.setenv("PATH", str(path_executable.parent))

    assert (
        demo.discover_sam3_executable(
            str(explicit), prefix=tmp_path / "envs/sam3d-objects"
        )
        == explicit.resolve()
    )
    assert (
        demo.discover_sam3_executable(None, prefix=tmp_path / "envs/sam3d-objects")
        == configured.resolve()
    )
    monkeypatch.delenv("SAM3_MASK_EXECUTABLE")
    assert (
        demo.discover_sam3_executable(None, prefix=tmp_path / "envs/sam3d-objects")
        == sibling.resolve()
    )
    sibling.unlink()
    assert (
        demo.discover_sam3_executable(None, prefix=tmp_path / "envs/sam3d-objects")
        == path_executable.resolve()
    )


def test_discover_executable_reports_setup_action(tmp_path, monkeypatch):
    monkeypatch.delenv("SAM3_MASK_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="create the sam3-masking environment"):
        demo.discover_sam3_executable(None, prefix=tmp_path / "envs/sam3d-objects")


def test_demo_forwards_prompts_defaults_and_mesh_options(tmp_path, monkeypatch, capsys):
    image, model, config, executable = make_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(demo, "REPO_ROOT", tmp_path)
    captured = {}

    def fake_bridge(args):
        captured["args"] = args
        manifest = args.output_dir / "segmentation" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        mesh = args.output_dir / "meshes" / "car.glb"
        mesh.parent.mkdir()
        mesh.write_bytes(b"glTF")
        manifest.write_text(
            json.dumps(
                {
                    "schema": "sam3-mask-manifest/v1",
                    "image": {"width": 1, "height": 1},
                    "predictions": [
                        {
                            "id": "p000-i000",
                            "prompt": "parked car",
                            "score": 0.9,
                            "box_xyxy": [0, 0, 1, 1],
                            "mask": "masks/car.png",
                            "mesh": {"status": "ok", "path": "../meshes/car.glb"},
                        }
                    ],
                }
            )
        )
        return 0

    monkeypatch.setattr(demo, "run_bridge", fake_bridge)
    args = demo.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--prompts",
            " parked car , traffic cone ",
            "--sam3-model-dir",
            str(model),
            "--sam3d-config",
            str(config),
            "--mesh-target-faces",
            "5000",
            "--flat-shading",
            "--stage1-inference-steps",
            "12",
            "--stage2-inference-steps",
            "8",
            "--profile-memory",
        ]
    )

    assert demo.run(args) == 0
    forwarded = captured["args"]
    assert forwarded.prompts == "parked car,traffic cone"
    assert forwarded.sam3_executable == str(executable.resolve())
    assert forwarded.output_dir == tmp_path / "outputs/sam3-demo/road-scene"
    assert forwarded.score_threshold == forwarded.mask_threshold == 0.5
    assert forwarded.seed == 42
    assert forwarded.memory_profile == "low_vram"
    assert forwarded.mesh_target_faces == 5000
    assert forwarded.flat_shading is True
    assert forwarded.stage1_inference_steps == 12
    assert forwarded.stage2_inference_steps == 8
    assert forwarded.profile_memory is True
    output = capsys.readouterr().out
    assert "Detections: 1" in output
    assert "1 succeeded, 0 failed" in output
    assert str(forwarded.output_dir / "meshes/car.glb") in output


@pytest.mark.parametrize("return_code", [1, 7])
def test_demo_preserves_failure_status_and_summarizes_new_manifest(
    tmp_path, monkeypatch, capsys, return_code
):
    image, model, config, _ = make_inputs(tmp_path, monkeypatch)
    output_dir = tmp_path / "output"

    def fake_bridge(args):
        if return_code == 1:
            manifest = args.output_dir / "segmentation" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "sam3-mask-manifest/v1",
                        "image": {"width": 1, "height": 1},
                        "predictions": [
                            {
                                "id": "p000-i000",
                                "prompt": "car",
                                "score": 0.8,
                                "box_xyxy": [0, 0, 1, 1],
                                "mask": "masks/car.png",
                                "mesh": {"status": "failed", "error": "failure"},
                            }
                        ],
                    }
                )
            )
        return return_code

    monkeypatch.setattr(demo, "run_bridge", fake_bridge)
    args = demo.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--prompts",
            "car",
            "--output-dir",
            str(output_dir),
            "--sam3-model-dir",
            str(model),
            "--sam3d-config",
            str(config),
        ]
    )
    assert demo.run(args) == return_code
    captured = capsys.readouterr()
    if return_code == 1:
        assert "0 succeeded, 1 failed" in captured.out
    else:
        assert "failed before producing a new manifest" in captured.err


def test_demo_zero_detections_succeeds(tmp_path, monkeypatch, capsys):
    image, model, config, _ = make_inputs(tmp_path, monkeypatch)
    output_dir = tmp_path / "empty"

    def fake_bridge(args):
        manifest = args.output_dir / "segmentation" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema": "sam3-mask-manifest/v1",
                    "image": {"width": 1, "height": 1},
                    "predictions": [],
                }
            )
        )
        return 0

    monkeypatch.setattr(demo, "run_bridge", fake_bridge)
    args = demo.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--prompts",
            "car",
            "--output-dir",
            str(output_dir),
            "--sam3-model-dir",
            str(model),
            "--sam3d-config",
            str(config),
        ]
    )
    assert demo.run(args) == 0
    assert "Detections: 0" in capsys.readouterr().out


def test_help_smoke():
    help_text = demo.build_parser().format_help()
    assert "--image" in help_text
    assert "--prompts" in help_text
    assert "--prompt " not in help_text
    assert "--mesh-target-faces" in help_text


@pytest.mark.parametrize("argv", [[], ["--image", "image.png"]])
def test_parser_requires_image_and_prompt(argv):
    with pytest.raises(SystemExit) as exc_info:
        demo.build_parser().parse_args(argv)
    assert exc_info.value.code == 2
