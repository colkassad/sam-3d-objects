import json
import os
from pathlib import Path

import pytest

from sam3_masking.mesh_bridge import build_parser, run


RUN_GPU = os.environ.get("RUN_SAM3_INTEGRATION") == "1"


@pytest.mark.gpu
@pytest.mark.skipif(not RUN_GPU, reason="set RUN_SAM3_INTEGRATION=1 to run")
def test_real_sam3_masks_feed_sam3d_mesh_generation(tmp_path):
    executable = os.environ.get("SAM3_MASK_EXECUTABLE")
    model_dir = os.environ.get("SAM3_MODEL_DIR", "checkpoints/sam3-hf")
    if not executable:
        pytest.fail("SAM3_MASK_EXECUTABLE must point to the masking-env sam3-mask")
    image = Path("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
    args = build_parser().parse_args(
        [
            "--image",
            str(image),
            "--prompts",
            os.environ.get("SAM3_TEST_PROMPT", "box"),
            "--output-dir",
            str(tmp_path),
            "--sam3-executable",
            executable,
            "--sam3-model-dir",
            model_dir,
            "--memory-profile",
            "low_vram",
            "--stage1-inference-steps",
            "1",
            "--stage2-inference-steps",
            "1",
            "--mesh-target-faces",
            "1000",
            "--profile-memory",
        ]
    )
    assert run(args) == 0

    manifest = json.loads((tmp_path / "segmentation/manifest.json").read_text())
    manifest_dir = tmp_path / "segmentation"
    assert manifest["predictions"], "SAM 3 returned no masks for the smoke prompt"
    for prediction in manifest["predictions"]:
        assert prediction["mesh"]["status"] == "ok"
        mesh_path = (manifest_dir / prediction["mesh"]["path"]).resolve()
        assert mesh_path.is_file()
        assert mesh_path.stat().st_size > 0
