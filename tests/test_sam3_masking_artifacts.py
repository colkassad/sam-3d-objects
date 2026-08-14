import json

import numpy as np
import pytest
from PIL import Image

from sam3_masking.artifacts import load_mask_manifest, write_mask_manifest
from sam3_masking.images import normalize_image
from sam3_masking.types import MaskFrame, MaskPrediction


def make_prediction(prediction_id="p000-i000", value=True):
    mask = np.zeros((4, 5), dtype=np.bool_)
    if value:
        mask[1:3, 2:4] = True
    return MaskPrediction(
        id=prediction_id,
        prompt="parked car",
        score=0.875,
        box_xyxy=(2.0, 1.0, 4.0, 3.0),
        mask=mask,
    )


def test_normalize_image_accepts_grayscale_rgba_and_float_arrays(tmp_path):
    grayscale = normalize_image(np.zeros((3, 4), dtype=np.uint16))
    rgba = normalize_image(np.zeros((3, 4, 4), dtype=np.uint8))
    floats = normalize_image(np.full((3, 4, 3), 0.5, dtype=np.float32))

    assert grayscale.mode == rgba.mode == floats.mode == "RGB"
    assert grayscale.size == rgba.size == floats.size == (4, 3)
    assert np.asarray(floats)[0, 0, 0] == 128

    path = tmp_path / "image.png"
    Image.new("L", (6, 2), 255).save(path)
    assert normalize_image(path).size == (6, 2)


def test_normalize_image_rejects_invalid_arrays():
    with pytest.raises(ValueError, match="channel count"):
        normalize_image(np.zeros((3, 4, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_image(np.full((3, 4), np.nan, dtype=np.float32))


def test_normalize_image_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (3, 5), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)

    assert normalize_image(path).size == (5, 3)


def test_manifest_round_trip_preserves_boolean_mask_and_metadata(tmp_path):
    frame = MaskFrame(
        width=5,
        height=4,
        predictions=(make_prediction(),),
        source_id="frame-42",
    )
    manifest_path = write_mask_manifest(
        frame, tmp_path / "result", image_path=tmp_path / "source.png"
    )
    loaded = load_mask_manifest(manifest_path)

    assert loaded.width == 5
    assert loaded.height == 4
    assert loaded.source_id == "frame-42"
    assert loaded.predictions[0].mask.dtype == np.bool_
    np.testing.assert_array_equal(loaded.predictions[0].mask, frame.predictions[0].mask)
    document = json.loads(manifest_path.read_text())
    assert document["schema"] == "sam3-mask-manifest/v1"
    assert document["predictions"][0]["score"] == 0.875


def test_manifest_rejects_mask_path_traversal(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    document = {
        "schema": "sam3-mask-manifest/v1",
        "image": {"width": 5, "height": 4, "source_id": None, "path": None},
        "predictions": [
            {
                "id": "bad",
                "prompt": "car",
                "score": 0.8,
                "box_xyxy": [0, 0, 2, 2],
                "mask": "../outside.png",
            }
        ],
    }
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="escapes"):
        load_mask_manifest(manifest_path)


def test_frame_rejects_empty_or_wrong_sized_masks():
    with pytest.raises(ValueError, match="foreground"):
        make_prediction(value=False)
    prediction = make_prediction()
    with pytest.raises(ValueError, match="expected"):
        MaskFrame(width=7, height=4, predictions=(prediction,))


def test_prediction_rejects_unsafe_id():
    with pytest.raises(ValueError, match="id must start"):
        make_prediction("../../outside")
