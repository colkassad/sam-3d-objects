import types

import pytest
import torch

from sam3d_objects.model.backbone.dit.embedder.dino import Dino
from sam3d_objects.pipeline.inference_pipeline import (
    LOW_VRAM_MAX_BYTES,
    InferencePipeline,
    StageResidencyManager,
    normalize_inference_steps,
    normalize_mesh_target_faces,
    normalize_output_formats,
    resolve_compile_model,
    resolve_memory_profile,
)


def test_output_format_validation_and_deduplication():
    assert normalize_output_formats(("mesh", "mesh", "gaussian")) == (
        "mesh",
        "gaussian",
    )
    with pytest.raises(ValueError, match="Unsupported output"):
        normalize_output_formats(("radiance_field",))


def test_low_poly_argument_validation():
    assert normalize_mesh_target_faces(None) is None
    assert normalize_mesh_target_faces(10_000) == 10_000
    assert normalize_inference_steps(15, "steps") == 15
    with pytest.raises(ValueError, match="greater than zero"):
        normalize_mesh_target_faces(0)
    with pytest.raises(TypeError, match="integer"):
        normalize_inference_steps(12.5, "steps")


def test_auto_profile_uses_total_vram(monkeypatch):
    assert resolve_memory_profile("low_vram", "cuda") == "low_vram"
    assert resolve_memory_profile("resident", "cuda") == "resident"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: types.SimpleNamespace(total_memory=LOW_VRAM_MAX_BYTES),
    )
    assert resolve_memory_profile("auto", "cuda") == "low_vram"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: types.SimpleNamespace(total_memory=LOW_VRAM_MAX_BYTES + 1),
    )
    assert resolve_memory_profile("auto", "cuda") == "resident"


def test_stage_cleanup_runs_after_failure():
    class Movable:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))

    module = Movable()
    manager = StageResidencyManager("cuda", enabled=True, profile_memory=False)
    with pytest.raises(RuntimeError, match="boom"):
        with manager.activate("test", (module,)):
            raise RuntimeError("boom")

    assert module.devices == ["cuda", "cpu"]


def test_stage_reclaims_large_unused_cuda_pool(monkeypatch):
    class Movable:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))

    cleared = []
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 64 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 3 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    module = Movable()
    manager = StageResidencyManager("cuda", enabled=True, profile_memory=False)

    with manager.activate("decoder", (module,)):
        pass

    assert module.devices == ["cuda", "cpu"]
    assert cleared == [True]


def test_stage_report_splits_transfer_compute_and_cache_timings(monkeypatch):
    class Movable:
        def to(self, _device):
            return self

    cleared = []
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 2 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 3 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 64 * 1024**2)
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda _device: 64 * 1024**2 if cleared else 3 * gib,
    )
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    manager = StageResidencyManager("cuda", enabled=True, profile_memory=True)

    with manager.activate("decoder", (Movable(),)):
        pass

    report = manager.measurements[0]
    assert report["activation_seconds"] >= 0
    assert report["compute_seconds"] >= 0
    assert report["offload_seconds"] >= 0
    assert report["cache_clear_seconds"] >= 0
    assert report["cache_cleared"] is True
    assert report["cache_clear_reason"] == "large_unused_pool"
    assert report["pre_clear_reserved_bytes"] == 3 * gib
    assert report["post_offload_reserved_bytes"] == 64 * 1024**2


def test_compile_is_disabled_only_for_low_vram():
    assert resolve_compile_model(True, "low_vram") is False
    assert resolve_compile_model(True, "resident") is True
    assert resolve_compile_model(False, "low_vram") is False


def _fake_dino(backbone, prenorm_features):
    module = Dino.__new__(Dino)
    torch.nn.Module.__init__(module)
    module.backbone = backbone
    module.prenorm_features = prenorm_features
    module.resize_input_size = (224, 224)
    module.normalize_images = True
    module.feature_cache = None
    module._preprocess_input = types.MethodType(lambda self, value: value, module)
    return module


def test_shared_dino_cache_preserves_both_token_views():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_features(self, value):
            self.calls += 1
            prenorm = value.transpose(1, 2)
            return {
                "x_norm_clstoken": prenorm[:, 0],
                "x_norm_patchtokens": prenorm[:, 2:],
                "x_prenorm": prenorm,
            }

    backbone = Backbone()
    stage_one = _fake_dino(backbone, prenorm_features=False)
    stage_two = _fake_dino(backbone, prenorm_features=True)
    cache = {}
    stage_one.feature_cache = cache
    stage_two.feature_cache = cache
    value = torch.randn(1, 4, 3)

    stage_one_tokens = stage_one(value)
    stage_two_tokens = stage_two(value)

    assert backbone.calls == 1
    assert stage_one_tokens.shape[1] == 2
    assert stage_two_tokens.shape[1] == 3


def test_conditioning_stage_projects_both_legacy_token_layouts_once():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_features(self, value):
            self.calls += 1
            prenorm = value.transpose(1, 2)
            return {
                "x_norm_clstoken": prenorm[:, 0],
                "x_norm_patchtokens": prenorm[:, 2:],
                "x_prenorm": prenorm,
            }

    class ProjectedConditioner(torch.nn.Module):
        def __init__(self, dino, scale):
            super().__init__()
            self.dino = dino
            self.scale = scale
            self.embedder_list = [(dino, None)]

        def forward(self, value):
            return self.dino(value) * self.scale

    backbone = Backbone()
    stage_one_dino = _fake_dino(backbone, prenorm_features=False)
    stage_two_dino = _fake_dino(backbone, prenorm_features=True)
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.device = torch.device("cpu")
    pipeline.shape_model_dtype = torch.float32
    pipeline.dtype = torch.float32
    pipeline.low_vram = True
    pipeline.compile_model = False
    pipeline.stage_residency = StageResidencyManager(
        "cpu", enabled=True, profile_memory=False
    )
    pipeline.condition_embedders = {
        "ss_condition_embedder": ProjectedConditioner(stage_one_dino, 2.0),
        "slat_condition_embedder": ProjectedConditioner(stage_two_dino, 3.0),
    }
    pipeline.ss_condition_input_mapping = ["image"]
    pipeline.slat_condition_input_mapping = ["image"]
    value = torch.randn(1, 4, 3)

    ss_condition, slat_condition = pipeline.prepare_conditioning(
        {"image": value}, {"image": value}
    )

    assert backbone.calls == 1
    expected_prenorm = value.transpose(1, 2)
    expected_ss = torch.cat(
        [expected_prenorm[:, :1], expected_prenorm[:, 2:]], dim=1
    ) * 2.0
    expected_slat = torch.nn.functional.layer_norm(
        expected_prenorm, expected_prenorm.shape[-1:]
    ) * 3.0
    torch.testing.assert_close(ss_condition[0][0], expected_ss)
    torch.testing.assert_close(slat_condition[0][0], expected_slat)


def test_mesh_decoder_is_loaded_without_gaussian_decoder(monkeypatch):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.models = torch.nn.ModuleDict()
    pipeline.load_device = torch.device("cpu")
    pipeline._decoder_specs = {
        "mesh": ("mesh.yaml", "mesh.ckpt"),
        "gaussian": ("gs.yaml", "gs.ckpt"),
        "gaussian_4": (None, None),
    }
    loaded = []
    mesh_decoder = torch.nn.Identity()
    monkeypatch.setattr(
        pipeline,
        "init_slat_decoder_mesh",
        lambda config, checkpoint: loaded.append("mesh") or mesh_decoder,
    )
    monkeypatch.setattr(
        pipeline,
        "init_slat_decoder_gs",
        lambda config, checkpoint: loaded.append("gaussian") or torch.nn.Identity(),
    )

    assert pipeline._get_or_load_decoder("mesh") is mesh_decoder
    assert loaded == ["mesh"]


def test_returned_mesh_and_metadata_are_moved_to_cpu():
    class MovableMesh:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.low_vram = False
    mesh = MovableMesh()
    result = pipeline._move_outputs_to_cpu(
        {"mesh": [mesh], "pose": {"translation": torch.ones(3)}}
    )

    assert mesh.moves == ["cpu"]
    assert result["pose"]["translation"].device.type == "cpu"


def test_low_vram_moves_optional_decoder_outputs_to_cpu():
    class MovableGaussian:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(str(device))
            return self

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.low_vram = True
    gaussian = MovableGaussian()
    pipeline._move_outputs_to_cpu({"gaussian": [gaussian]})

    assert gaussian.moves == ["cpu"]


def test_low_vram_discards_dense_stage_one_latent():
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.low_vram = True
    outputs = {
        "shape": torch.randn(1, 4096, 8),
        "coords": torch.ones(4, 4, dtype=torch.int32),
        "translation": torch.ones(3),
    }

    compact = pipeline._compact_sparse_outputs(outputs)

    assert "shape" not in compact
    assert compact["coords"].device.type == "cpu"
