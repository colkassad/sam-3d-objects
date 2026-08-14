# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import time
from contextlib import contextmanager
from numbers import Integral
from typing import Iterable, List, Literal, Sequence, Union

from tqdm import tqdm
import torch
from loguru import logger
from functools import wraps
from torch.utils._pytree import tree_map_only


def set_attention_backend():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    logger.info(f"GPU name is {gpu_name}")
    logger.info(
        "Attention backend: {} (PyTorch SDPA performs capability-based CUDA dispatch)",
        os.environ.get("ATTN_BACKEND", "sdpa"),
    )


set_attention_backend()

from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from PIL import Image
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)

from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
from sam3d_objects.model.backbone.dit.embedder.dino import Dino
from safetensors.torch import load_file


MemoryProfile = Literal["auto", "low_vram", "resident"]
OutputFormat = Literal["mesh", "gaussian", "gaussian_4"]
SUPPORTED_OUTPUT_FORMATS = frozenset({"mesh", "gaussian", "gaussian_4"})
LOW_VRAM_MAX_BYTES = 17 * 1024**3
CACHE_CLEAR_MIN_RECLAIMABLE_BYTES = 2 * 1024**3
CACHE_CLEAR_MIN_INACTIVE_SPLIT_BYTES = 512 * 1024**2


def resolve_memory_profile(
    memory_profile: MemoryProfile,
    device: Union[str, torch.device] = "cuda",
) -> Literal["low_vram", "resident"]:
    """Resolve ``auto`` without allocating CUDA memory."""
    if memory_profile not in {"auto", "low_vram", "resident"}:
        raise ValueError(
            f"Unknown memory profile {memory_profile!r}; expected auto, low_vram, or resident"
        )
    if memory_profile != "auto":
        return memory_profile

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "resident"
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    return "low_vram" if total_memory <= LOW_VRAM_MAX_BYTES else "resident"


def normalize_output_formats(formats: Sequence[str]) -> tuple[OutputFormat, ...]:
    formats = tuple(dict.fromkeys(formats))
    unknown = set(formats) - SUPPORTED_OUTPUT_FORMATS
    if unknown:
        raise ValueError(
            f"Unsupported output formats {sorted(unknown)}; expected values from "
            f"{sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def normalize_mesh_target_faces(target_faces):
    if target_faces is None:
        return None
    if isinstance(target_faces, bool) or not isinstance(target_faces, Integral):
        raise TypeError("mesh_target_faces must be an integer or None")
    if target_faces <= 0:
        raise ValueError("mesh_target_faces must be greater than zero")
    return int(target_faces)


def normalize_inference_steps(inference_steps, name):
    if inference_steps is None:
        return None
    if isinstance(inference_steps, bool) or not isinstance(inference_steps, Integral):
        raise TypeError(f"{name} must be an integer or None")
    if inference_steps <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(inference_steps)


def resolve_compile_model(
    compile_model: bool, memory_profile: Literal["low_vram", "resident"]
) -> bool:
    if memory_profile == "low_vram" and compile_model:
        logger.warning(
            "torch.compile is disabled in low_vram mode because compiled CUDA graphs "
            "can retain stage allocations"
        )
        return False
    return compile_model


class StageResidencyManager:
    """Moves complete inference stages between CPU and the execution device."""

    def __init__(
        self, device: torch.device, enabled: bool, profile_memory: bool = True
    ):
        self.device = torch.device(device)
        self.enabled = enabled and self.device.type == "cuda"
        self.profile_memory = profile_memory and self.device.type == "cuda"
        self.measurements = []

    @staticmethod
    def _move(module, device):
        if module is None:
            return
        if hasattr(module, "to"):
            module.to(device)
        elif hasattr(module, "model") and hasattr(module.model, "to"):
            module.model.to(device)
            if hasattr(module, "device"):
                module.device = torch.device(device)
        else:
            raise TypeError(f"Cannot move stage object of type {type(module)!r}")

    @staticmethod
    def _cache_clear_reason(allocated, reserved, inactive_split):
        reclaimable = max(0, reserved - allocated)
        if (
            inactive_split > CACHE_CLEAR_MIN_INACTIVE_SPLIT_BYTES
            and inactive_split > reserved // 4
        ):
            return "fragmented"
        if (
            reclaimable > CACHE_CLEAR_MIN_RECLAIMABLE_BYTES
            and reclaimable > reserved // 2
        ):
            return "large_unused_pool"
        return None

    @contextmanager
    def activate(self, name: str, modules: Iterable[object]):
        modules = tuple(module for module in modules if module is not None)
        total_started = time.perf_counter()
        if self.profile_memory:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        moved_modules = []
        activation_started = time.perf_counter()
        activation_seconds = 0.0
        compute_started = None
        try:
            if self.enabled:
                for module in modules:
                    moved_modules.append(module)
                    self._move(module, self.device)
            if self.profile_memory:
                torch.cuda.synchronize(self.device)
            activation_seconds = time.perf_counter() - activation_started
            compute_started = time.perf_counter()
            yield
        finally:
            measurement = None
            try:
                if self.profile_memory:
                    torch.cuda.synchronize(self.device)
                    if compute_started is None:
                        activation_seconds = time.perf_counter() - activation_started
                    compute_ended = time.perf_counter()
                    compute_seconds = (
                        compute_ended - compute_started
                        if compute_started is not None
                        else 0.0
                    )
                    measurement = {
                        "stage": name,
                        "started_monotonic_seconds": total_started,
                        "compute_started_monotonic_seconds": compute_started,
                        "compute_ended_monotonic_seconds": compute_ended,
                        # Kept for compatibility with earlier profiler output.
                        "elapsed_seconds": activation_seconds + compute_seconds,
                        "activation_seconds": activation_seconds,
                        "compute_seconds": compute_seconds,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            self.device
                        ),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                            self.device
                        ),
                        "allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "reserved_bytes": torch.cuda.memory_reserved(self.device),
                    }
            finally:
                offload_started = time.perf_counter()
                for module in reversed(moved_modules):
                    self._move(module, "cpu")
                if self.profile_memory:
                    torch.cuda.synchronize(self.device)
                offload_seconds = time.perf_counter() - offload_started

                cache_clear_reason = None
                cache_clear_seconds = 0.0
                if self.device.type == "cuda":
                    pre_clear_allocated = torch.cuda.memory_allocated(self.device)
                    pre_clear_reserved = torch.cuda.memory_reserved(self.device)
                else:
                    pre_clear_allocated = 0
                    pre_clear_reserved = 0
                if moved_modules:
                    stats = torch.cuda.memory_stats(self.device)
                    inactive = stats.get("inactive_split_bytes.all.current", 0)
                    cache_clear_reason = self._cache_clear_reason(
                        pre_clear_allocated, pre_clear_reserved, inactive
                    )
                    if cache_clear_reason is not None:
                        logger.info(
                            "Clearing CUDA cache after stage {} ({}, {:.2f} GiB reclaimable)",
                            name,
                            cache_clear_reason,
                            (pre_clear_reserved - pre_clear_allocated) / 1024**3,
                        )
                        cache_clear_started = time.perf_counter()
                        torch.cuda.empty_cache()
                        if self.profile_memory:
                            torch.cuda.synchronize(self.device)
                        cache_clear_seconds = (
                            time.perf_counter() - cache_clear_started
                        )
                if self.profile_memory and measurement is not None:
                    post_clear_allocated = torch.cuda.memory_allocated(self.device)
                    post_clear_reserved = torch.cuda.memory_reserved(self.device)
                    measurement.update(
                        {
                            "offload_seconds": offload_seconds,
                            "cache_clear_seconds": cache_clear_seconds,
                            "total_elapsed_seconds": time.perf_counter()
                            - total_started,
                            "ended_monotonic_seconds": time.perf_counter(),
                            "pre_clear_allocated_bytes": pre_clear_allocated,
                            "pre_clear_reserved_bytes": pre_clear_reserved,
                            "post_offload_allocated_bytes": post_clear_allocated,
                            "post_offload_reserved_bytes": post_clear_reserved,
                            "reclaimed_reserved_bytes": max(
                                0, pre_clear_reserved - post_clear_reserved
                            ),
                            "cache_cleared": cache_clear_reason is not None,
                            "cache_clear_reason": cache_clear_reason,
                        }
                    )
                    self.measurements.append(measurement)
                    logger.info(
                        "Stage {}: activate {:.3f}s, compute {:.3f}s, offload {:.3f}s, "
                        "peak allocated {:.2f} GiB, peak reserved {:.2f} GiB",
                        name,
                        measurement["activation_seconds"],
                        measurement["compute_seconds"],
                        measurement["offload_seconds"],
                        measurement["peak_allocated_bytes"] / 1024**3,
                        measurement["peak_reserved_bytes"] / 1024**3,
                    )


class InferencePipeline:
    def __init__(
        self,
        ss_generator_config_path,
        ss_generator_ckpt_path,
        slat_generator_config_path,
        slat_generator_ckpt_path,
        ss_decoder_config_path,
        ss_decoder_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        slat_decoder_gs_4_config_path=None,
        slat_decoder_gs_4_ckpt_path=None,
        ss_encoder_config_path=None,
        ss_encoder_ckpt_path=None,
        decode_formats=("mesh",),
        dtype="bfloat16",
        pad_size=1.0,
        version="v0",
        device="cuda",
        ss_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        ss_condition_input_mapping=["image"],
        slat_condition_input_mapping=["image"],
        pose_decoder_name="default",
        workspace_dir="",
        downsample_ss_dist=0,  # the distance we use to downsample
        ss_inference_steps=25,
        ss_rescale_t=3,
        ss_cfg_strength=7,
        ss_cfg_interval=[0, 500],
        ss_cfg_strength_pm=0.0,
        slat_inference_steps=25,
        slat_rescale_t=3,
        slat_cfg_strength=5,
        slat_cfg_interval=[0, 500],
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d,
        shape_model_dtype=None,
        compile_model=False,
        memory_profile: MemoryProfile = "auto",
        profile_memory=False,
        slat_mean=SLAT_MEAN,
        slat_std=SLAT_STD,
    ):
        self.rendering_engine = rendering_engine
        self.device = torch.device(device)
        self.memory_profile = resolve_memory_profile(memory_profile, self.device)
        self.low_vram = self.memory_profile == "low_vram"
        self.compile_model = resolve_compile_model(compile_model, self.memory_profile)
        self.load_device = torch.device("cpu") if self.low_vram else self.device
        self.stage_residency = StageResidencyManager(
            self.device, enabled=self.low_vram, profile_memory=profile_memory
        )
        logger.info(f"self.device: {self.device}")
        logger.info(f"memory profile: {self.memory_profile}")
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
        )
        if self.device.type == "cuda" and torch.cuda.is_available():
            logger.info(f"Actually using GPU: {torch.cuda.current_device()}")
        with self.load_device:
            self.decode_formats = normalize_output_formats(decode_formats)
            self.pad_size = pad_size
            self.version = version
            self.ss_condition_input_mapping = ss_condition_input_mapping
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.workspace_dir = workspace_dir
            self.downsample_ss_dist = downsample_ss_dist
            self.ss_inference_steps = ss_inference_steps
            self.ss_rescale_t = ss_rescale_t
            self.ss_cfg_strength = ss_cfg_strength
            self.ss_cfg_interval = ss_cfg_interval
            self.ss_cfg_strength_pm = ss_cfg_strength_pm
            self.slat_inference_steps = slat_inference_steps
            self.slat_rescale_t = slat_rescale_t
            self.slat_cfg_strength = slat_cfg_strength
            self.slat_cfg_interval = slat_cfg_interval

            self.dtype = self._get_dtype(dtype)
            if shape_model_dtype is None:
                self.shape_model_dtype = self.dtype
            else:
                self.shape_model_dtype = self._get_dtype(shape_model_dtype)

            # Setup preprocessors
            self.pose_decoder = self.init_pose_decoder(
                ss_generator_config_path, pose_decoder_name
            )
            self.ss_preprocessor = self.init_ss_preprocessor(
                ss_preprocessor, ss_generator_config_path
            )
            self.slat_preprocessor = slat_preprocessor

            self._decoder_specs = {
                "mesh": (slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path),
                "gaussian": (slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path),
                "gaussian_4": (
                    slat_decoder_gs_4_config_path,
                    slat_decoder_gs_4_ckpt_path,
                ),
            }

            logger.info("Loading model weights...")
            self._checkpoint_cache = {}
            self._cache_checkpoints = True

            ss_generator = self.init_ss_generator(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            ss_decoder = self.init_ss_decoder(
                ss_decoder_config_path, ss_decoder_ckpt_path
            )
            ss_encoder = self.init_ss_encoder(
                ss_encoder_config_path, ss_encoder_ckpt_path
            )
            # Load conditioner embedder so that we only load it once
            # Scope the weak construction cache to this pipeline. Live pipeline
            # instances must not unexpectedly share a backbone whose device is
            # controlled by another residency manager.
            Dino._shared_backbones.clear()
            try:
                ss_condition_embedder = self.init_ss_condition_embedder(
                    ss_generator_config_path, ss_generator_ckpt_path
                )
                slat_condition_embedder = self.init_slat_condition_embedder(
                    slat_generator_config_path, slat_generator_ckpt_path
                )
            finally:
                Dino._shared_backbones.clear()

            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
                "slat_condition_embedder": slat_condition_embedder,
            }
            self._share_dino_backbones()
            self._checkpoint_cache.clear()
            self._cache_checkpoints = False

            # override generator and condition embedder setting
            self.override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
                cfg_strength_pm=ss_cfg_strength_pm,
            )
            self.override_slat_generator_cfg_config(
                slat_generator,
                cfg_strength=slat_cfg_strength,
                inference_steps=slat_inference_steps,
                rescale_t=slat_rescale_t,
                cfg_interval=slat_cfg_interval,
            )

            models = {
                "ss_generator": ss_generator,
                "slat_generator": slat_generator,
                "ss_decoder": ss_decoder,
            }
            if ss_encoder is not None:
                models["ss_encoder"] = ss_encoder
            self.models = torch.nn.ModuleDict(models)
            for output_format in self.decode_formats:
                self._get_or_load_decoder(output_format)
            logger.info("Loading model weights completed!")

            if self.compile_model:
                logger.info("Compiling model...")
                self._compile()
                logger.info("Model compilation completed!")
            self.slat_mean = torch.tensor(slat_mean)
            self.slat_std = torch.tensor(slat_std)

    def _iter_dino_embedders(self):
        for condition_embedder in self.condition_embedders.values():
            if condition_embedder is None:
                continue
            for embedder, _ in condition_embedder.embedder_list:
                if isinstance(embedder, Dino):
                    yield embedder

    def _share_dino_backbones(self):
        """All released checkpoints contain the same frozen DINO-ViT-L weights."""
        dino_embedders = list(self._iter_dino_embedders())
        if not dino_embedders:
            return
        shared_backbone = dino_embedders[0].backbone
        for embedder in dino_embedders[1:]:
            embedder.backbone = shared_backbone
        logger.info(
            "Sharing one DINO backbone across {} conditioning adapters",
            len(dino_embedders),
        )

    def _start_condition_feature_cache(self):
        if self.compile_model:
            return None
        cache = {}
        for embedder in self._iter_dino_embedders():
            embedder.feature_cache = cache
        return cache

    def _clear_condition_feature_cache(self):
        for embedder in self._iter_dino_embedders():
            embedder.feature_cache = None

    @staticmethod
    def _tree_to_device(value, device):
        return tree_map_only(torch.Tensor, lambda tensor: tensor.to(device), value)

    def prepare_conditioning(self, ss_input_dict, slat_input_dict):
        """Encode both stage conditions before either generator becomes resident."""
        modules = tuple(self.condition_embedders.values())
        self._start_condition_feature_cache()
        try:
            with self.stage_residency.activate("conditioning", modules):
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.shape_model_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    ss_condition = self.get_condition_input(
                        self.condition_embedders["ss_condition_embedder"],
                        ss_input_dict,
                        self.ss_condition_input_mapping,
                    )
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.device.type == "cuda",
                ):
                    slat_condition = self.get_condition_input(
                        self.condition_embedders["slat_condition_embedder"],
                        slat_input_dict,
                        self.slat_condition_input_mapping,
                    )
        finally:
            self._clear_condition_feature_cache()

        if self.low_vram:
            ss_condition = self._tree_to_device(ss_condition, "cpu")
            slat_condition = self._tree_to_device(slat_condition, "cpu")
        return ss_condition, slat_condition

    @staticmethod
    def _share_identical_condition_inputs(ss_input, slat_input):
        for key in set(ss_input).intersection(slat_input):
            left, right = ss_input[key], slat_input[key]
            if (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
                and torch.equal(left, right)
            ):
                slat_input[key] = left

    def _get_or_load_decoder(self, output_format: OutputFormat):
        output_format = normalize_output_formats((output_format,))[0]
        model_key = {
            "mesh": "slat_decoder_mesh",
            "gaussian": "slat_decoder_gs",
            "gaussian_4": "slat_decoder_gs_4",
        }[output_format]
        if model_key in self.models:
            return self.models[model_key]

        config_path, ckpt_path = self._decoder_specs[output_format]
        if config_path is None or ckpt_path is None:
            raise ValueError(f"No decoder is configured for {output_format!r}")
        logger.info("Lazy-loading {} decoder on {}", output_format, self.load_device)
        with self.load_device:
            if output_format == "mesh":
                decoder = self.init_slat_decoder_mesh(config_path, ckpt_path)
            else:
                decoder = self.init_slat_decoder_gs(config_path, ckpt_path)
        self.models[model_key] = decoder
        return decoder

    def get_memory_report(self):
        return tuple(
            dict(measurement) for measurement in self.stage_residency.measurements
        )

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"
        logger.info(f"Compile mode {compile_mode}")

        def clone_output_wrapper(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                outputs = f(*args, **kwargs)
                return tree_map_only(
                    torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
                )

            return wrapped

        self.embed_condition = clone_output_wrapper(
            torch.compile(
                self.embed_condition,
                mode=compile_mode,
                fullgraph=True,  # _preprocess_input in dino is not compatible with fullgraph
            )
        )
        self.models["ss_generator"].reverse_fn.inner_forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_generator"].reverse_fn.inner_forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self.models["ss_decoder"].forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_decoder"].forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)

        for _ in tqdm(range(num_warmup_iters)):
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            ss_return_dict = self.sample_sparse_structure(ss_input_dict)
            coords = ss_return_dict["coords"]
            slat = self.sample_slat(slat_input_dict, coords)

    def instantiate_and_load_from_pretrained(
        self,
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cpu",
    ):
        model = instantiate(config)

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device=str(device))
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            checkpoint = None
            if self._cache_checkpoints and os.path.isfile(ckpt_path):
                checkpoint = self._checkpoint_cache.get(ckpt_path)
                if checkpoint is None:
                    checkpoint = torch.load(
                        ckpt_path,
                        map_location="cpu",
                        weights_only=False,
                        mmap=True,
                    )
                    self._checkpoint_cache[ckpt_path] = checkpoint
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
                checkpoint=checkpoint,
            )
        model = model.to(device)

        return model

    def init_pose_decoder(self, ss_generator_config_path, pose_decoder_name):
        if pose_decoder_name is None:
            pose_decoder_name = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_generator_config_path)
            )["module"]["pose_target_convention"]
        logger.info(f"Using pose decoder: {pose_decoder_name}")
        return get_pose_decoder(pose_decoder_name)

    def init_ss_preprocessor(self, ss_preprocessor, ss_generator_config_path):
        if ss_preprocessor is not None:
            return ss_preprocessor
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )["tdfy"]["val_preprocessor"]
        return instantiate(config)

    def init_ss_generator(self, ss_generator_config_path, ss_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )["module"]["generator"]["backbone"]

        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )

        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.load_device,
        )

    def init_slat_generator(self, slat_generator_config_path, slat_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_generator_config_path)
        )["module"]["generator"]["backbone"]
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.load_device,
        )

    def init_ss_encoder(self, ss_encoder_config_path, ss_encoder_ckpt_path):
        if ss_encoder_ckpt_path is not None:
            # override to avoid problem loading
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_encoder_config_path)
            )
            if "pretrained_ckpt_path" in config:
                del config["pretrained_ckpt_path"]
            return self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, ss_encoder_ckpt_path),
                device=self.load_device,
                state_dict_key=None,
            )
        else:
            return None

    def init_ss_decoder(self, ss_decoder_config_path, ss_decoder_ckpt_path):
        # override to avoid problem loading
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_decoder_config_path)
        )
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_decoder_ckpt_path),
            device=self.load_device,
            state_dict_key=None,
        )

    def init_slat_decoder_gs(
        self, slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
    ):
        if slat_decoder_gs_config_path is None:
            return None
        else:
            return self.instantiate_and_load_from_pretrained(
                OmegaConf.load(
                    os.path.join(self.workspace_dir, slat_decoder_gs_config_path)
                ),
                os.path.join(self.workspace_dir, slat_decoder_gs_ckpt_path),
                device=self.load_device,
                state_dict_key=None,
            )

    def init_slat_decoder_mesh(
        self, slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
    ):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_decoder_mesh_config_path)
        )
        config["device"] = str(self.load_device)
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_decoder_mesh_ckpt_path),
            device=self.load_device,
            state_dict_key=None,
        )

    def init_ss_condition_embedder(
        self, ss_generator_config_path, ss_generator_ckpt_path
    ):
        conf = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )
        if "condition_embedder" in conf["module"]:
            backbone_config = conf["module"]["condition_embedder"]["backbone"]
            for embedder_entry in backbone_config.get("embedder_list", ()):
                embedder_config = embedder_entry[0]
                if str(embedder_config.get("_target_", "")).endswith(".Dino"):
                    embedder_config["share_backbone"] = True
            return self.instantiate_and_load_from_pretrained(
                backbone_config,
                os.path.join(self.workspace_dir, ss_generator_ckpt_path),
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.condition_embedder."
                ),
                device=self.load_device,
            )
        else:
            return None

    def init_slat_condition_embedder(
        self, slat_generator_config_path, slat_generator_ckpt_path
    ):
        return self.init_ss_condition_embedder(
            slat_generator_config_path, slat_generator_ckpt_path
        )

    def override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
        cfg_strength_pm=0.0,
    ):
        # override generator setting
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = cfg_strength_pm

        logger.info(
            "ss_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}, cfg_strength_pm={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
            cfg_strength_pm,
        )

    def override_slat_generator_cfg_config(
        self,
        slat_generator,
        cfg_strength=5,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        slat_generator.inference_steps = inference_steps
        slat_generator.reverse_fn.strength = cfg_strength
        slat_generator.reverse_fn.interval = cfg_interval
        slat_generator.rescale_t = rescale_t

        logger.info(
            "slat_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        decode_formats=None,
        mesh_target_faces=None,
        flat_shading=False,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        mesh_target_faces = normalize_mesh_target_faces(mesh_target_faces)
        stage1_inference_steps = normalize_inference_steps(
            stage1_inference_steps, "stage1_inference_steps"
        )
        stage2_inference_steps = normalize_inference_steps(
            stage2_inference_steps, "stage2_inference_steps"
        )
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            self._share_identical_condition_inputs(ss_input_dict, slat_input_dict)
            ss_condition, slat_condition = self.prepare_conditioning(
                ss_input_dict, slat_input_dict
            )
            torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
                prepared_condition=ss_condition,
            )

            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if "scale" in ss_return_dict:
                logger.info(
                    f"Rescaling scale by {ss_return_dict['downsample_factor']}"
                )
                ss_return_dict["scale"] = (
                    ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
                )
            ss_return_dict = self._compact_sparse_outputs(ss_return_dict)
            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return self._move_outputs_to_cpu(ss_return_dict)

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                prepared_condition=slat_condition,
            )

            requested_formats = normalize_output_formats(
                self.decode_formats if decode_formats is None else decode_formats
            )
            decode_request = list(requested_formats)
            if (
                with_texture_baking
                and "mesh" in requested_formats
                and "gaussian" not in decode_request
            ):
                decode_request.append("gaussian")
            outputs = self.decode_slat(slat, decode_request)
            outputs = self.postprocess_slat_output(
                outputs,
                with_mesh_postprocess,
                with_texture_baking,
                use_vertex_color,
                mesh_target_faces=mesh_target_faces,
                flat_shading=flat_shading,
            )
            logger.info("Finished!")

            return self._move_outputs_to_cpu(
                {
                    **ss_return_dict,
                    **outputs,
                }
            )

    def postprocess_slat_output(
        self,
        outputs,
        with_mesh_postprocess,
        with_texture_baking,
        use_vertex_color,
        mesh_target_faces=None,
        flat_shading=False,
    ):
        # GLB files can be extracted from the outputs
        logger.info(
            f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
        )
        if "mesh" in outputs:
            appearance = outputs.get("gaussian")
            if with_texture_baking and not appearance:
                raise ValueError("Texture baking requires the gaussian output format")
            if mesh_target_faces is not None:
                outputs["mesh"][0] = postprocessing_utils.simplify_mesh_representation(
                    outputs["mesh"][0], mesh_target_faces
                )
            glb = postprocessing_utils.to_glb(
                appearance[0] if appearance else None,
                outputs["mesh"][0],
                # Optional parameters
                simplify=0.95,  # Ratio of triangles to remove in the simplification process
                texture_size=1024,  # Size of the texture used for the GLB
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                flat_shading=flat_shading,
                rendering_engine=self.rendering_engine,
            )

        # glb.export("sample.glb")
        else:
            glb = None

        outputs["glb"] = glb

        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]

        if "gaussian_4" in outputs:
            outputs["gs_4"] = outputs["gaussian_4"][0]

        return outputs

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)

            mask = np.array(mask)
            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]
            image = np.concatenate([image[..., :3], mask], axis=-1)

        image = np.array(image)
        return image

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: Sequence[str] = ("mesh",),
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        logger.info("Decoding sparse latent...")
        formats = normalize_output_formats(formats)
        ret = {}
        for output_format in formats:
            decoder = self._get_or_load_decoder(output_format)
            with self.stage_residency.activate(f"decode_{output_format}", (decoder,)):
                with torch.inference_mode():
                    ret[output_format] = decoder(slat)
        # if "radiance_field" in formats:
        #     ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def _move_outputs_to_cpu(self, outputs):
        formats_to_move = SUPPORTED_OUTPUT_FORMATS if self.low_vram else ("mesh",)
        for output_format in formats_to_move:
            for value in outputs.get(output_format, ()):
                if hasattr(value, "to"):
                    value.to("cpu")
        return tree_map_only(torch.Tensor, lambda tensor: tensor.cpu(), outputs)

    def _compact_sparse_outputs(self, outputs):
        if not self.low_vram:
            return outputs
        # The dense stage-one latent is only needed by the sparse decoder and
        # pose decoder. Keeping it alive would overlap with stage two.
        outputs.pop("shape", None)
        return self._tree_to_device(outputs, "cpu")

    def is_mm_dit(self, model_name="ss_generator"):
        return hasattr(self.models[model_name].reverse_fn.backbone, "latent_mapping")

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        logger.info("Running condition embedder ...")
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        logger.info("Condition embedder finishes!")
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def sample_sparse_structure(
        self,
        ss_input_dict: dict,
        inference_steps=None,
        use_distillation=False,
        prepared_condition=None,
    ):
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        logger.info(
            "Sampling sparse structure: inference_steps={}, strength={}, interval={}, rescale_t={}, cfg_strength_pm={}",
            ss_generator.inference_steps,
            ss_generator.reverse_fn.strength,
            ss_generator.reverse_fn.interval,
            ss_generator.rescale_t,
            ss_generator.reverse_fn.strength_pm,
        )

        stage_modules = (ss_generator, ss_decoder)
        try:
            with self.stage_residency.activate("sparse_structure", stage_modules):
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.shape_model_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    if self.is_mm_dit():
                        latent_shape_dict = {
                            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                        }
                    else:
                        latent_shape_dict = (bs,) + (4096, 8)

                    if prepared_condition is None:
                        condition_args, condition_kwargs = self.get_condition_input(
                            self.condition_embedders["ss_condition_embedder"],
                            ss_input_dict,
                            self.ss_condition_input_mapping,
                        )
                    else:
                        condition_args, condition_kwargs = self._tree_to_device(
                            prepared_condition, self.device
                        )
                    return_dict = ss_generator(
                        latent_shape_dict,
                        image.device,
                        *condition_args,
                        **condition_kwargs,
                    )
                    if not self.is_mm_dit():
                        return_dict = {"shape": return_dict}

                    shape_latent = return_dict["shape"]
                    ss = ss_decoder(
                        shape_latent.permute(0, 2, 1)
                        .contiguous()
                        .view(shape_latent.shape[0], 8, 16, 16, 16)
                    )
                    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                    return_dict["coords_original"] = coords
                    original_shape = coords.shape
                    if self.downsample_ss_dist > 0:
                        coords = prune_sparse_structure(
                            coords,
                            max_neighbor_axes_dist=self.downsample_ss_dist,
                        )
                    coords, downsample_factor = downsample_sparse_structure(coords)
                    logger.info(
                        f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                    )
                    return_dict["coords"] = coords
                    return_dict["downsample_factor"] = downsample_factor
        finally:
            ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def sample_slat(
        self,
        slat_input: dict,
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        prepared_condition=None,
    ) -> sp.SparseTensor:
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            "Sampling sparse latent: inference_steps={}, strength={}, interval={}, rescale_t={}",
            slat_generator.inference_steps,
            slat_generator.reverse_fn.strength,
            slat_generator.reverse_fn.interval,
            slat_generator.rescale_t,
        )

        stage_modules = (slat_generator,)
        try:
            with self.stage_residency.activate("structured_latent", stage_modules):
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.device.type == "cuda",
                ):
                    if prepared_condition is None:
                        condition_args, condition_kwargs = self.get_condition_input(
                            self.condition_embedders["slat_condition_embedder"],
                            slat_input,
                            self.slat_condition_input_mapping,
                        )
                    else:
                        condition_args, condition_kwargs = self._tree_to_device(
                            prepared_condition, self.device
                        )
                    condition_args += (coords.cpu().numpy(),)
                    slat = slat_generator(
                        latent_shape, DEVICE, *condition_args, **condition_kwargs
                    )
                    slat = sp.SparseTensor(
                        coords=coords,
                        feats=slat[0],
                    ).to(DEVICE)
                    slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)
        finally:
            slat_generator.inference_steps = prev_inference_steps
        return slat

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None:
            input = transform(input)

        return input

    def _preprocess_image_and_mask(
        self, rgb_image, mask_image, img_mask_joint_transform
    ):
        for trans in img_mask_joint_transform:
            rgb_image, mask_image = trans(rgb_image, mask_image)
        return rgb_image, mask_image

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray], preprocessor
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(input, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
        )

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }

        return item

    @staticmethod
    def _get_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            raise NotImplementedError
