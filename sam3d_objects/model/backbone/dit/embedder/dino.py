# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Optional, Dict, Any
import warnings
import weakref
from torchvision.transforms import Normalize
import torch.nn.functional as F
from loguru import logger


class Dino(torch.nn.Module):
    _shared_backbones = weakref.WeakValueDictionary()

    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vitb14",
        source: str = "github",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
        share_backbone: bool = False,
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        cache_key = (
            repo_or_dir,
            dino_model,
            source,
            repr(sorted(backbone_kwargs.items())),
        )
        self.backbone = (
            self._shared_backbones.get(cache_key) if share_backbone else None
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.backbone is None:
                logger.info(
                    f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})"
                )
                if backbone_kwargs:
                    logger.info(f"DINO backbone kwargs: {backbone_kwargs}")

                self.backbone = torch.hub.load(
                    repo_or_dir=repo_or_dir,
                    model=dino_model,
                    source=source,
                    verbose=False,
                    **backbone_kwargs,
                )
                if share_backbone:
                    self._shared_backbones[cache_key] = self.backbone
            else:
                logger.info("Reusing shared DINO model: {}", dino_model)

            # Log model properties after loading
            logger.info(
                f"Loaded DINO model - type: {type(self.backbone)}, "
                f"embed_dim: {self.backbone.embed_dim}, "
                f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}"
            )

        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.feature_cache = None
        self.register_buffer(
            "mean",
            torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1),
            persistent=False,
        )

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img, cache_key=None):
        output = None
        if self.feature_cache is not None and cache_key is not None:
            output = self.feature_cache.get(cache_key)
        if output is None:
            output = self.backbone.forward_features(input_img)
            if self.feature_cache is not None and cache_key is not None:
                self.feature_cache[cache_key] = output
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        cache_key = None
        if self.feature_cache is not None:
            autocast_enabled = torch.is_autocast_enabled()
            autocast_dtype = None
            if autocast_enabled:
                autocast_dtype = (
                    torch.get_autocast_gpu_dtype()
                    if x.is_cuda
                    else torch.get_autocast_cpu_dtype()
                )
            cache_key = (
                x.data_ptr(),
                x._version,
                tuple(x.shape),
                x.dtype,
                x.device,
                tuple(self.resize_input_size),
                self.normalize_images,
                autocast_enabled,
                autocast_dtype,
            )
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images, cache_key=cache_key)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)
