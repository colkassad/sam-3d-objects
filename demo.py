# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False, memory_profile="auto")

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# Route-scene assets only need to remain recognizable. A 10k-face budget and
# 15-step samplers reduce output complexity and warm inference time.
output = inference(
    image,
    mask,
    seed=42,
    mesh_target_faces=10_000,
    flat_shading=True,
    stage1_inference_steps=15,
    stage2_inference_steps=15,
)

# export vertex-colored mesh
output["glb"].export("mesh_low_poly.glb")
print(
    f"Saved mesh_low_poly.glb with {len(output['mesh'][0].faces):,} triangles"
)
