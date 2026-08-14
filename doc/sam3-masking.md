# SAM 3 text prompts to SAM 3D meshes

This integration uses the Hugging Face Transformers implementation of SAM 3.
The existing `/mnt/d/Data/models/sam3.safetensors` file is already in that
format, so the Meta SAM 3 repository is not needed as a submodule. Keeping SAM
3 in its own environment also avoids changing SAM 3D Objects' compiled PyTorch,
CUDA, `timm`, and rendering dependencies.

On a 16 GB GPU, use the CLI workflow below. `sam3-mask` exits before
`sam3d-prompt-to-mesh` constructs SAM 3D, so the two models never occupy GPU
memory at the same time.

## Create the masking environment

From the repository root:

```bash
micromamba create -n sam3-masking python=3.11 pip -y
micromamba run -n sam3-masking \
  pip install -r requirements.sam3.txt
micromamba run -n sam3-masking \
  pip install -e . --no-deps
```

`requirements.sam3.txt` pins PyTorch 2.8 with CUDA 12.8, torchvision 0.23,
and Transformers 5.14.1. Use a different PyTorch wheel only if required by the
installed NVIDIA driver; do not install Transformers into the SAM 3D Objects
environment.

Install the updated console entry points in the existing SAM 3D environment as
well, without resolving its dependencies again:

```bash
micromamba run -n sam3d-objects pip install -e . --no-deps
```

## Prepare an offline model bundle

The repository-root `.env` is ignored by git and defines `HF_TOKEN`.
`sam3-prepare` checks the process environment first, then that `.env`, then the
Hugging Face CLI token. It never accepts or prints a token. Only the small
configuration and tokenizer files are downloaded; the existing weights are
symlinked.

```bash
micromamba run -n sam3-masking sam3-prepare \
  --weights /mnt/d/Data/models/sam3.safetensors \
  --model-dir checkpoints/sam3-hf
```

After this one-time command, inference loads `checkpoints/sam3-hf` with
`local_files_only=True` and does not access Hugging Face.

## Quick prompt-to-mesh demo

Run the demo from the SAM 3D Objects environment. It discovers `sam3-mask` in a
sibling `sam3-masking` environment automatically; alternatively, set
`SAM3_MASK_EXECUTABLE` or pass `--sam3-executable`.

```bash
python scripts/demo_prompt_to_mesh.py \
  --image /path/to/image.jpg \
  --prompt "parked car"
```

Repeat `--prompt` to find multiple concepts and select an explicit destination:

```bash
python scripts/demo_prompt_to_mesh.py \
  --image /path/to/image.jpg \
  --prompt "parked car" \
  --prompt "traffic cone" \
  --output-dir outputs/demo
```

Without `--output-dir`, results are written to
`outputs/sam3-demo/<image-stem>/`. Each result contains
`segmentation/manifest.json`, one binary PNG under `segmentation/masks/` per
detected instance, and one successful GLB under `meshes/` per instance.

## Generate masks

```bash
micromamba run -n sam3-masking sam3-mask \
  --model-dir checkpoints/sam3-hf \
  --image notebook/images/edited_id14_032207032-scenes-traffic-rome-12-12_processed/image.png \
  --prompt "parked car" \
  --prompt "traffic sign" \
  --output-dir outputs/route-frame-0001 \
  --source-id route-frame-0001 \
  --profile-memory
```

The output contains `manifest.json` and one lossless binary PNG for every
detected instance. A zero-detection result is valid and produces an empty
manifest.

The same implementation is available as a persistent Python API. It computes
the image features once and reuses them for all prompts:

```python
from sam3_masking import Sam3MaskGenerator

with Sam3MaskGenerator.from_pretrained(
    "checkpoints/sam3-hf", device="cuda", dtype="auto"
) as generator:
    frame = generator.segment(
        "image.png",
        ["parked car", "traffic cone"],
        source_id="frame-0001",
    )

for prediction in frame.predictions:
    print(prediction.prompt, prediction.score, prediction.mask.shape)
```

## Generate one mesh per detected instance

Run this command from the SAM 3D Objects environment. Point it to the executable
installed in the dedicated masking environment:

```bash
sam3d-prompt-to-mesh \
  --sam3-executable /home/ubuntu/micromamba/envs/sam3-masking/bin/sam3-mask \
  --sam3-model-dir checkpoints/sam3-hf \
  --sam3d-config checkpoints/hf/pipeline.yaml \
  --image notebook/images/edited_id14_032207032-scenes-traffic-rome-12-12_processed/image.png \
  --prompt "parked car" \
  --prompt "traffic sign" \
  --output-dir outputs/route-frame-0001 \
  --memory-profile low_vram
```

High-quality SAM 3D sampler and mesh defaults are retained. Route-oriented
options can be added explicitly:

```bash
  --mesh-target-faces 10000 --flat-shading \
  --stage1-inference-steps 15 --stage2-inference-steps 15
```

The segmentation manifest is augmented with an `ok` or `failed` mesh record for
each instance. Failures do not prevent remaining objects from being attempted,
but the command exits nonzero if any mesh failed.

## Future Ouster ingestion

OSF/PCAP decoding is intentionally outside this integration. A route reader can
later assign the Ouster frame/timestamp to `source_id`, perform segmentation or
tracking as a first pass, terminate SAM 3, and reconstruct selected manifest
instances as a second pass. This preserves the 16 GB GPU lifecycle boundary.

## Optional GPU smoke test

```bash
RUN_SAM3_INTEGRATION=1 \
SAM3_MASK_EXECUTABLE=/home/ubuntu/micromamba/envs/sam3-masking/bin/sam3-mask \
SAM3_MODEL_DIR=checkpoints/sam3-hf \
pytest -m gpu tests/test_sam3_gpu_integration.py -s
```
