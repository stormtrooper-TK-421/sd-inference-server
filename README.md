# SD Inference Server

## Tested compatibility matrix

This compatibility branch is validated against the following software/hardware matrix and pinned dependency ranges in `requirements.txt`.

| Component | Tested value |
|---|---|
| Python | 3.10.x |
| CUDA runtime | 12.1 |
| GPU architecture | NVIDIA Ampere / Ada (sm80+), tested on RTX 30xx/40xx |
| torch | `2.4.1` (range: `>=2.4.1,<2.6`) |
| torchvision | `0.19.1` (range: `>=0.19.1,<0.21`) |
| diffusers | `0.30.3` (range: `>=0.30.3,<0.32`) |
| transformers | `4.44.2` (range: `>=4.44.2,<4.48`) |
| accelerate | `0.34.2` (range: `>=0.34.2,<1.1`) |
| safetensors | `0.4.5` (range: `>=0.4.5,<0.6`) |
| Pillow | `10.4.0` (range: `>=10.4.0,<11`) |
| NumPy | `1.26.4` (range: `>=1.26.4,<2.1`) |

## Upgrade safeguards included

- `accelerate.utils.modeling.set_module_tensor_to_device` now uses a signature-aware wrapper so the code remains compatible with recent `accelerate` releases.
- Diffusers checkpoint conversion now handles scheduler config variants by reading prediction type from either:
  - `scheduler_config.json["prediction_type"]`
  - `scheduler_config.json["config"]["prediction_type"]`
  - falling back to `unet.config.prediction_type` or `"epsilon"`
- Smoke tests now validate txt2img + img2img paths for SD1.5/SD2.x/SDXL model families and assert decoded outputs are non-black.
