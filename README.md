# SD Inference Server

## Tested compatibility matrix

This branch is validated against the following software/hardware matrix and dependency ranges in `requirements.txt`.

| Component | Tested value(s) | Allowed range in `requirements.txt` |
|---|---|---|
| Python | 3.12.x / 3.13.x (experimental) | `>=3.12,<3.14` (nightly availability dependent) |
| CUDA runtime | 13.x (experimental target) | PyTorch nightly `cu130` wheels |
| GPU architecture | NVIDIA Ampere / Ada (`sm_80+`), tested on RTX 30xx/40xx | `sm_75+` may run with reduced throughput |
| torch | nightly (`--pre`) | resolved from `https://download.pytorch.org/whl/nightly/cu130` |
| torchvision | nightly (`--pre`) | resolved from `https://download.pytorch.org/whl/nightly/cu130` |
| diffusers | `0.38.0` | pinned |
| transformers | `4.58.0` | pinned |
| accelerate | `1.12.0` | pinned |
| safetensors | `0.6.2` | pinned |
| Pillow | `11.3.0` | pinned |
| NumPy | `2.3.0` | pinned |
| timm | `1.0.20` | pinned |
| pytorch-lightning | `2.6.0` | pinned |

## Upgrade constraints

This branch is now intentionally **experimental** and tracks a CUDA 13-first stack.

- `torch`, `torchvision`, and `torchaudio` are resolved as **nightly pre-release wheels** from the PyTorch `cu130` index.
- Companion libraries are pinned to a recent set that is expected to interoperate with current nightly PyTorch builds.
- Breakage from upstream nightly churn is expected; update pins in `requirements.txt` as needed.

Install with:

```bash
python -m pip install -r requirements.txt
```

This command uses `--pre` and the `cu130` nightly index already embedded in `requirements.txt`.

## Upgrade safeguards included

- `accelerate.utils.modeling.set_module_tensor_to_device` now uses a signature-aware wrapper so the code remains compatible with recent `accelerate` releases.
- Diffusers checkpoint conversion now handles scheduler config variants by reading prediction type from either:
  - `scheduler_config.json["prediction_type"]`
  - `scheduler_config.json["config"]["prediction_type"]`
  - falling back to `unet.config.prediction_type` or `"epsilon"`
- Smoke tests now validate txt2img + img2img paths for SD1.5/SD2.x/SDXL model families and assert decoded outputs are non-black.

For the full repo-wide migration approach, see `docs/python-cuda13-migration-plan.md`.
