# SD Inference Server

## Tested compatibility matrix

This branch is validated against the following software/hardware matrix and dependency ranges in `requirements.txt`.

| Component | Tested value(s) | Allowed range in `requirements.txt` |
|---|---|---|
| Python | 3.10.14, 3.11.9 | `>=3.10,<3.12` (project assumption) |
| CUDA runtime | 12.1 (primary), 12.4 (secondary smoke) | 12.x wheel builds for PyTorch 2.6 |
| GPU architecture | NVIDIA Ampere / Ada (`sm_80+`), tested on RTX 30xx/40xx | `sm_75+` may run with reduced throughput |
| torch | `2.6.0` | `>=2.6.0,<2.7` |
| torchvision | `0.21.0` | `>=0.21.0,<0.22` |
| diffusers | `0.35.0` | `>=0.35.0,<0.36` |
| transformers | `4.50.0` | `>=4.50.0,<4.53` |
| accelerate | `1.2.1` | `>=1.2.1,<1.4` |
| safetensors | `0.5.3` | `>=0.5.3,<0.6` |
| Pillow | `10.4.0` | `>=10.4.0,<11` |
| NumPy | `1.26.4` | `>=1.26.4,<2.2` |
| timm | `1.0.11` | `>=1.0.11,<1.1` |
| pytorch-lightning | `2.4.0` | `>=2.4.0,<2.5` |

## Upgrade constraints

The following packages are intentionally held back or constrained to avoid known API/ABI regressions during rollout:

- `torch` / `torchvision` are capped below the next major/minor pair (`<2.7`, `<0.22`) to avoid unvalidated CUDA kernel, Triton, and extension ABI changes.
- `diffusers` is capped at `<0.36` while checkpoint loading and scheduler wiring are validated against any new pipeline defaults.
- `transformers` is capped at `<4.53` to avoid tokenizer/model loader behavior changes that can impact local model repos.
- `accelerate` is capped at `<1.4` because offload and dispatch internals may shift across minor releases and require adapter updates.
- `safetensors` is capped at `<0.6` until all runtime readers/writers are validated against new metadata handling semantics.
- `numpy` is capped at `<2.2` to avoid binary compatibility surprises with transitive native dependencies.
- `Pillow` is capped at `<11` to keep image decode/encode behavior stable while migration tests are expanded.
- `k_diffusion` remains pinned at `0.0.15` because sampler integration in this repository uses legacy interfaces not yet refactored for newer snapshots.
- `basicsr` remains pinned at `1.4.2` due to historically brittle dependency trees and occasional breakage from transitive upgrades.

Use `requirements-legacy.txt` for rollback to the previous known-good baseline while migration validation is ongoing.

## Upgrade safeguards included

- `accelerate.utils.modeling.set_module_tensor_to_device` now uses a signature-aware wrapper so the code remains compatible with recent `accelerate` releases.
- Diffusers checkpoint conversion now handles scheduler config variants by reading prediction type from either:
  - `scheduler_config.json["prediction_type"]`
  - `scheduler_config.json["config"]["prediction_type"]`
  - falling back to `unet.config.prediction_type` or `"epsilon"`
- Smoke tests now validate txt2img + img2img paths for SD1.5/SD2.x/SDXL model families and assert decoded outputs are non-black.
