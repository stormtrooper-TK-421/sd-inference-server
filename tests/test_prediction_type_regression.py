import torch
import safetensors.torch

from convert import convert_checkpoint
from models import UNET


def test_convert_checkpoint_sdxl_without_metadata_keeps_unknown_prediction_type(tmp_path):
    checkpoint = tmp_path / "sdxl_missing_meta.safetensors"
    safetensors.torch.save_file(
        {
            "conditioner.embedders.1.model.text_projection": torch.zeros((1, 1), dtype=torch.float16),
        },
        str(checkpoint),
        metadata={},
    )

    _, metadata = convert_checkpoint(str(checkpoint))

    assert metadata["model_type"] == "SDXL-Base"
    assert metadata["prediction_type"] == "unknown"


def test_prediction_type_normalization_accepts_only_epsilon_or_v_for_runtime():
    assert UNET.normalize_prediction_type("EPS") == "epsilon"
    assert UNET.normalize_prediction_type("v_prediction", strict=True) == "v"

    try:
        UNET.normalize_prediction_type("unknown", strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown should be rejected in strict mode")
