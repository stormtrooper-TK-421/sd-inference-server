import json
import sys
import types

import pytest
import torch
import safetensors.torch

from convert import convert_checkpoint, convert_diffusers_folder, normalize_prediction_type


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


def test_convert_prediction_type_normalization_maps_expected_values():
    assert normalize_prediction_type("epsilon") == "epsilon"
    assert normalize_prediction_type("v_prediction") == "v"


def test_convert_prediction_type_normalization_rejects_sample():
    with pytest.raises(ValueError, match="Unsupported prediction type 'sample'"):
        normalize_prediction_type("sample")


def _install_fake_diffusers_and_transformers(monkeypatch, cross_attention_dim=1024):
    class _FakeModel:
        def __init__(self, config=None):
            self.config = config or types.SimpleNamespace(cross_attention_dim=cross_attention_dim)

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def to(self, _dtype):
            return self

        def state_dict(self):
            return {"weight": torch.zeros((1,), dtype=torch.float16)}

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKL = _FakeModel
    fake_diffusers.UNet2DConditionModel = _FakeModel

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPTextModel = _FakeModel

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _make_min_diffusers_folder(tmp_path, prediction_type):
    model_dir = tmp_path / f"model_{prediction_type}"
    for sub in ["unet", "vae", "text_encoder", "scheduler"]:
        (model_dir / sub).mkdir(parents=True, exist_ok=True)
    with open(model_dir / "scheduler" / "scheduler_config.json", "w", encoding="utf-8") as f:
        json.dump({"prediction_type": prediction_type}, f)
    return model_dir


def test_convert_diffusers_folder_normalizes_v_prediction_metadata(tmp_path, monkeypatch):
    _install_fake_diffusers_and_transformers(monkeypatch)
    model_dir = _make_min_diffusers_folder(tmp_path, "v_prediction")

    _, metadata = convert_diffusers_folder(str(model_dir))

    assert metadata["prediction_type"] == "v"


def test_convert_diffusers_folder_keeps_epsilon_metadata(tmp_path, monkeypatch):
    _install_fake_diffusers_and_transformers(monkeypatch)
    model_dir = _make_min_diffusers_folder(tmp_path, "epsilon")

    _, metadata = convert_diffusers_folder(str(model_dir))

    assert metadata["prediction_type"] == "epsilon"
