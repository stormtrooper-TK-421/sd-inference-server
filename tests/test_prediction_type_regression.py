import json
import os
import sys
import types

import pytest
import torch
import safetensors.torch

from convert import (
    _detect_diffusers_architecture,
    convert_checkpoint,
    convert_diffusers_folder,
    normalize_prediction_type,
)


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


def _install_fake_diffusers_and_transformers(monkeypatch):
    class _FakeUNetModel:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_pretrained(cls, path):
            config_path = os.path.join(path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = types.SimpleNamespace(**json.load(f))
            return cls(config=config)

        def to(self, _dtype):
            return self

        def state_dict(self):
            return {"weight": torch.zeros((1,), dtype=torch.float16)}

    class _FakeComponent:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def to(self, _dtype):
            return self

        def state_dict(self):
            return {"weight": torch.zeros((1,), dtype=torch.float16)}

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKL = _FakeComponent
    fake_diffusers.UNet2DConditionModel = _FakeUNetModel

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPTextModel = _FakeComponent

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _make_min_diffusers_folder(tmp_path, prediction_type):
    model_dir = tmp_path / f"model_{prediction_type}"
    for sub in ["unet", "vae", "text_encoder", "scheduler"]:
        (model_dir / sub).mkdir(parents=True, exist_ok=True)
    with open(model_dir / "unet" / "config.json", "w", encoding="utf-8") as f:
        json.dump({"cross_attention_dim": 1024}, f)
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


@pytest.mark.parametrize(
    "fixture_name,expected_model_type,expected_model_variant",
    [
        ("sd1", "SDv1", ""),
        ("sd2", "SDv2", ""),
        ("sdxl_base", "SDXL-Base", ""),
        ("sdxl_refiner", "SDXL-Base", "Refiner"),
    ],
)
def test_detect_diffusers_architecture_fixtures(fixture_name, expected_model_type, expected_model_variant):
    model_dir = os.path.join("tests", "fixtures", "diffusers_arch", fixture_name)

    architecture = _detect_diffusers_architecture(model_dir)

    assert architecture["model_type"] == expected_model_type
    assert architecture["model_variant"] == expected_model_variant


def test_detect_diffusers_architecture_guardrail_for_missing_unet_config(tmp_path):
    with pytest.raises(ValueError, match="missing 'unet/config.json'"):
        _detect_diffusers_architecture(str(tmp_path))


def test_unet_determine_type_sdxl_does_not_crash_without_metadata(monkeypatch):
    """SDXL UNets with unknown prediction_type must survive determine_type().

    diffusers >=0.35 requires non-empty added_cond_kwargs (text_embeds + time_ids)
    for any UNet with addition_embed_type='text_time'.  A bare {} dict raises
    ValueError, so determine_type() must supply minimal dummy tensors for SDXL.
    """
    import types
    import torch

    # Build a minimal fake UNET that records the added_cond_kwargs it receives
    # and returns a constant prediction tensor (below the v-detection threshold).
    received_kwargs = {}

    class _FakeSample:
        def __init__(self, val):
            self._t = torch.full((1, 4, 8, 8), val)

        def any(self):
            return False  # no NaNs

    class _FakeOutput:
        def __init__(self, val):
            self.sample = torch.full((1, 4, 8, 8), val)

    class _FakeUNET:
        model_type = "SDXL-Base"
        prediction_type = "unknown"
        determined = False
        upcast_attention = False
        model_variant = ""
        device = torch.device("cpu")
        dtype = torch.float32

        class config:
            cross_attention_dim = 2048
            in_channels = 4

        def __call__(self, latent, timestep, encoder_hidden_states=None, **kwargs):
            received_kwargs.update(kwargs)
            # epsilon-like output (mean close to 0, well above -1 threshold)
            return _FakeOutput(0.0)

        # bind the real determine_type implementation onto our fake class
        from models import UNET as _UNET
        determine_type = _UNET.determine_type
        normalize_prediction_type = staticmethod(_UNET.normalize_prediction_type)

    fake = _FakeUNET()
    fake.determine_type(fake)  # must not raise

    assert fake.prediction_type in {"epsilon", "v"}, (
        f"prediction_type should be resolved, got {fake.prediction_type!r}"
    )
    assert "added_cond_kwargs" in received_kwargs, (
        "determine_type() must pass added_cond_kwargs for SDXL"
    )
    ack = received_kwargs["added_cond_kwargs"]
    assert "text_embeds" in ack and "time_ids" in ack, (
        "added_cond_kwargs must contain text_embeds and time_ids"
    )
    assert ack["text_embeds"].shape == (1, 1280)
    assert ack["time_ids"].shape == (1, 6)


@pytest.mark.parametrize(
    "fixture_name,expected_model_type,expected_model_variant",
    [
        ("sd1", "SDv1", ""),
        ("sd2", "SDv2", ""),
        ("sdxl_base", "SDXL-Base", ""),
        ("sdxl_refiner", "SDXL-Base", "Refiner"),
    ],
)
def test_convert_diffusers_folder_uses_detected_metadata(
    fixture_name, expected_model_type, expected_model_variant, monkeypatch
):
    _install_fake_diffusers_and_transformers(monkeypatch)
    model_dir = os.path.join("tests", "fixtures", "diffusers_arch", fixture_name)

    _, metadata = convert_diffusers_folder(model_dir)

    assert metadata["model_type"] == expected_model_type
    assert metadata["model_variant"] == expected_model_variant
    assert metadata["prediction_type"] == "epsilon"
