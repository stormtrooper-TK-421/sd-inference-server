import torch

from convert import _read_prediction_type
from models import set_module_tensor_to_device


class _FakeConfig:
    prediction_type = "v_prediction"


class _FakeUNet:
    config = _FakeConfig()


def test_read_prediction_type_prefers_scheduler_top_level(tmp_path):
    scheduler_file = tmp_path / "scheduler_config.json"
    scheduler_file.write_text('{"prediction_type": "epsilon"}')

    prediction_type = _read_prediction_type(str(scheduler_file), _FakeUNet())

    assert prediction_type == "epsilon"


def test_read_prediction_type_supports_nested_config_and_unet_fallback(tmp_path):
    scheduler_file = tmp_path / "scheduler_config.json"
    scheduler_file.write_text('{"config": {"prediction_type": "sample"}}')

    prediction_type = _read_prediction_type(str(scheduler_file), _FakeUNet())
    assert prediction_type == "sample"

    missing_prediction_type = _read_prediction_type(str(tmp_path / "missing.json"), _FakeUNet())
    assert missing_prediction_type == "v_prediction"


def test_set_module_tensor_to_device_compatibility_wrapper_updates_parameter():
    module = torch.nn.Linear(4, 3, bias=False)
    replacement = torch.ones_like(module.weight) * 2

    set_module_tensor_to_device(module, "weight", replacement)

    assert torch.equal(module.weight.detach(), replacement)
