import ast
import sys
from pathlib import Path

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


def test_python_runtime_matches_project_policy():
    assert (3, 14) <= sys.version_info[:2] < (3, 15)


def test_all_cli_entrypoints_enforce_python_compatibility_guard():
    entrypoints = [
        Path("server.py"),
        Path("remote.py"),
        Path("convert.py"),
        Path("scripts/example.py"),
    ]

    for entrypoint in entrypoints:
        source = entrypoint.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(entrypoint))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "python_compat"
            and any(alias.name == "require_supported_python" for alias in node.names)
            for node in module.body
        ), f"{entrypoint} must import require_supported_python"

        assert "require_supported_python()" in source, (
            f"{entrypoint} must call require_supported_python() in its startup path"
        )


def test_requirements_cover_runtime_top_level_dependencies():
    requirements = {
        line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("--")
    }

    runtime_import_to_requirement = {
        "cv2": "opencv-python-headless",
        "lycoris": "lycoris-lora",
        "packaging": "packaging",
        "pyside6": "pyside6",
        "requests": "requests",
        "scipy": "scipy",
        "skimage": "scikit-image",
        "spandrel": "spandrel",
        "ultralytics": "ultralytics",
        "yaml": "pyyaml",
    }

    missing = [
        requirement
        for requirement in runtime_import_to_requirement.values()
        if requirement not in requirements
    ]
    assert not missing, f"requirements.txt is missing runtime dependency pins: {missing}"
