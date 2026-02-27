import sys

MIN_PYTHON = (3, 12)
MAX_PYTHON_EXCLUSIVE = (3, 14)


def _version_tuple(version_info):
    return (version_info.major, version_info.minor)


def _version_text(version_info):
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def ensure_supported_python(version_info=None):
    current = version_info or sys.version_info
    current_tuple = _version_tuple(current)
    if MIN_PYTHON <= current_tuple < MAX_PYTHON_EXCLUSIVE:
        return

    detected = _version_text(current)
    required = f">={MIN_PYTHON[0]}.{MIN_PYTHON[1]},<{MAX_PYTHON_EXCLUSIVE[0]}.{MAX_PYTHON_EXCLUSIVE[1]}"
    suggested_minor = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"

    raise RuntimeError(
        "Unsupported Python version for sd-inference-server. "
        f"Detected: {detected}. "
        f"Required range: {required}.\n"
        "Create and activate a compatible virtual environment, then reinstall dependencies:\n"
        f"  python{suggested_minor} -m venv .venv\n"
        "  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n"
        "  python -m pip install -r requirements.txt"
    )
