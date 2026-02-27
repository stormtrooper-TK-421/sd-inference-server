from types import SimpleNamespace

import pytest

import runtime_compat


def _make_version(major, minor, micro=0):
    return SimpleNamespace(major=major, minor=minor, micro=micro)


def test_supported_python_range_allows_3_12_and_3_13():
    runtime_compat.ensure_supported_python(_make_version(3, 12, 1))
    runtime_compat.ensure_supported_python(_make_version(3, 13, 2))


def test_supported_python_range_rejects_out_of_range_with_actionable_message():
    with pytest.raises(RuntimeError) as exc:
        runtime_compat.ensure_supported_python(_make_version(3, 11, 9))

    message = str(exc.value)
    assert "Detected: 3.11.9" in message
    assert "Required range: >=3.12,<3.14" in message
    assert "python3.12 -m venv .venv" in message
