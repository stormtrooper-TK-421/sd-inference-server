import sys

MIN_VERSION = (3, 14)
MAX_VERSION_EXCLUSIVE = (3, 15)


def require_supported_python() -> None:
    version = sys.version_info[:2]
    if not (MIN_VERSION <= version < MAX_VERSION_EXCLUSIVE):
        raise RuntimeError(
            "Unsupported Python version "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "This project requires Python 3.14.x "
            "(>=3.14,<3.15)."
        )
