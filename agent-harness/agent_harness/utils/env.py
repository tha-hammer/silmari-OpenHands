"""Environment variable loading utilities."""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_env_file(env_path: Optional[str | Path] = None) -> None:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. If None, searches for .env in:
            - Current working directory
            - Parent directory (project root)
            - ~/.agent-harness/.env
    """
    if load_dotenv is None:
        # python-dotenv not installed, skip loading
        return

    if env_path is None:
        # Try multiple locations
        possible_paths = [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path.home() / ".agent-harness" / ".env",
        ]

        for path in possible_paths:
            if path.exists():
                load_dotenv(path, override=False)
                return
    else:
        env_path_obj = Path(env_path)
        if env_path_obj.exists():
            load_dotenv(env_path_obj, override=False)


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)

