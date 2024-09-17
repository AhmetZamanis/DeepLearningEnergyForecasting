from pathlib import Path

def get_root_dir() -> Path:
    """
    Returns the project root directory, anywhere in the project.
    """
    return Path(__file__).parent.parent