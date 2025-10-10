import os
from typing import Iterable

def validate_file_location(filepath: str) -> None:
    """Check if given file exists"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Provided file '{filepath}' does not exist!")

def validate_file_type(filepath: str, valid_extensions: Iterable[str]) -> None:
    """Check if given file is of valid type"""
    if not any(filepath.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Provided file '{filepath}' is not a valid file type!")
