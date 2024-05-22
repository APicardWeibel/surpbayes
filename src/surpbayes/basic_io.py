"""Shared functions for basic read/write to human readable format"""
import json
from typing import Optional


def safe_load_json(
    path: str,
    buffering: int = -1,
    encoding: str = "utf-8",
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    closefd: bool = True,
    opener=None,
    **kwargs,
) -> dict:
    """Load a json file.
    If loading fails, makes sure the path is printed before raising the exception.
    """
    try:
        with open(
            file=path,
            mode="r",
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        ) as file:
            to_return = json.load(file, **kwargs)
        return to_return
    except FileNotFoundError as exc:
        print(f"{path} does not exist")
        raise exc
    except Exception as exc:
        print(f"File at {path} could not be loaded with json")
        raise exc
