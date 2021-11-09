from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union


def check_jobscript_is_readable(jobscript: Path) -> Path:
    if not jobscript.is_file():
        raise FileNotFoundError(f"{jobscript} does not exist")

    if not (os.access(jobscript, os.R_OK) and os.access(jobscript, os.X_OK)):
        raise PermissionError(f"{jobscript} must be readable and executable by user")

    try:
        js = jobscript.open()
        js.close()
    except IOError:
        logging.error(f"{jobscript} cannot be opened")
        raise

    else:
        return jobscript

def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location).resolve()
    top = location_path.parts[1]
    if top in ("dls", "dls_sw", "home"):
        return location_path
    raise ValueError(f"{location_path} must be located within /dls, /dls_sw or /home (to be accessible from the cluster)")


def decode_to_string(any_string: Union[bytes, str]) -> str:
    output = any_string.decode() if not isinstance(any_string, str) else any_string
    return output


def get_absolute_path(filename: Union[Path, str]) -> str:
    p = Path(filename).resolve()
    if p.is_file():
        return str(p)
    from shutil import which
    f = which(filename)
    if f:
        return f
    raise ValueError(f"{filename} not found")


def slice_to_string(s: Optional[slice]) -> str:
    if s is None:
        return '::'
    start = s.start
    stop = '' if s.stop is None else s.stop
    step = s.step
    return f"{start}:{stop}:{step}"
