from __future__ import annotations

import os
from pathlib import Path
from typing import AnyStr, Union


def check_jobscript_is_readable(jobscript: Path) -> Path:
    if not jobscript.is_file():
        raise FileNotFoundError(f"{jobscript} does not exist\n")

    if not (os.access(jobscript, os.R_OK) and os.access(jobscript, os.X_OK)):
        raise PermissionError(f"{jobscript} must be readable and executable by user\n")

    try:
        js = jobscript.open()
        js.close()
    except IOError:
        logging.error(f"{jobscript} cannot be opened\n")
        raise

    else:
        return jobscript

def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location)
    if Path("/dls") in location_path.parents or Path("/home") in location_path.parents or Path("/dls_sw") in location_path.parents:
        return location_path
    raise ValueError(f"{location_path} must be located within /dls, 'dls_sw or /home")


def decode_to_string(any_string: AnyStr) -> str:
    output = any_string.decode() if not isinstance(any_string, str) else any_string
    return output


def get_absolute_path(filename: Union[Path, str]) -> str:
    python_path = os.environ['PYTHONPATH'].split(os.pathsep)
    for search_path in python_path:
        for root, dir, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)

    return os.path.abspath(filename)


def slice_to_string(s: slice) -> str:
    start = s.start
    stop = '' if s.stop is None else s.stop
    step = s.step
    return f"{start}:{stop}:{step}"


def string_to_slice(s: str) -> slice:
    start_str, stop_str, step_str = s.split(':')
    start = 0 if start_str == '' else int(start_str)
    stop = None if stop_str == '' else stop_str
    step = int(step_str)
    return slice(start, stop, step)
