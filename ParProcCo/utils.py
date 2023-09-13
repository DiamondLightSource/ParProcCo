from __future__ import annotations

import logging
import os
import yaml

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Union
from yaml import YAMLObject, SafeLoader


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
    return jobscript


def jobscript_to_string(jobscript: Path) -> str:
    js = check_jobscript_is_readable(jobscript)
    with open(js, "r") as f:
        data = f.read()
    return data


def get_filepath_on_path(filename: Optional[str]) -> Optional[Path]:
    if filename is None:
        return None
    paths = os.environ["PATH"].split(":")
    path_gen = (
        os.path.join(p, filename)
        for p in paths
        if os.path.isfile(os.path.join(p, filename))
    )
    try:
        filepath = next(path_gen)
        return Path(filepath)
    except StopIteration:
        raise FileNotFoundError(f"{filename} not found on PATH {paths}")


def get_slurm_token() -> str:
    return os.environ["SLURM_JWT"]


def get_user() -> str:
    return os.environ["USER"]


def get_ppc_dir() -> str | None:
    return os.getenv("TEST_PPC_DIR")


def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location).resolve()
    top = location_path.parts[1]
    if top in ("dls", "dls_sw", "home"):
        return location_path
    raise ValueError(
        f"{location_path} must be located within /dls, /dls_sw or /home (to be accessible from the cluster)"
    )


def format_timestamp(t: datetime) -> str:
    return t.strftime("%Y%m%d_%H%M")


def decode_to_string(any_string: Union[bytes, str]) -> str:
    output = any_string.decode() if not isinstance(any_string, str) else any_string
    return output


def get_absolute_path(filename: Path | str | None) -> str:
    if filename is not None:
        p = Path(filename).resolve()
        if p.is_file():
            return str(p)
        from shutil import which

        f = which(filename)
        if f:
            return str(f)
    raise FileNotFoundError(f"{filename} not found")


def slice_to_string(s: Optional[slice]) -> str:
    if s is None:
        return "::"
    start = s.start
    stop = "" if s.stop is None else s.stop
    step = s.step
    return f"{start}:{stop}:{step}"


@dataclass
class PPCCluster(YAMLObject):
    yaml_tag = "!PPCCluster"
    yaml_loader = SafeLoader

    module: str  # module loaded to submit jobs
    default_queue: str  # default cluster queue
    user_queues: Optional[
        Dict[str, List[str]]
    ] = None  # specific queues with allowed users
    resources: Optional[Dict[str, str]] = None  # job resources


@dataclass
class PPCConfig(YAMLObject):
    yaml_tag = "!PPCConfig"
    yaml_loader = SafeLoader

    allowed_programs: Dict[str, str]  # program name, python package with wrapper module
    project_env_var: str  # name of environment variable holding project passed to qsub
    url: str  # slurm rest url


PPC_YAML = "par_proc_co.yaml"


def load_cfg() -> PPCConfig:
    """
    Load configuration from par_proc_co.yaml
    --- !PPCConfig
    allowed_programs:
        blah1: whatever_package1 # program name: python package (module expected to be called blah1_wrapper and contain a Wrapper class)
        blah2: whatever_package2
    url: slurm rest url
    """
    cfg = find_cfg_file(PPC_YAML)
    with open(cfg, "r") as cff:
        ppc_config = yaml.safe_load(cff)

    for ccfg in ppc_config.clusters.values():
        if ccfg.user_queues:
            users: Set[str] = set()  # check for overlaps
            for us in ccfg.user_queues.values():
                common = users.intersection(set(us))
                if common:
                    raise ValueError(
                        "Users %s cannot be assigned to more than one queue",
                        ", ".join(common),
                    )
                users.update(us)
    return ppc_config


PPC_ENTRY_POINT = "ParProcCo.allowed_programs"


def get_token(filepath: str | None) -> str:
    token = ""
    if filepath is None:
        try:
            token = get_slurm_token()
        except KeyError:
            raise ValueError(
                "No slurm token found. No slurm token filepath provided and no environment variable 'SLURM_JWT'"
            )
    else:
        if os.path.isfile(filepath):
            with open(filepath) as f:
                token = f.read()

    if token != "":
        return token
    raise FileNotFoundError(f"Slurm token file f{filepath} not found")


def set_up_wrapper(cfg: PPCConfig, program: str):
    allowed = cfg.allowed_programs
    if program in allowed:
        logging.info(f"{program} on allowed list in {cfg}")
        package = allowed[program]
    else:
        import sys

        if sys.version_info < (3, 10):
            from backports.entry_points_selectable import (
                entry_points,
            )  # @UnresolvedImport
        else:
            from importlib.metadata import entry_points  # @UnresolvedImport

        logging.info(f"Checking entry points for {program}")
        eps = entry_points(group=PPC_ENTRY_POINT)
        try:
            package = eps[program].module
        except Exception as exc:
            raise ValueError(
                f"Cannot find {program} in {PPC_ENTRY_POINT} {eps}"
            ) from exc

    import importlib

    try:
        wrapper_module = importlib.import_module(f"{package}.{program}_wrapper")
    except Exception as exc:
        raise ValueError(
            f"Cannot import {program}_wrapper as a Python module from package {package}"
        ) from exc
    try:
        return wrapper_module.Wrapper()
    except Exception as exc:
        raise ValueError(
            f"Cannot create Wrapper from {program}_wrapper module"
        ) from exc


def find_cfg_file(name: str) -> Path:
    """ """
    cp = os.getenv("PPC_CONFIG")
    if cp:
        return Path(cp)

    cp = Path.home() / ("." + name)
    if cp.is_file():
        return cp

    g_parent = Path(os.path.realpath(__file__)).parent.parent
    places = (g_parent, Path(os.getenv("CONDA_PREFIX", "")) / "etc", Path("/etc"))
    for p in places:
        cp = p / name
        if cp.is_file():
            return cp
    raise ValueError("Cannot find {} in {}".format(name, places))
