from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


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

def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location).resolve()
    top = location_path.parts[1]
    if top in ("dls", "dls_sw", "home"):
        return location_path
    raise ValueError(
        f"{location_path} must be located within /dls, /dls_sw or /home (to be accessible from the cluster)")


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

from yaml import YAMLObject, SafeLoader
from dataclasses import dataclass

@dataclass
class PPCCluster(YAMLObject):
    yaml_tag = "!PPCCluster"
    yaml_loader = SafeLoader

    default_queue: str # default cluster queue
    user_queues: Optional[Dict[str, List[str]]] = None # specific queues with allowed users
    resources: Optional[Dict[str,str]] = None # job resources

@dataclass
class PPCConfig(YAMLObject):
    yaml_tag = "!PPCConfig"
    yaml_loader = SafeLoader

    allowed_programs: Dict[str,str] # program name, python package with wrapper module
    project_env_var: str # name of environment variable holding project passed to qsub
    cluster_help_msg: str # Message to display if cluster commands are not available
    clusters: Dict[str, PPCCluster] # per cluster configuration, key is UGE_CELL

PPC_YAML='par_proc_co.yaml'

def load_cfg() -> PPCConfig:
    '''
    Load configuration from par_proc_co.yaml
    --- !PPCConfig
    allowed_programs:
        blah1: whatever_package1 # program name: python package (module expected to be called blah1_wrapper and contain a Wrapper class) 
        blah2: whatever_package2
    project_env_var: CLUSTER_PROJECT # name of environment variable holding project passed to qsub
    cluster_help_msg: Please module load blah # Message to display if cluster commands are not available
    clusters:
        cluster_one: !PPCCluster # cluster name 
            default_queue: basic.q # default cluster queue
            user_queues: # dictionary of queues and list of users 
                better.q: middle_user1
                best.q: power_user1, power_user2
        cluster_two: !PPCCluster
            default_queue: only.q
            resources:
                cpu_model: arm64
    '''
    cfg = find_cfg_file(PPC_YAML)
    import yaml
    with open(cfg, 'r') as cff:
        ppc_config = yaml.safe_load(cff)

    for ccfg in ppc_config.clusters.values():
        if ccfg.user_queues:
            users = set() # check for overlaps
            for us in ccfg.user_queues.values():
                common = users.intersection(set(us))
                if common:
                    raise ValueError('Users %s cannot be assigned to more than one queue', ', '.join(common))
                users.update(us)
    return ppc_config

def find_cfg_file(name: str) -> Path:
    '''
    '''
    cp = Path.home() / ("." + name)
    if cp.is_file():
        return cp
    
    g_parent = Path(os.path.realpath(__file__)).parent.parent
    places = (g_parent, Path(os.getenv('CONDA_PREFIX', '')) / 'etc', Path('/etc'))
    for p in places:
        cp = p / name
        if cp.is_file():
            return cp
    raise ValueError('Cannot find {} in {}'.format(name, places))
