# ParProcCo

Requires a YAML configuration file in grandparent directory of package, CONDA_PREFIX/etc or /etc


```
--- !PPCConfig
allowed_programs:
    rs_map: msmapper_utils
    blah1: whatever_package1
    blah2: whatever_package2
url: https://slurm.local:8443
extra_property_envs: # optional mapping of properties to pass to Slurm's JobDescMsg
    account: MY_ACCOUNT # env var that holds account
    comment: mega job
valid_top_directories: # optional mapping of top directories accessible from cluster nodes
                       # (used to check job scripts, log and working directories)
    - /cluster_home
    - /cluster_apps
```

An entry point called `ParProcCo.allowed_programs` can be added to other packages' `setup.py`:

```
setup(
...
    entry_points={PPC_ENTRY_POINT: ['blah1 = whatever_package1']},
)
```

which will look for a module called `blah1_wrapper` in `whatever_package1` package.


## Testing

Tests can be run with
```
$ pytest tests
```
To test interactions with Slurm, set the following environment variables:
```
SLURM_REST_URL  # URL for server and port where the REST endpoints are hosted
SLURM_PARTITION # Slurm cluster parition 
SLURM_JWT       # JSON web token for access to REST endpoints
```

The environment can be set up and managed by running the `create_env` task in VSCode. This will read the token from
`~/.ssh/slurm.tkn` but will not check or generate the key. The resulting file `.vscode/.env` is used by the
`python.envFile` setting to propagate these values automatically.

On the initial run, `SLURM_REST_URL` and `SLURM_PARTITION` will need to be given values manually (unless already set as
environment variables). Those values will be kept whenever the task is rerun, with only the token being updated. As
`.vscode/.env` is ignored by git, it is safe to save these values in that file.

If you are not using VSCode, running `.vscode/create_env.sh` will create the env file, and the variables can be exported
using:
```
set -a
source ".vscode/.env"
set +a
```
