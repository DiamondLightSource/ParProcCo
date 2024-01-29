# ParProcCo

Requires a YAML configuration file in grandparent directory of package, CONDA_PREFIX/etc or /etc


```
--- !PPCConfig
allowed_programs:
    rs_map: msmapper_utils
    blah1: whatever_package1
    blah2: whatever_package2
url: https://slurm.local:8443
extra_property_envs: # optional dictionary for slurm job properties and environment variables
    account: MY_ACCOUNT # env var that holds account
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
