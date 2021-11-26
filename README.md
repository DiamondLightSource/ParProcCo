# ParProcCo

Requires a YAML configuration file in grandparent directory of package, CONDA_PREFIX/etc or /etc


```
--- !PPCConfig
allowed_programs:
    rs_map: msmapper_utils
    blah1: whatever_package1
    blah2: whatever_package2
project_env_var: CLUSTER_PROJECT
cluster_help_msg: Please module load blah
clusters:
    cluster_one: !PPCCluster
        default_queue: basic.q
        user_queues:
            better.q: middle_user1
            best.q: power_user1, power_user2
    cluster_two: !PPCCluster
        default_queue: only.q
        resources:
            cpu_model: arm64
```
