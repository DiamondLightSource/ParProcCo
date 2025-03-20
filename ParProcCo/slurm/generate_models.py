import argparse
import json

import yaml


def filter_refs(in_dict: dict, version_prefix: str, all_refs: set):
    sind = len("#/components/schemas/")
    for k, v in in_dict.items():
        if isinstance(v, dict):
            filter_refs(v, version_prefix, all_refs)
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    filter_refs(i, version_prefix, all_refs)
        if k == "$ref":  # and isinstance(v, str):
            assert isinstance(v, str)
            if version_prefix in v:
                nv = v.replace(version_prefix, "")
                all_refs.add(nv[sind:])
                in_dict[k] = nv


def filter_paths(paths: dict, version: str, slurm_only: bool):
    new_dict = {}
    path_head = "slurm" if slurm_only else "slurmdb"
    for k, v in paths.items():
        kparts = k.split(
            "/"
        )  # '/slurm/v0.0.40/shares' => ['', 'slurm', 'v0.0.40', 'shares']
        kp1 = kparts[1]
        if len(kparts) > 2:
            if kp1 == path_head and kparts[2] == version:
                new_dict[k] = v
        else:  # global paths
            new_dict[k] = v
    print(new_dict.keys())
    return new_dict


def filter_components(components: dict, version_prefix: str, all_refs: dict):
    new_dict = {}
    vind = len(version_prefix)
    for k, v in components.items():
        if k.startswith(version_prefix):
            filter_refs(v, version_prefix, all_refs)
            new_dict[k[vind:]] = v
    return new_dict


def generate_slurm_models(input_file: str, version: str, slurm_only: bool):
    with open(input_file, "r") as f:
        schema = json.load(f)

    schema["paths"] = filter_paths(schema["paths"], version, slurm_only)
    all_refs = set()
    version_prefix = f"{version}_"
    filter_refs(schema["paths"], version_prefix, all_refs)
    all_schemas = filter_components(
        schema["components"]["schemas"], version_prefix, all_refs
    )
    print(
        "Removing these unreferenced schema parts:", set(all_schemas.keys()) - all_refs
    )
    schema["components"]["schemas"] = {
        k: s for k, s in all_schemas.items() if k in all_refs
    }
    return schema


def create_argparser():
    ap = argparse.ArgumentParser(
        description="Generate YAML for given version of OpenAPI schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--db",
        "-d",
        help="output slurmdb models instead of slurm models",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--version", "-v", help="str: slurm OpenAPI version string", default="v0.0.38"
    )
    ap.add_argument(
        "input_file",
        help="str: path to file containing output from slurm OpenAPI endpoint",
    )
    ap.add_argument(
        "output_file",
        help="str: path to YAML file for versioned schema",
        nargs="?",
        default="slurm-rest.yaml",
    )
    return ap


if __name__ == "__main__":
    ap = create_argparser()
    args = ap.parse_args()
    schema = generate_slurm_models(args.input_file, args.version, not args.db)
    with open(args.output_file, "w") as f:
        yaml.dump(schema, f)
