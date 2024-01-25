import argparse
import json

import yaml


def replace_refs(in_dict: dict, version_prefix: str, db_version_prefix: str):
    for k, v in in_dict.items():
        if isinstance(v, dict):
            replace_refs(v, version_prefix, db_version_prefix)
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    replace_refs(i, version_prefix, db_version_prefix)
        if k == "$ref":  # and isinstance(v, str):
            assert isinstance(v, str)
            nv = v.replace(db_version_prefix, "db").replace(version_prefix, "")
            in_dict[k] = nv


def filter_paths(paths: dict, version: str, slurm_only: bool):
    new_dict = {}
    for k, v in paths.items():
        kparts = k.split("/")
        kp1 = kparts[1]
        if len(kparts) > 2:
            if kparts[2] == version:
                if (slurm_only and kp1 == "slurm") or (
                    not slurm_only and kp1 == "slurmdb"
                ):
                    new_dict[k] = v
        else:
            new_dict[k] = v
    print(new_dict.keys())
    return new_dict


def filter_components(components: dict, version: str, slurm_only: bool):
    new_dict = {}
    if not slurm_only:
        version = f"db{version}"
    vind = len(version) + 1
    kp = "" if slurm_only else "db_"
    for k, v in components.items():
        if k.startswith(version):
            new_dict[kp + k[vind:]] = v
    return new_dict


def generate_slurm_models(input_file: str, version: str, slurm_only: bool):
    with open(input_file, "r") as f:
        schema = json.load(f)

    schema["paths"] = filter_paths(schema["paths"], version, slurm_only)
    schema["components"]["schemas"] = filter_components(
        schema["components"]["schemas"], version, slurm_only
    )
    replace_refs(schema, f"{version}_", f"db{version}")
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
