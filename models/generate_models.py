import argparse
import json
import yaml


def replace_refs(input: dict, version_prefix: str, db_version_prefix: str):
    for k, v in input.items():
        if isinstance(v, dict):
            replace_refs(v, version_prefix, db_version_prefix)
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    replace_refs(i, version_prefix, db_version_prefix)
        if k == "$ref":  # and isinstance(v, str):
            assert isinstance(v, str)
            nv = v.replace(db_version_prefix, "db").replace(version_prefix, "")
            input[k] = nv


def filter_paths(paths: dict, version: str):
    new_dict = {}
    for k, v in paths.items():
        kparts = k.split("/")
        if kparts[1].startswith("slurm"):
            if kparts[2] == version:
                new_dict[k] = v
        else:
            new_dict[k] = v

    return new_dict


def filter_components(components: dict, version: str):
    new_dict = {}
    db_version = f"db{version}"
    vind = len(version) + 1
    for k, v in components.items():
        if k.startswith(version):
            new_dict[k[vind:]] = v
        elif k.startswith(db_version):
            new_dict["db" + k[vind + 1 :]] = v
    return new_dict


def generate_slurm_models(input_file: str, version: str):
    with open(input_file, "r") as f:
        schema = json.load(f)

    schema["paths"] = filter_paths(schema["paths"], version)
    schema["components"]["schemas"] = filter_components(
        schema["components"]["schemas"], version
    )
    replace_refs(schema, f"{version}_", f"db{version}")
    return schema


def create_argparser():
    ap = argparse.ArgumentParser(
        description="Generate YAML for given version of OpenAPI schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default="slurm-api.yaml",
    )
    return ap


if __name__ == "__main__":
    ap = create_argparser()
    args = ap.parse_args()
    schema = generate_slurm_models(args.input_file, args.version)
    with open(args.output_file, "w") as f:
        yaml.dump(schema, f)
