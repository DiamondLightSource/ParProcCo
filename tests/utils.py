from __future__ import annotations

import getpass
import os
from os import path
from pathlib import Path

PARTITION = os.getenv("SLURM_PARTITION")


def get_slurm_rest_url() -> str | None:
    return os.getenv("SLURM_REST_URL")


def get_tmp_base_dir() -> str:
    if path.isdir("/dls"):
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}"
    else:
        tmp_dir = "test_dir"
    assert path.isdir(tmp_dir), f"{tmp_dir} is not a directory"
    base_dir = f"{tmp_dir}/tests/"
    if not path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_aggregator_data_files(working_directory: Path) -> list[Path]:
    # create test files
    file_paths = [working_directory / f"file_0{i}.txt" for i in range(4)]
    file_contents = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    for file_path, content in zip(file_paths, file_contents):
        with open(file_path, "w") as f:
            f.write(content)
    return file_paths


def setup_data_file(working_directory: str) -> Path:
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n")
    return input_file_path


def setup_data_files(
    working_directory: str, cluster_output_dir: Path
) -> tuple[Path, list[Path], list[str], list[slice]]:
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
        slices = []
    for i in range(4):
        slices.append(slice(i, 8, 4))

    output_file_paths = [Path(cluster_output_dir) / f"out_{i}" for i in range(4)]
    output_nums = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    return input_file_path, output_file_paths, output_nums, slices


def setup_jobscript(working_directory: str) -> Path:
    jobscript = Path(working_directory) / "test_script"
    with open(jobscript, "x") as f:
        jobscript_lines = """
#!/usr/bin/env python3

import argparse


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="str: path to input file", type=str)
    parser.add_argument("--output", help="str: path to output file", type=str)
    parser.add_argument("--images", help="str: slice selection of images per input file (as 'start:stop:step')")
    return parser


def check_args(args):
    empty_fields = [k for k, v in vars(args).items() if v is None]
    if len(empty_fields) > 0:
        raise ValueError(f"Missing arguments: {empty_fields}")


def write_lines(input_path, output_path, images):
    start, stop, step = images.split(":")
    start = int(start) if start else 0
    stop = int(stop) if stop else None
    step = int(step) if step else 1
    with open(input_path, "r") as in_f:
        for i, line in enumerate(in_f):
            if stop and i >= stop:
                break

            elif i >= start and ((i - start) % step == 0):
                doubled = int(line.strip("\\n")) * 2
                doubled_str = f"{doubled}\\n"
                with open(output_path, "a+") as out_f:
                    out_f.write(doubled_str)


if __name__ == '__main__':
    '''
    $ jobscript --input-path input_path --output output_path --images slice_param
    '''
    parser = setup_parser()
    args, other_args = parser.parse_known_args()
    check_args(args)

    write_lines(args.input_path, args.output, args.images)
"""
        jobscript_lines = jobscript_lines.lstrip()
        f.write(jobscript_lines)
    os.chmod(jobscript, 0o775)
    return jobscript


def setup_aggregation_script(working_directory: str) -> Path:
    jobscript = Path(working_directory) / "aggregation_test_script"
    with open(jobscript, "x") as f:
        jobscript_lines = """
#!/usr/bin/env python3

import argparse
from pathlib import Path

from example.simple_data_aggregator import SimpleDataAggregator


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="str: path to aggregation output file", type=str)
    parser.add_argument("sliced_files", help="str: paths to sliced results files", type=str, nargs="+")
    return parser

if __name__ == '__main__':
    '''
    $ jobscript --output aggregation_file_path .. [sliced_data_files]
    '''
    parser = setup_parser()
    args, other_args = parser.parse_known_args()
    args.output = Path(args.output)

    SimpleDataAggregator().aggregate(args.output, args.sliced_files)
"""
        jobscript_lines = jobscript_lines.lstrip()
        f.write(jobscript_lines)
    os.chmod(jobscript, 0o775)
    return jobscript


def setup_runner_script(working_directory: str) -> Path:
    parent_dir = Path(__file__).parent.resolve().parent
    runner_script = Path(working_directory) / "test_runner_script"
    with open(runner_script, "x") as f:
        runner_script_lines = f"""
#!/usr/bin/bash
. /etc/profile.d/modules.sh

module load python/3.11
export PYTHONPATH="${{PYTHONPATH}}:{parent_dir}"

echo "Executing |$@|"
eval "$@"
"""
        runner_script_lines = runner_script_lines.lstrip()
        f.write(runner_script_lines)
    os.chmod(runner_script, 0o775)
    return runner_script
