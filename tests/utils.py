from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


def setup_aggregator_data_files(working_directory: Path) -> List[Path]:
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


def setup_data_files(working_directory: str, cluster_output_dir: Path) -> Tuple[Path, List[Path], List[str],
                                                                                List[slice]]:
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
    jobscript = Path(working_directory) / "test_script.py"
    with open(jobscript, "x") as f:
        jobscript_lines = """
#!/usr/bin/env python3

import argparse


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="str: path to input file", type=str)
    parser.add_argument("--output-path", help="str: path to output file", type=str)
    parser.add_argument("-I", help="str: slice selection of images per input file (as 'start:stop:step')")
    return parser


def check_args(args):
    empty_fields = [k for k, v in vars(args).items() if v is None]
    if len(empty_fields) > 0:
        raise ValueError(f"Missing arguments: {empty_fields}")


def write_lines(input_path, output_path, images):
    start, stop, step = images.split(":")
    start = int(start)
    stop = int(stop)
    step = int(step)
    with open(input_path, "r") as in_f:
        for i, line in enumerate(in_f):
            if i >= stop:
                break

            elif i >= start and ((i - start) % step == 0):
                doubled = int(line.strip("\\n")) * 2
                doubled_str = f"{doubled}\\n"
                with open(output_path, "a+") as out_f:
                    out_f.write(doubled_str)


if __name__ == '__main__':
    '''
    $ python jobscript.py --input-path input_path --output-path output_path -I slice_param
    '''
    parser = setup_parser()
    args = parser.parse_args()
    check_args(args)

    write_lines(args.input_path, args.output_path, args.I)
"""
        jobscript_lines = jobscript_lines.lstrip()
        f.write(jobscript_lines)
    os.chmod(jobscript, 0o777)
    return jobscript