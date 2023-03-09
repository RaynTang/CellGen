from typing import Tuple
import os
import subprocess
import sys
import unittest


SEVERALGPU_TEST = [
    "bert_minimal_test",
    "gpt_minimal_test",
    "dynamic_batchsize_test",
]


def get_multigpu_launch_option(min_gpu):
    should_skip = False
    import torch

    num_devices = torch.cuda.device_count()
    if num_devices < min_gpu:
        should_skip = True
    distributed_run_options = f"-m torch.distributed.run --nproc_per_node={num_devices}"
    return should_skip, distributed_run_options


def get_launch_option(test_filename) -> Tuple[bool, str]:
    should_skip = False
    for severalgpu_test in SEVERALGPU_TEST:
        if severalgpu_test in test_filename:
            return get_multigpu_launch_option(3)
    return should_skip, ""


def run_transformer_tests():
    python_executable_path = sys.executable
    directory = os.path.dirname(__file__)
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("run_") and os.path.isfile(os.path.join(directory, f))
    ]
    print("#######################################################")
    print(f"# Python executable path: {python_executable_path}")
    print(f"# {len(files)} tests: {files}")
    print("#######################################################")
    errors = []
    for i, test_file in enumerate(files, 1):
        is_denied = False
        should_skip, launch_option = get_launch_option(test_file)
        if should_skip:
            print(
                f"### {i} / {len(files)}: {test_file} skipped. Requires multiple GPUs."
            )
            continue
        test_run_cmd = (
            f"{python_executable_path} {launch_option} {test_file} "
            "--micro-batch-size 2 --num-layers 16 --hidden-size 256 --num-attention-heads 8 --max-position-embeddings "
            "512 --seq-length 512 --global-batch-size 128"
        )
        if "bert" in test_file or "gpt" in test_file:
            import torch

            num_devices = torch.cuda.device_count()
            if "bert" in test_file:
                # "bert" uses the interleaving.
                tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices > 4 else 1
            if "gpt" in test_file:
                # "gpt" uses the non-interleaving.
                tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices >= 4 else 1
            pipeline_model_parallel_size = num_devices // tensor_model_parallel_size
            test_run_cmd += f" --pipeline-model-parallel-size {pipeline_model_parallel_size} --tensor-model-parallel-size {tensor_model_parallel_size}"

            if "bert" in test_file:
                test_run_cmd += f" --bert-no-binary-head"
        else:
            test_run_cmd += f" --use-cpu-initialization"
        print(f"### {i} / {len(files)}: cmd: {test_run_cmd}")
        try:
            output = (
                subprocess.check_output(test_run_cmd, shell=True)
                .decode(sys.stdout.encoding)
                .strip()
            )
        except Exception as e:
            errors.append((test_file, str(e)))
        else:
            if ">> passed the test :-)" not in output:
                errors.append((test_file, output))
    else:
        if not errors:
            print("### PASSED")
        else:
            print("### FAILED")
            short_msg = f"{len(errors)} out of {len(files)} tests failed"
            print(short_msg)
            for (filename, log) in errors:
                print(f"File: {filename}\nLog: {log}")
            raise RuntimeError(short_msg)


class TestTransformer(unittest.TestCase):
    def test_transformer(self):
        run_transformer_tests()


if __name__ == "__main__":
    unittest.main()
