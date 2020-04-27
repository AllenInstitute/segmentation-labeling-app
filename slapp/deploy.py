import contextlib
import argparse
import os
import subprocess
from pathlib import Path
import time

import boto3
import git
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--region", default="us-west-2",
                    choices=["us-east-1", "us-west-2"])
parser.add_argument("--lambda-stack-name", default="gt-lambdas")
parser.add_argument("--lambda-s3-bucket")


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def deploy_lambdas(deploy_dir, name, region, prefix, bucket=None):
    try:
        with working_directory(deploy_dir):
            build = subprocess.run("sam build".split())
            if build.returncode != 0:
                raise SystemError
            deploy_cmd = (f"sam deploy --region {region} --stack-name {name} "
                          f"--s3-prefix {prefix}")
            if bucket:
                deploy_cmd = deploy_cmd + f" --s3-bucket {bucket}"
            deploy = subprocess.run(deploy_cmd.split())
            if deploy.returncode != 0:
                raise SystemError
    except SystemError:
        sys.exit(1)

if __name__ == "__main__":
    args = parser.parse_args()
    base_directory = Path(__file__).parents[1]     # segmentation-labeling-app
    branch = str(git.Repo().active_branch)
    prefix = f"{args.lambda_stack_name}/{branch}/{round(time.time())}"
    deploy_lambdas(base_directory, args.lambda_stack_name, args.region,
                   prefix, args.lambda_s3_bucket)
