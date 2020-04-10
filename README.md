# segmentation-labeling-app
This repository contains the required infrastructure and data transformations to deploy a custom labeling workflow using AWS Sagemaker Ground Truth. It is currently in active development. The community is welcome to use this repository as a template or resource for creating Sagemaker Ground Truth applications.

## Level of support
We are not currently supporting this code, but simply releasing it to the community AS IS. We are not able to provide any guarantees of support.  The community may submit issues, but you should not expect an active response.

## Contributing
This tool is important for internal use at the Allen Institute. Because it's designed for internal needs, we are not anticipating external contributions. Pull requests may not be accepted if they conflict with our existing plans.

# Steps to deploy on Sagemaker Ground Truth

## Deploying pre annotation lambda handler

### Creating package
From upper level directory run the command
```console
sam package --s3-bucket <uri> --template-file templates/pre-annotation-template.yaml
--output-template-file templates/packaged-pre-annotation-template.yaml
```
This command packages the labmda function into a template that can be built
and deployed through SAM deploy command

### Deploying package
From same level run the command
```console
sam deploy --stack-name <stack-name> --template-file templates/packaged-pre-annotation-template.yaml
--capabilities CAPABILITY_IAM
```
This builds and deploys the lambda function to the AWS account linked through
the AWS CLI. It associates it with the stack-name or creates a new stack with
provided name. You can specify a role by providing the --role argument.

## Deploying post annotation lambda handler

### Creating package
From upper level directory run the command
```console
sam package --s3-bucket <uri> --template-file templates/post-annotation-template.yaml
--output-template-file templates/packaged-post-annotation-template.yaml
```
This command packages the labmda function into a template that can be built
and deployed through SAM deploy command

### Deploying package
From same level run the command
```console
sam deploy --stack-name <stack-name> --template-file templates/packaged-post-annotation-template.yaml
--capabilities CAPABILITY_IAM
```
This builds and deploys the lambda function to the AWS account linked through
the AWS CLI. It associates it with the stack-name or creates a new stack with
provided name. You can specify a role by providing the --role argument.
