# segmentation-labeling-app
Data pipeline and UI for human labeling of putative ROIs from 2p cell segmentations

# Deploying pre annotation lambda handler

## Creating package
From upper level directory run the command
```console
sam package --s3-bucket <uri> --template-file templates/pre-annotation-template.yaml
--output-template-file templates/packaged-pre-annotation-template.yaml
```
This command packages the labmda function into a template that can be built
and deployed through SAM deploy command

## Deploying package
From same level run the command
```console
sam deploy --stack-name <stack-name> --template-file templates/packaged-pre-annotation-template.yaml
--capabilities CAPABILITY_IAM
```
This builds and deploys the lambda function to the AWS account linked through
the AWS CLI. It associates it with the stack-name or creates a new stack with
provided name. You can specify a role by providing the --role argument.