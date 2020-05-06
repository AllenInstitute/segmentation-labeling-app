[![CircleCI](https://circleci.com/gh/AllenInstitute/segmentation-labeling-app.svg?style=svg)](https://circleci.com/gh/AllenInstitute/segmentation-labeling-app)
[![codecov](https://codecov.io/gh/AllenInstitute/segmentation-labeling-app/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenInstitute/segmentation-labeling-app)

# segmentation-labeling-app
This repository contains the required infrastructure and data transformations to deploy a custom labeling workflow using AWS Sagemaker Ground Truth. It is currently in active development. The community is welcome to use this repository as a template or resource for creating Sagemaker Ground Truth applications.

## Level of support
We are not currently supporting this code, but simply releasing it to the community AS IS. We are not able to provide any guarantees of support.  The community may submit issues, but you should not expect an active response.

## Contributing
This tool is important for internal use at the Allen Institute. Because it's designed for internal needs, we are not anticipating external contributions. Pull requests may not be accepted if they conflict with our existing plans.

# Steps to deploy on Sagemaker Ground Truth

## Deploy Lambdas
From the top level directory, run:

```console
sam build
sam deploy
```

This will deploy the lambda stack with the following values:
```
Stack name                 : gt-lambdas
Region                     : us-west-2
Confirm changeset          : True
Deployment s3 bucket       : aws-sam-cli-managed-default-samclisourcebucket-1jbjkzibjwsu5
Capabilities               : ["CAPABILITY_IAM"]
```

To change these default values, see the documentation for `sam deploy`,
or use `sam deploy --guided` to deploy in guided mode.

## Create SageMaker Ground Truth Job

### Create Private Workforce
If you want to use a private labelling workforce (that doesn't already exist), you'll need to create it ahead of time. The easist way to do this is through the console.

1. Access SageMaker through the AWS Console
2. Ground Truth > Labeling Workforces > Private > Create private team
3. Follow the instructions in the console to finish team creation

Once you create the team, you'll need to add workers to it. 

1. Access SageMaker through the AWS Console
2. Ground Truth > Labeling Workforces > Private
3. Under Workers, click "Invite new workers"
4. Add the email addresses of the workers you want to invite
5. Go back to SageMaker > Ground Truth > Labeling Workforces > Private
6. Click on the name of the team you want to add the user to, under Private Teams
7. Under the summary, click on the Workers tab
8. Select "Add workers to team". You should see the users you invited in step 4.


### Collect Input Data
Prior to creating a labeling job, ensure the following:
1. All data required for the labeling job are uploaded to an s3 bucket 
2. A json manifest file in the [proper format](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-data-input.html) specifies the inputs to be labeled is uploaded to s3
3. If you want to save your annotation data in a separate bucket, ensure that bucket is created.

Note that the region of the Sagemaker GT Job and S3 buckets must be the same.

### Create Job via Console
1. Access SageMaker through the AWS Console
2. Ground Truth > Labeling Jobs > Create Labeling Job
3. Follow the prompts to provide your input data manifest, output data location, etc.
4. For Task Type, choose "Custom" from the dropdown menu
5. Click "next"
6. Under "Workers", click "Private"
7. Select your private team from the dropdown menu below, and follow the rest of the instructions for timeout and workers per task
8. Under Templates, "Custom" should be selected from the dropdown menu, and you should see a form field with some boilerplate html code
9. Copy the contents of `slapp/app/index.html` into the form box
10. Select your Lambda functions. If you used the default values for deployment during the [deploy lambdas](##deploy-lambdas) step, then the relevant Lambdas should be named "gt-lambdas-PreAnnotationLabelingFunction-\<id\>"
and "gt-lambdas-PostAnnotationLabelingFunction-\<id\>".
11. Click "Create"

# User Interface

The user interface contains the following features:
* 128x128 px views of the ROI, including:
    - Full playable 2p recording (downsampled to 4Hz)
    - Maximum projection image
    - Average projection image
    - ROI "postal stamp"
* Toggle-able ROI overlays, including:
    - Weighted mask
    - Outline
    - None
* Interactive chart of 2p recording trace with the following features:
    - Select and zoom points by clicking and dragging
    - Select and navigate points with navigation bar
    - Play 2p recording for the selected points
    - Skip to a point in movie by selecting point in trace
* Adjust 2p recording (using CSS Filters)
    - [Contrast](https://developer.mozilla.org/en-US/docs/Web/CSS/filter-function/brightness)
    - [Brightness](https://developer.mozilla.org/en-US/docs/Web/CSS/filter-function/contrast)
* Zoom in views on mouse hover
* Colorblind mode
