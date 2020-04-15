from pathlib import Path
import os
import pytest
import json
from urllib.parse import urlparse

import boto3
from moto import (mock_s3)

import segmentation_labeling_app.post_annotation.post_annotation_lambda as post_annotation_lambda


@pytest.fixture()
def json_fixture():
    default_data = [
        {
            "datasetObjectId": 'dummy_id',
            "dataObject":
        {
            "s3Uri": "dummy_id",
            "content": "dummy"
        },
            "annotations":
            [{
                "workerId": "dummy",
                "annotationData":
                {
                    "content": '{ "name": "John", "age": 30, "car": "None" }',
                    "s3Uri": "dummy"
                }
            }]
        }
    ]
    yield default_data


def payload_fixture(updates: dict):
    default = {
        "version": "2018-10-16",
        "labelingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/example-job",
        "labelCategories": ['True', 'False', 'Unknown'],
        "labelAttributeName": 'Valid Cell',
        "roleArn": "string",
        "payload": {
                       "s3Uri": "file:///"
        }
    }
    for key, value in updates.items():
        default[key] = value
    return default


@mock_s3
@pytest.mark.parametrize("json_fixture", [({})],
                         indirect=["json_fixture"])
def test_post_annotation_lambda(json_fixture):
    payload_update = {"s3Uri": "s3://test_bucket/key/file.txt"}
    payload_fix = payload_fixture({"payload": payload_update})

    client = boto3.client('s3')
    client.create_bucket(Bucket="test_bucket")
    client.put_object(Bucket="test_bucket", Key="key/file.txt", Body=json.dumps(json_fixture))

    dataset = json_fixture[0]
    annotation = dataset['annotations'][0]
    new_annotation = json.loads(annotation['annotationData']['content'])


    expected_response = {
        'datasetObjectId': dataset['datasetObjectId'],
        'consolidatedAnnotation': {
            'content': {
                payload_fix['labelAttributeName']: {
                    'workerId': annotation['workerId'],
                    'result': new_annotation,
                    'labeledContent': dataset['dataObject']
                }
            }
        }
    }

    consolidate_response = post_annotation_lambda.lambda_handler(event=payload_fix, context=None)
    assert expected_response == consolidate_response[0]
