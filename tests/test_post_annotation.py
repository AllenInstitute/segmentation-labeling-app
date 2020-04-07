from pathlib import Path
import os
import pytest
import json
from urllib.parse import urlparse

import boto3
from moto import (mock_s3)

import segmentation_labeling_app.post_annotation.post_annotation_lambda as post_annotation_lambda


@pytest.fixture()
def json_bucket_fixture():
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
    test_path = Path(__file__).parent
    bucket_item_path = test_path / 'test_resources'
    os.mkdir(bucket_item_path)
    bucket_item_path = bucket_item_path / 'test_bucket.txt'
    with open(bucket_item_path, 'w') as open_file:
        json.dump(default_data, open_file)
        open_file.close()

    yield bucket_item_path.as_posix()


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
@pytest.mark.parametrize("json_bucket_fixture", [({})],
                         indirect=["json_bucket_fixture"])
def test_post_annotation_lambda(json_bucket_fixture):
    file_url = '//mock_bucket/' + str(json_bucket_fixture)
    payload_update = {"s3Uri": file_url}
    payload_fix = payload_fixture({"payload": payload_update})

    client = boto3.client('s3')
    parsed_url = urlparse(file_url)
    client.create_bucket(Bucket=parsed_url.netloc)
    client.upload_file(Filename=parsed_url.path[1:], Bucket=parsed_url.netloc,
                       Key=parsed_url.path[1:])

    with open(parsed_url.path[1:], 'r') as open_json:
        json_data = json.load(open_json)
        dataset = json_data[0]
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

    os.remove(Path(parsed_url.path[1:]))
    os.rmdir(Path(parsed_url.path[1:]).parent)
