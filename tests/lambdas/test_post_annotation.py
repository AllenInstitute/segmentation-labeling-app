import pytest
from pathlib import Path
import boto3
from moto import mock_s3

from slapp.lambdas.post_annotation import lambda_handler as post_lambda


consolidation_request = {
    "version": "2018-10-06",
    "labelingJobArn": "arn:aws:sagemaker:us-west-2:111111111111:labeling-job",
    "payload": {
       "s3Uri": "s3://test-bucket/consolidation_payload.json"
    },
    "labelAttributeName": "test-label-job",
    "roleArn": "awn:aws:us-west-2:111111111111:labeling-job"
 }

consolidation_payload = str(
    Path(__file__).parent / "resources" / "consolidation_payload.json")


@pytest.fixture
def s3_bucket(scope="function"):
    mock = mock_s3()
    mock.start()
    s3 = boto3.resource("s3")
    s3.create_bucket(Bucket="test-bucket")
    s3.meta.client.upload_file(consolidation_payload, "test-bucket",
                               "consolidation_payload.json")
    yield
    mock.stop()


def test_post_annotation_lambda(s3_bucket):
    expected = [
        {
            "datasetObjectId": "1",
            "consolidatedAnnotation": {
                "content": {
                    "test-label-job": {
                        "sourceData": "s3://test-bucket/test-img.png",
                        "majorityLabel": "cell",
                        "workerAnnotations": [
                            {"workerId": "private.us-west-2.11111",
                             "roiLabel": "not cell"},
                            {"workerId": "private.us-west-2.22222",
                             "roiLabel": "cell"},
                            {"workerId": "private.us-west-2.33333",
                             "roiLabel": "cell"},
                        ]
                    }
                }
            }
        },
        {
            "datasetObjectId": "3",
            "consolidatedAnnotation": {
                "content": {
                    "test-label-job": {
                        "sourceData": "s3://test-bucket/test-img-2.png",
                        "majorityLabel": "not cell",
                        "workerAnnotations": [
                            {"workerId": "private.us-west-2.11111",
                             "roiLabel": "not cell"},
                            {"workerId": "private.us-west-2.22222",
                             "roiLabel": "not cell"},
                            {"workerId": "private.us-west-2.33333",
                             "roiLabel": "not cell"},
                        ]
                    }
                }
            }
        },
    ]
    actual = post_lambda(consolidation_request, None)
    assert expected == actual
