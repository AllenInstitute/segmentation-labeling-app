import json
import pytest
from pathlib import Path
import sys
import segmentation_labeling_app.pre_annotation.pre_annotation_lambda as pre_annotation_lambda


@pytest.fixture()
def payload_fixture(request):
    default = {
        "version": "2018-10-16",
        "labelingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/example-job",
        "dataObject": {
            "source-ref": "dummy_uri",
            "video-source-ref": "dummy_uri",
            "max-source-ref": "dummy_uri",
            "avg-source-ref": "dummy_uri",
            "trace-source-ref": "dummy_uri",
            "roi-data-source-ref": "dummy_uri",
            "roi-id": 1,
            "experiment-id": 1}
    }
    default.update(request.param)
    return default


@pytest.mark.parametrize("payload_fixture", [({})],
                         indirect=["payload_fixture"])
def test_pre_annotation_lambda(payload_fixture):
    payload = pre_annotation_lambda.lambda_handler(payload_fixture, None)

    assert payload['taskInput'] == payload_fixture['dataObject']
