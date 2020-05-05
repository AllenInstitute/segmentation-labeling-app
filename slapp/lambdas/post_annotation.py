import json
import boto3
from urllib.parse import urlparse
import math
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context) -> dict:
    """
    The lambda function to be run upon the completion of a labeling task
    by a user. Transports the data to an s3 bucket for later use or movement to
    the allen server.
    Parameters
    ----------
        event: dict
            Json-formatted event from Sagemaker Ground Truth
            Event doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html    # noqa
        context: object
            Lambda Context runtime methods and attributes
            Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html    #noqa

    Returns
    -------
        Json-formatted list of consolidated labels, with majority
        label calculated if it exists.
        [
            {
                "datasetObjectId": <id>,
                "consolidatedAnnotation": {
                    "content" : {
                        <labelAttributeName> : {
                            "sourceData": <source>,
                            "majorityLabel": <majority label>,
                            "workerAnnotations": [
                                {"workerId": <worker_id>, "roiLabel": <label>},
                                ...
                            ]
                        }
                    }
                }
            },
            ...
        ]
        Return doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html    # noqa
    """
    # For CloudWatch Logs
    logger.info("#### EVENT ####")
    logger.info(json.dumps(event))
    consolidated_labels = []
    label_attribute_name = event["labelAttributeName"]
    parsed_url = urlparse(event['payload']['s3Uri'])
    s3 = boto3.client('s3')
    textFile = s3.get_object(Bucket=parsed_url.netloc, Key=parsed_url.path[1:])
    filecont = textFile['Body'].read()
    annotations = json.loads(filecont)
    for dataset in annotations:
        consolidated_dataset = {
            "datasetObjectId": dataset["datasetObjectId"]
        }
        worker_annotations = []
        worker_labels = []
        # Consolidate list of annotations
        for labels in dataset["annotations"]:
            label_dict = json.loads(
                labels.get("annotationData").get("content")).get("roiLabel")
            if label_dict is not None:
                label = label_dict["label"]
            else:
                label = None
            worker_annotations.append(
                {"workerId": labels["workerId"], "roiLabel": label})
            if label == "cell":
                worker_labels.append(1)
            elif label == "not cell":
                worker_labels.append(0)
            else:    # possible None if timeout
                pass
        majority = compute_majority(worker_labels)
        consolidated_dataset.update({
            "consolidatedAnnotation": {
                "content": {
                    label_attribute_name: {
                        "sourceData": dataset["dataObject"]["s3Uri"],
                        "majorityLabel": majority,
                        "workerAnnotations": worker_annotations
                    }
                }
            }
        })
        consolidated_labels.append(consolidated_dataset)
    logger.info("### CONSOLIDATED LABELS ###")
    logger.info(json.dumps(consolidated_labels))
    return consolidated_labels


def compute_majority(labels):
    label_count = len(labels)
    threshold = math.ceil(label_count/2)
    yeas = sum(labels)
    nays = len(labels) - sum(labels)
    if yeas >= threshold:
        majority = "cell"
    elif nays >= threshold:
        majority = "not cell"
    else:
        majority = None
    return majority
