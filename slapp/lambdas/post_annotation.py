import json
import boto3
from urllib.parse import urlparse


def lambda_handler(event, context) -> dict:
    """
    The lambda function to be run upon the completion of a labeling task
    by a user. Transports the data to an s3 bucket for later use or movement to
    the allen server.
    Args:
        event: dict structure containing input data passed from label task to
        lambda function
        context: context of the event
    Returns: bool status of uploading to bucket
    """
    consolidated_labels = []

    parsed_url = urlparse(event['payload']['s3Uri'])
    s3 = boto3.client('s3')
    textFile = s3.get_object(Bucket=parsed_url.netloc, Key=parsed_url.path[1:])
    filecont = textFile['Body'].read()
    annotations = json.loads(filecont)

    for dataset in annotations:
        for annotation in dataset['annotations']:
            new_annotation = json.loads(
                annotation['annotationData']['content'])
            label = {
                'datasetObjectId': dataset['datasetObjectId'],
                'consolidatedAnnotation': {
                    'content': {
                        event['labelAttributeName']: {
                            'workerId': annotation['workerId'],
                            'result': new_annotation,
                            'labeledContent': dataset['dataObject']
                        }
                    }
                }
            }
            consolidated_labels.append(label)
    return consolidated_labels
