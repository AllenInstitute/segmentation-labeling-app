import json


def lambda_handler(event, context):
    """
    Lambda function tied to new task in the AWS ground truth workflow.
    Args:
        event: The data coming from the triggering event
        context: The context for the event, not used by lambda

    Returns:
        Dictionary with taskInput as key and manifest dictionary as value
    """
    return {
        "taskInput": event['dataObject']
    }
