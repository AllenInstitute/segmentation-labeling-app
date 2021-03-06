from typing import List, Union, Optional
from pathlib import Path
import copy
import jsonlines
import boto3
from urllib.parse import urlparse
import tempfile
import shutil
import logging
import json
import sys

from slapp.transfers.utils import read_jsonlines
from slapp.lambdas.post_annotation import compute_majority

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class MergingException(Exception):
    pass


class WorkerAnnotation(TypedDict):
    workerId: str
    roiLabel: str


class Project(TypedDict):
    sourceData: str
    majorityLabel: str
    workerAnnotations: List[WorkerAnnotation]


def get_project_key(record: dict) -> Union[str, None]:
    """returns the key name for the project, performs some validation.
    See Notes.

    Parameters
    ----------
    record: dict
        the dictionary provided

    Returns
    -------
    project_key: str
       the key containing the project sub-dict or None if validation fails

    Notes
    -----
    SagemakerGroundTruth returns a dictionary where 2 keys "<project>" and
    "<project>-metadata" are dynamically named by the name of the launched
    labeling job.

    """
    # check that is identified
    if 'roi-id' not in record:
        raise MergingException("record does not contain key 'roi-id'")

    # check that it conforms to
    # https://docs.aws.amazon.com/sagemaker/latest/dg/sms-data-output.html
    candidates = [k for k in record.keys() if '-metadata' in k]
    if len(candidates) != 1:
        raise MergingException(f"Record for roi-id {record['roi-id']} has "
                               f"{len(candidates)} keys containing "
                               "'-metadata' expecting 1 and only 1")

    meta_key = candidates[0]
    project_key = meta_key.replace("-metadata", "")

    # check that there is a project entry related to the metadata
    if project_key not in record:
        project_key = None

    return project_key


def merge_projects(project1: Project, project2: Project) -> Project:
    """merges worker annotations between 2 record project entries and updates
    majorityLabel

    Parameters
    ----------
    project1: Project
        first project to merge
    project2: Project
        second project to merge

    Returns
    -------
    new_project: Project
        new with workerAnnotations concatenated and sourceData appended
        if different. Any other entities inherited from project1

    """
    new_project = Project(copy.deepcopy(project1))
    for annotation in project2['workerAnnotations']:
        if annotation not in new_project['workerAnnotations']:
            new_project['workerAnnotations'].append(annotation)
    if new_project['sourceData'] != project2['sourceData']:
        new_project['sourceData'] += ',' + project2['sourceData']
    return new_project


def merge_records(record1: dict, record2: dict) -> dict:
    """merge 2 records, maintaining project and job key names from
    record1

    Parameters
    ----------
    record1: dict
        first record to merge
    record2: dict
        second record to merge

    Returns
    -------
    new_record: dict
        merged record. new_record['Project'] is merged. Everything else
        inherits from record1.

    """
    new_record = copy.deepcopy(record1)

    # merge the annotations
    pk1 = get_project_key(record1)
    pk2 = get_project_key(record2)
    new_project = merge_projects(record1[pk1], record2[pk2])
    new_record[pk1] = new_project

    return new_record


def merge_outputs(src_uris: List[Union[str, Path]],
                  dst_uri: Optional[Union[str, Path]] = None,
                  new_project_key: Optional[str] = "merged-project",
                  new_job_name: Optional[str] = "merged-job",
                  exact_nlabels: Optional[int] = 3) -> dict:
    """merge outputs from multiple labeling jobs

    Parameters
    ----------
    src_uris: list of s3 uris or local filepaths
        source labeling job outputs to merge
    dst_uri: s3 uri or local filepath
        destination uri. If none (default), not written.
    new_project_key: str
        merged project key in returned records will be this str
    new_job_name: str
        job name in returned records will be this str
    exact_nlabels: int
        if the number of labels is not this number, the majority
        label will be set to None

    Returns
    -------
    merged: list of records

    """

    # merge all the records
    htable = {}
    nskipped = {}
    nused = {}
    for src_uri in src_uris:
        reader = read_jsonlines(src_uri)
        nskipped[str(src_uri)] = 0
        nused[str(src_uri)] = 0
        for record in reader:
            # some light validation on every record
            pkey = get_project_key(record)
            if pkey is None:
                nskipped[str(src_uri)] += 1
                continue
            else:
                nused[str(src_uri)] += 1

            # if already in the hash table, merge
            if record['roi-id'] in htable:
                record = merge_records(htable[record['roi-id']], record)

            # homogenize key names across the output
            pkey = get_project_key(record)
            record[new_project_key] = record.pop(pkey)
            mkey = pkey + '-metadata'
            new_meta_key = new_project_key + '-metadata'
            record[new_meta_key] = record.pop(mkey)
            record[new_meta_key]['job-name'] = new_job_name

            # set the hash value
            htable[record['roi-id']] = record

    merged = list(htable.values())

    if max(nskipped.values()) > 0:
        logging.warning("skipped some records "
                        f"{json.dumps(nskipped, indent=2)}")
    logging.info(f"n records used: {json.dumps(nused, indent=2)}")

    # set the majority label
    nvalid = 0
    # translate back to the inputs compute_majority expects
    translator = {"cell": 1, "not cell": 0}
    for record in merged:
        labels = [i['roiLabel']
                  for i in record[new_project_key]['workerAnnotations']]
        labels = [translator[i] for i in labels]
        majority = compute_majority(labels, exact_len=exact_nlabels)
        record[new_project_key]['majorityLabel'] = majority
        if majority is not None:
            nvalid += 1

    if dst_uri is not None:
        with tempfile.NamedTemporaryFile() as tmpout:
            with open(tmpout.name, "w") as fp:
                jsonlines.Writer(fp).write_all(merged)

            if str(dst_uri).startswith("s3://"):
                s3 = boto3.client("s3")
                parsed_s3 = urlparse(dst_uri)
                bucket = parsed_s3.netloc
                file_key = parsed_s3.path[1:]
                s3.upload_file(tmpout.name, bucket, file_key)
            else:
                shutil.copyfile(tmpout.name, dst_uri)
        logging.info(f"wrote {dst_uri} with {len(merged)} records "
                     f"and {nvalid} valid majorities.")

    return merged
