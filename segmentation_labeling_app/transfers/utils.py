import boto3
from botocore.exceptions import ClientError
import pathlib
import sqlite3
import json
import jsonlines
import tempfile


def upload_file(file_name, bucket, key=None):
    """Upload a file to an S3 bucket

    Parameters
    ----------
    file_name: path-like object
        full path to local file
    bucket: str
        name of bucket
    key: str
        object key. If not passed, object key will be
        basename of the file_name

    Returns
    -------
    s3_uri : str
        URI to S3 object just uploaded


    """
    if key is None:
        key = pathlib.PurePath(file_name).name

    client = boto3.client('s3')
    client.upload_file(file_name, bucket, key)

    s3_uri = 's3://' + bucket + '/' + key

    return s3_uri


def object_exists(bucket, key):
    """whether an object exists in an S3 bucket

    Parameters
    ----------
    bucket: str
        name of bucket
    key: str
        object key. If not passed, object key will be
        basename of the file_name

    Returns
    -------
    exists : bool

    """
    exists = False
    client = boto3.client('s3')
    try:
        client.head_object(Bucket=bucket, Key=key)
        exists = True
    except ClientError as exc:
        pass
    return exists


class UploadManifestException(Exception):
    pass


def upload_manifest(file_name, bucket, key, mode=None):
    """upload or update an S3 manifest

    Parameters
    ----------
    file_name: path-like object
        full path to local file
    bucket: str
        name of bucket
    key: str
        object key
    mode : str
        'w' (truncate & write) or 'a' (append) for an existing file

    Returns
    -------
    s3_uri : str
        URI to S3 object just uploaded
    """

    exists = object_exists(bucket, key)

    if exists & (mode not in ['w', 'a']):
        # specify a correct way to handle existing object
        raise UploadManifestException(
                f"s3://{bucket}/{key} exists"
                "specify mode='a' or 'w'")

    if (not exists) | (mode == 'w'):
        # new file, or truncate and write
        s3_uri = upload_file(file_name, bucket, key=key)

    else:
        # append
        tfile = tempfile.NamedTemporaryFile()
        boto3.client('s3').download_file(bucket, key, tfile.name)
        with open(tfile.name, "r") as fp:
            lines = fp.readlines()
        tfile.close()
        with open(file_name, "r") as fp:
            lines.extend(fp.readlines())
        tfile = tempfile.NamedTemporaryFile()
        with open(tfile.name, "w") as fp:
            fp.writelines(lines)
        s3_uri = upload_file(tfile.name, bucket, key=key)
        tfile.close()

    return s3_uri


def upload_manifest_contents(local_manifest, bucket, prefix):
    """upload the contents of a manifest, returning a copy with
    updated S3 URIs

    Parameters
    ----------
    local_manifest: dict
        keys are manifest content names and values are path-like objects
        to local files
    bucket: str
        name of s3 bucket
    prefix: str
        prefix for object keys

    Returns
    -------
    s3_manifest: dict
        keys are manifest content names and values are s3 URIs to uploaded
        files

    """
    s3_manifest = {}
    for k, v in local_manifest.items():
        object_key = prefix + "/" + pathlib.PurePath(v).name
        s3_manifest[k] = upload_file(v, bucket, object_key)

    return s3_manifest


def get_manifests_from_db(sqlite_db_path, sql_table, sql_filter=""):
    """get back a list of manifests, as jsons

    Parameters
    ----------
    sqlite_db_path: path-like object
        passed to sqlite3.connect()
    sql_table: str
        table to query
    sql_filter: str
        "portion of a SQL query starting with WHERE for filtering"

    Returns
    -------
    results: List
        each element of the list is a manifest dictionary

    """
    conn = sqlite3.connect(sqlite_db_path)
    query_string = f"SELECT manifest FROM {sql_table} {sql_filter}"
    results = [json.loads(result[0]) for result in conn.execute(query_string)]
    conn.close()
    return results


def manifest_file_from_jsons(filepath, manifest_jsons):
    """write a jsonlines format file from a list of dicts

    Parameters
    ----------
    filepath : path-like object
        destination path for output file
    manifest_jsons : list
        list of dictionaries

    """
    with open(filepath, "w") as fp:
        w = jsonlines.Writer(fp)
        w.write_all(manifest_jsons)
