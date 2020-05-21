import boto3
from botocore.exceptions import ClientError
import pathlib
import jsonlines
import tempfile
import filecmp
import time


class S3TransferException(Exception):
    pass


def s3_uri(bucket, key):
    uri = 's3://' + bucket + '/' + key
    return uri


def object_exists(bucket, key, max_retries=5, delay=2.0):
    """whether an object exists in an S3 bucket

    Parameters
    ----------
    bucket: str
        name of bucket
    key: str
        object key.
    max_retries: int
        max number of retries, in case of s3 latency
    delay: float
        number of seconds to wait between retries

    Returns
    -------
    exists : bool

    """
    exists = False
    client = boto3.client('s3')
    for i in range(max_retries):
        try:
            client.head_object(Bucket=bucket, Key=key)
            exists = True
            break
        except ClientError:
            pass
        time.sleep(delay)
    return exists


def local_s3_compare(file_name, bucket, key):
    uri = s3_uri(bucket, key)
    client = boto3.client('s3')

    try:
        assert object_exists(bucket, key)
    except AssertionError:
        raise S3TransferException(f"{uri} does not exist")
    try:
        tfile = tempfile.NamedTemporaryFile()
        client.download_file(bucket, key, tfile.name)
        assert filecmp.cmp(file_name, tfile.name)
    except AssertionError:
        raise S3TransferException(
                f"destination object {uri} does not match "
                f"source file {file_name}")
    finally:
        tfile.close()

    return True


def upload_file(file_name, bucket, key=None, roundtrip=False):
    """Upload a file to an S3 bucket and perform either an object
    exists validation or a roundtrip file comparison validation

    Parameters
    ----------
    file_name: path-like object
        full path to local file
    bucket: str
        name of bucket
    key: str
        object key. If not passed, object key will be
        basename of the file_name
    roundtrip: bool
        whether to perform a roundtrip download and comparison with
        the original file

    Returns
    -------
    uri : str
        URI to S3 object just uploaded

    """
    if key is None:
        key = pathlib.PurePath(file_name).name

    # if the object already exists, the upload will be an overwrite.
    exists = object_exists(bucket, key)

    client = boto3.client('s3')
    client.upload_file(file_name, bucket, key)
    uri = s3_uri(bucket, key)

    if exists | roundtrip:
        # slow for large files
        assert local_s3_compare(file_name, bucket, key)
    else:
        # can be fast, has some retries and delays
        try:
            assert object_exists(bucket, key)
        except AssertionError:
            raise S3TransferException(f"{bucket}/{key} does not exist")

    return uri


def upload_manifest_contents(local_manifest, bucket, prefix, skip_keys=[]):
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
    skip_keys: list
        skip the upload for these keys. Used to not duplicate upload of
        objects common to many manifests.

    Returns
    -------
    s3_manifest: dict
        keys are manifest content names and values are s3 URIs to uploaded
        files

    """
    s3_manifest = {}
    for k, v in local_manifest.items():
        if k in skip_keys:
            continue
        if k in ['experiment-id', 'roi-id']:
            s3_manifest[k] = v
        else:
            object_key = prefix + "/" + pathlib.PurePath(v).name
            s3_manifest[k] = upload_file(v, bucket, object_key)

    return s3_manifest


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
