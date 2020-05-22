import botocore.session
import botocore.config
import pathlib
import jsonlines
from typing import Union, List, TypedDict, Tuple
import hashlib
import base64


def s3_uri(bucket, key):
    uri = 's3://' + bucket + '/' + key
    return uri


def get_checksum(file_name: Union[pathlib.Path, str]) -> str:
    """returns the base64-encoded 128-bit MD5 digest of the file contents

    Parameters
    ----------
    file_name: path-like object
       contents will be checksummed

    Returns
    -------
    checksum: str
        base64-encoded 128-bit MD5 digest

    """
    hash_object = hashlib.md5(open(file_name, "rb").read())
    checksum = base64.b64encode(hash_object.digest()).decode('utf-8')
    return checksum


class ConfiguredUploadClient():
    """
    Notes
    -----
    it is not possible to inherit from the botocore.client.S3 class
    because it does not exist until runtime
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/events.html#extensibility-guide # noqa
    """
    def __init__(self, *args, **kwargs):
        session = botocore.session.get_session()
        config = botocore.config.Config(*args, **kwargs)
        self.client = session.create_client('s3', config=config)

    def put_object(self, *args, **kwargs):
        return self.client.put_object(*args, **kwargs)


class UploadResult(TypedDict):
    file_name: Union[pathlib.Path, str]
    bucket: str
    key: str
    reponse: dict


def upload_file(
        client: Union[ConfiguredUploadClient, botocore.client.BaseClient],
        file_name: Union[pathlib.Path, str],
        bucket: str,
        key: str) -> UploadResult:
    """Upload a file to an S3 bucket

    Parameters
    ----------
    client: ConfiguredUploadClient
        has a put_object() method
    file_name: path-like object
        full path to local file
    bucket: str
        name of bucket
    key: str
        object key.

    Returns
    -------
    result : UploadResult
        contains the `file_name`, the destination `bucket` and `key`
        and the server `response`

    """
    checksum = get_checksum(file_name)
    with open(file_name, 'rb') as fp:
        response = client.put_object(
                Body=fp,
                Bucket=bucket,
                Key=key,
                ContentMD5=checksum)

    result = {
            'file_name': file_name,
            'bucket': bucket,
            'key': key,
            'response': response
            }

    return result


class UploadFileArgs(TypedDict):
    file_name: Union[pathlib.Path, str]
    bucket: str
    key: str


def upload_files(
        client: Union[ConfiguredUploadClient, botocore.client.BaseClient],
        upload_file_args: List[UploadFileArgs]) -> List[UploadResult]:
    """Uploads a list of files to an S3 bucket. Can be useful for parallelizing
    with clients.

    Parameters
    ----------
    client: ConfiguredUploadClient
        has a put_object() method
    upload_file_args: list of UploadFileArgs

    Returns
    -------
    results : list of UploadResult
        each entry contains the `file_name`, the destination `bucket` and
        `key` and the server `response`

    """
    results = []
    for args in upload_file_args:
        results.append(upload_file(client, **args))

    return results


def upload_manifest_contents(
        client: Union[ConfiguredUploadClient, botocore.client.BaseClient],
        local_manifest: dict, bucket: str, prefix: str,
        skip_keys: List = []) -> Tuple[dict, List[UploadResult]]:
    """upload the contents of a manifest, returning a copy with
    updated S3 URIs

    Parameters
    ----------
    client: ConfiguredUploadClient
        has a put_object() method
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
        objects
    responses: list
        list of upload_file responses

    """
    dict_keys = []
    s3_manifest = {}
    upload_file_args = []
    for k, v in local_manifest.items():
        if k in skip_keys:
            continue
        if k in ['experiment-id', 'roi-id']:
            s3_manifest[k] = v
        else:
            upload_file_args.append(
                    {
                        'file_name': v,
                        'bucket': bucket,
                        'key': prefix + "/" + pathlib.PurePath(v).name
                        })
            dict_keys.append(k)

    responses = upload_files(client, upload_file_args)
    for k, r in zip(dict_keys, responses):
        s3_manifest[k] = s3_uri(r['bucket'], r['key'])

    return s3_manifest, responses


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
