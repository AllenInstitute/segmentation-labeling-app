import sqlite3
import pytest
import slapp.transfers.utils as utils
import json
import boto3
from moto import mock_s3
import pathlib
import numpy as np
import string


@mock_s3
def test_s3_get_object():
    # Set up the fake bucket and object
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")
    body = b'{"a": 1}\n{"b": 2}'
    s3.put_object(Bucket="mybucket", Key="my/file.json",
                  Body=body)
    # Run the test
    response = utils.s3_get_object("s3://mybucket/my/file.json")
    assert body == response["Body"].read()


@pytest.mark.parametrize(
    "body, expected",
    [
        (b'{"a": 1, "b": 3}\n{"b": 2}', [{"a": 1, "b": 3}, {"b": 2}]),
        (b'{"a": 1}', [{"a": 1}]),
        (b'', []),
    ]
)
def test_read_jsonlines_file(tmp_path, body, expected):
    with open(tmp_path / "filename", "wb") as f:
        f.write(body)
    reader = utils.read_jsonlines(tmp_path / "filename")
    response = []
    for record in reader:
        response.append(record)
    assert expected == response


@mock_s3
@pytest.mark.parametrize(
    "body, expected",
    [
        (b'{"a": 1, "b": 3}\n{"b": 2}', [{"a": 1, "b": 3}, {"b": 2}]),
        (b'{"a": 1}', [{"a": 1}]),
        (b'', []),
    ]
)
def test_read_jsonlines_s3(body, expected):
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")
    s3.put_object(Bucket="mybucket", Key="my/file.json",
                  Body=body)
    reader = utils.read_jsonlines("s3://mybucket/my/file.json")
    response = []
    for record in reader:
        response.append(record)
    assert expected == response


@pytest.fixture(scope='module')
def local_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp("files").join("test.txt")
    with open(fn, "w") as fp:
        fp.write("contents")
    yield str(fn)


@pytest.fixture(scope='module')
def local_files(tmpdir_factory):
    fdir = tmpdir_factory.mktemp("files")
    files = []
    for i in range(5):
        fn = fdir.join(f"test_multiple_{i}.txt")
        with open(fn, "w") as fp:
            fp.write("contents")
        files.append(str(fn))
    yield files


@pytest.fixture(scope='module')
def local_manifest(tmpdir_factory):
    tdir = tmpdir_factory.mktemp("manifest_files")
    manifest = {}
    manifest['experiment-id'] = 12345
    manifest['roi-id'] = 98765
    for i in range(3):
        fname = tdir.join(f"test{i}.txt")
        with open(fname, "w") as fp:
            fp.write("contents")
        manifest[f"key{i}"] = str(fname)
    yield manifest


@pytest.fixture(scope='module')
def bucket():
    with mock_s3():
        bucket_name = 'mybucket'
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.fixture(scope='function')
def db_file(tmpdir_factory):
    db = tmpdir_factory.mktemp("upload").join("mydatabase.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE manifest_table (manifest text)")
    conn.close
    yield db


@pytest.fixture(scope='module')
def list_of_dicts():
    letters = [i for i in string.ascii_lowercase]
    jsons = []
    for _ in range(10):
        jsons.append({})
        for _ in range(20):
            key = "".join(np.random.choice(letters, 10))
            value = "".join(np.random.choice(letters, 40))
            jsons[-1][key] = value
    yield jsons


@pytest.mark.parametrize("key", ["abc/temp.csv"])
def test_upload_file(local_file, bucket, key):
    client = utils.ConfiguredUploadClient()
    result = utils.upload_file(client, local_file, bucket, key=key)

    assert result['file_name'] == local_file
    assert result['bucket'] == bucket
    assert result['key'] == key
    assert isinstance(result['response'], dict)
    assert result['response']['ResponseMetadata']['HTTPStatusCode'] == 200


def test_upload_files(local_files, bucket):
    client = utils.ConfiguredUploadClient()
    upload_file_args = []
    keys = []
    for local_file in local_files:
        keys.append("some/prefix/" + pathlib.PurePath(local_file).name)
        upload_file_args.append(
                {
                    'file_name': local_file,
                    'bucket': bucket,
                    'key': keys[-1]
                    })
    results = utils.upload_files(client, upload_file_args)

    for local_file, key, result in zip(local_files, keys, results):
        print(result)
        assert result['file_name'] == local_file
        assert result['bucket'] == bucket
        assert result['key'] == key
        assert isinstance(result['response'], dict)
        assert result['response']['ResponseMetadata']['HTTPStatusCode'] == 200


@pytest.mark.parametrize("skip_keys", [[], ["key0"]])
@pytest.mark.parametrize("prefix", ["abc/datetime"])
def test_upload_manifest_contents(
        local_manifest, bucket, prefix, skip_keys):
    client = utils.ConfiguredUploadClient()
    s3_manifest, responses = utils.upload_manifest_contents(
            client,
            local_manifest,
            bucket,
            prefix,
            skip_keys=skip_keys)

    vbase = f"s3://{bucket}/{prefix}/"
    expected = {}
    for k, v in local_manifest.items():
        if k in ['experiment-id', 'roi-id']:
            expected[k] = v
        else:
            expected[k] = vbase + pathlib.PurePath(v).name

    for sk in skip_keys:
        assert sk not in s3_manifest.keys()
        expected.pop(sk)

    assert(set(list(s3_manifest.keys())) == set(list(expected.keys())))
    for key in expected.keys():
        assert (s3_manifest[key] == expected[key])

    uploaded_objs = {f"s3://{r['bucket']}/{r['key']}" for r in responses}
    manifest_objs = {v for k, v in s3_manifest.items()
                     if k not in ['experiment-id', 'roi-id']}

    assert uploaded_objs == manifest_objs

    for r in responses:
        assert isinstance(r, dict)
        assert r['response']['ResponseMetadata']['HTTPStatusCode'] == 200


def test_manifest_file_from_jsons(list_of_dicts, tmpdir_factory):
    fn = tmpdir_factory.mktemp("manifest_file").join("manifest.jsonl")
    utils.manifest_file_from_jsons(str(fn), list_of_dicts)

    with open(str(fn), "r") as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        read_json = json.loads(line)
        assert (
                set(list(read_json.keys())) ==
                set(list((list_of_dicts[i].keys()))))
        for key in read_json.keys():
            assert read_json[key] == list_of_dicts[i][key]
