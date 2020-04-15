import sqlite3
import pytest
import segmentation_labeling_app.transfers.utils as utils
import json
import boto3
from moto import mock_s3
import pathlib
import numpy as np
import string


@pytest.fixture(scope='module')
def local_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp("files").join("test.txt")
    with open(fn, "w") as fp:
        fp.write("contents")
    yield str(fn)


@pytest.fixture(scope='module')
def local_manifest(tmpdir_factory):
    tdir = tmpdir_factory.mktemp("manifest_files")
    manifest = {}
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


@pytest.mark.parametrize("key", ["abc/temp.csv", None])
def test_upload_file(local_file, bucket, key):
    bucket_path = utils.upload_file(local_file, bucket, key=key)

    if key is None:
        key = pathlib.PurePath(local_file).name
    expected = f"s3://{bucket}/{key}"
    assert bucket_path == expected


@pytest.mark.parametrize("prefix", "abc/datetime")
def test_upload_manifest_contents(
        local_manifest, bucket, prefix):
    s3_manifest = utils.upload_manifest_contents(
            local_manifest,
            bucket,
            prefix)

    vbase = f"s3://{bucket}/{prefix}/"
    expected = {}
    for k, v in local_manifest.items():
        expected[k] = vbase + pathlib.PurePath(v).name

    assert(set(list(s3_manifest.keys())) == set(list(expected.keys())))
    for key in expected.keys():
        assert (s3_manifest[key] == expected[key])


def test_get_manifests_from_db(db_file):
    conn = sqlite3.connect(db_file)
    test_dict = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4"}
    tstr = json.dumps(test_dict)
    for i in range(4):
        conn.execute(
                "INSERT INTO manifest_table (manifest) VALUES ('%s')" % tstr)
    conn.commit()
    conn.close()

    results = utils.get_manifests_from_db(db_file, "manifest_table")
    assert len(results) == 4
    for r in results:
        assert set(list(r.keys())) == set(list(test_dict.keys()))
        for key in test_dict.keys():
            assert r[key] == test_dict[key]

    results = utils.get_manifests_from_db(
            db_file, "manifest_table", "WHERE rowid=2")

    assert len(results) == 1
    for r in results:
        assert set(list(r.keys())) == set(list(test_dict.keys()))
        for key in test_dict.keys():
            assert r[key] == test_dict[key]


def test_manifest_file_from_jsons(tmpdir_factory):
    letters = [i for i in string.ascii_lowercase]

    jsons = []
    for i in range(10):
        jsons.append({})
        for j in range(20):
            key = "".join(np.random.choice(letters, 10))
            value = "".join(np.random.choice(letters, 40))
            jsons[-1][key] = value

    fn = tmpdir_factory.mktemp("manifest_file").join("manifest.jsonl")
    utils.manifest_file_from_jsons(str(fn), jsons)

    with open(str(fn), "r") as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        read_json = json.loads(line)
        assert set(list(read_json.keys())) == set(list((jsons[i].keys())))
        for key in read_json.keys():
            assert read_json[key] == jsons[i][key]
