import sqlite3
import pytest
import json
import boto3
from moto import mock_s3
import os
import segmentation_labeling_app.transfers.upload as up


@pytest.fixture(scope='module')
def db_file(tmpdir_factory):
    db = tmpdir_factory.mktemp("upload").join("mydatabase.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE manifest_table (manifest text)")
    conn.close
    yield db


@pytest.fixture(scope='module')
def populated_db(db_file, tmpdir_factory):
    tdir = tmpdir_factory.mktemp("local_files")
    test_dicts = []
    for j in range(4):
        test_dict = {}
        for i in range(4):
            fn = tdir.join(f"test_{i}_{j}.txt")
            with open(fn, "w") as fp:
                fp.write("contents")
            test_dict[f"key{i}"] = str(fn)
        test_dicts.append(test_dict)

    conn = sqlite3.connect(db_file)
    tstrs = [json.dumps(test_dict) for test_dict in test_dicts]
    for tstr in tstrs:
        conn.execute(
                "INSERT INTO manifest_table (manifest) VALUES ('%s')" % tstr)
    conn.commit()
    conn.close()
    yield db_file


@pytest.fixture(scope='function')
def bucket():
    with mock_s3():
        bucket_name = 'mybucket'
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.mark.parametrize("timestamp", [True, False])
def test_LabelDataUploader(populated_db, bucket, timestamp):
    args = {
            'sqlite_db_file': str(populated_db),
            's3_bucket_name': bucket,
            'sql_table': 'manifest_table',
            'sql_filter': "",
            'timestamp': timestamp,
            'prefix': 'abc/def',
            }
    ldu = up.LabelDataUploader(input_data=args, args=[])
    ldu.run()

    # get the contents of the db_file
    conn = sqlite3.connect(populated_db)
    response = conn.execute(f"SELECT manifest FROM {args['sql_table']}")
    files_in_db = []
    for r in response.fetchall():
        d = json.loads(r[0])
        for key in d:
            files_in_db.append(os.path.basename(d[key]))
    conn.close()

    # get what is in the bucket (function scoped)
    response = boto3.client('s3').list_objects_v2(Bucket=bucket)
    files_in_s3 = [c['Key'] for c in response['Contents']]

    # make sure we read whole contents
    assert not response['IsTruncated']

    # s3 should have all the files, + 1 manifest
    assert len(files_in_s3) == (len(files_in_db) + 1)

    # s3 should have every file that the db has
    s3_basenames = [
            os.path.basename(f)
            for f in files_in_s3
            if 'manifest' not in f]
    assert set(s3_basenames) == set(files_in_db)

    # s3 prefix should match an expectation
    expected = args['prefix']
    if timestamp:
        expected += '/' + ldu.timestamp
    for f in files_in_s3:
        dname = os.path.dirname(f)
        assert dname == expected
