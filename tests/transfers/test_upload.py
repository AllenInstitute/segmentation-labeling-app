import pytest
from unittest.mock import MagicMock
import boto3
from moto import mock_s3
import os
import slapp.transfers.upload as up


@pytest.fixture
def mock_db_conn_fixture(tmpdir_factory):
    tdir = tmpdir_factory.mktemp("contents")
    keys = [
            'source-ref', 'roi-mask-source-ref', 'video-source-ref',
            'max-source-ref', 'avg-source-ref', 'trace-source-ref',
            'full-video-source-ref']
    return_val = {
            'experiment-id': 1234,
            'roi-id': 98765
            }
    for ik, key in enumerate(keys):
        tpath = tdir.join(f"{ik}.txt")
        with open(tpath, "w") as fp:
            fp.write('content')
        return_val[key] = str(tpath)

    def mock_query(query_string):
        return [{'manifest': return_val}]

    mock_db_conn = MagicMock()
    mock_db_conn.query.side_effect = mock_query
    return mock_db_conn


@pytest.fixture(scope='function')
def bucket():
    with mock_s3():
        bucket_name = 'mybucket'
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.mark.parametrize("timestamp", [True, False])
def test_LabelDataUploader(mock_db_conn_fixture, bucket, timestamp):
    args = {
            's3_bucket_name': bucket,
            'roi_manifests_ids': [0],
            'timestamp': timestamp,
            'prefix': 'abc/def',
            }
    ldu = up.LabelDataUploader(input_data=args, args=[])
    ldu.run(mock_db_conn_fixture)

    # get what is in the bucket (function scoped)
    response = boto3.client('s3').list_objects_v2(Bucket=bucket)
    files_in_s3 = [c['Key'] for c in response['Contents']]

    # make sure we read whole contents
    assert not response['IsTruncated']

    in_local_postgres = mock_db_conn_fixture.query("something")[0]['manifest']
    files_in_db = []
    for k, v in in_local_postgres.items():
        if not isinstance(v, int):
            files_in_db.append(os.path.basename(v))

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
