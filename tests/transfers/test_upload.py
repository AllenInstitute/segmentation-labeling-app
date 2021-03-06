import pytest
import boto3
from unittest.mock import MagicMock, patch
from moto import mock_s3
import os
import slapp.transfers.upload as up
import json
import botocore


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


@pytest.fixture
def mock_manifest(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp("manifest-file-contents")
    keys = [
        'source-ref', 'roi-mask-source-ref', 'video-source-ref',
        'max-source-ref', 'avg-source-ref', 'trace-source-ref',
        'full-video-source-ref']
    return_val = {
            'experiment-id': 1234,
            'roi-id': 98765
            }
    for ik, key in enumerate(keys):
        tpath = test_dir.join(f"{ik}.txt")
        with open(tpath, "w") as fp:
            fp.write('content')
        return_val[key] = str(tpath)
    man_path = test_dir.join("manifest.jsonl")
    with open(man_path, "w") as mp:
        mp.write(json.dumps(return_val))
    return str(man_path)


@pytest.fixture(scope='function')
def bucket():
    with mock_s3():
        bucket_name = 'mybucket'
        boto3.setup_default_session()
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=bucket_name)
        yield bucket_name


orig = botocore.client.BaseClient._make_api_call


def mock_make_api_call(self, operation_name, kwarg):
    if operation_name == 'PutObject':
        response = {
                    'ResponseMetadata': {
                        'HTTPStatusCode': 500
                        }}
        return response
    return orig(self, operation_name, kwarg)


def test_failed_upload(mock_db_conn_fixture, bucket, tmp_path, mock_manifest):
    """makes all put_objects return HTTPStatusCode != 200 to check that
    the output_json logs them all as failed uploads
    """
    output_json_path = tmp_path / "output.json"
    args = {
            's3_bucket_name': bucket,
            'prefix': 'abc/def',
            'output_json': str(output_json_path),
            'manifest_file': mock_manifest,
            }
    with patch(
            'botocore.client.BaseClient._make_api_call',
            mock_make_api_call):
        ldu = up.LabelDataUploader(input_data=args, args=[])
        ldu.run(mock_db_conn_fixture)

    with open(output_json_path, 'r') as f:
        j = json.load(f)
    assert 'successful_uploads' in j
    assert 'failed_uploads' in j
    assert 'local_s3_manifest_copy' in j
    assert len(j['failed_uploads']) == 8
    assert len(j['successful_uploads']) == 0


@pytest.mark.parametrize("timestamp,manifest", [
    (True, None),
    (False, None),
    (False, True)
    ])
def test_LabelDataUploader(mock_db_conn_fixture, bucket, timestamp, manifest,
                           mock_manifest, tmp_path):
    output_json_path = tmp_path / "output.json"
    args = {
            's3_bucket_name': bucket,
            'timestamp': timestamp,
            'prefix': 'abc/def',
            'output_json': str(output_json_path)
            }
    if manifest:
        args.update({"manifest_file": mock_manifest})
    else:
        args.update({'roi_manifests_ids': [0]})
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
            special_prefix = ''
            if k == 'full-video-source-ref':
                special_prefix = f"{in_local_postgres['experiment-id']}_"
            files_in_db.append(special_prefix + os.path.basename(v))

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

    # check the output json
    with open(output_json_path, 'r') as f:
        j = json.load(f)
    assert 'successful_uploads' in j
    assert 'failed_uploads' in j
    assert 'local_s3_manifest_copy' in j
    assert len(j['failed_uploads']) == 0
    assert len(j['successful_uploads']) == 8
