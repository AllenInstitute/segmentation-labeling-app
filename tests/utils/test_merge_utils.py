import pytest
import jsonlines
import boto3
from moto import mock_s3

import slapp.utils.merge_utils as mu


@pytest.mark.parametrize(
        "record, exception_match",
        [
            # need an roi-id key
            ({'a': 1, 'b': 2}, r"record does not contain key 'roi-id'"),
            # need 1 and only 1 key matching *-metadata
            ({
                'roi-id': 123,
                'x-metadata': {'xmeta': 'stuff'},
                'y-metadata': {'ymeta': 'stuff'}
                },
                r".*'-metadata' expecting 1 and only 1"),
                ])
def test_get_project_key_exceptions(record, exception_match):
    with pytest.raises(mu.MergingException, match=exception_match):
        mu.get_project_key(record)


@pytest.mark.parametrize(
        "record, expected",
        [
            (
                {
                    'roi-id': 123,
                    'myproject-metadata': 'metastuff',
                    'myproject': 'stuff'
                },
                "myproject")])
def test_get_project_key(record, expected):
    assert mu.get_project_key(record) == expected


@pytest.mark.parametrize(
        "project1, project2, expected",
        [
            (
                {
                    'sourceData': 's3URI1',
                    'majorityLabel': 'cell',
                    'workerAnnotations': [
                        # repeat will not be duplicated
                        {
                            "workerId": "idA1",
                            "roiLabel": "cell"
                            },
                        {
                            "workerId": "idA2",
                            "roiLabel": "not cell"
                            },
                        ]
                        },
                {
                    'sourceData': 's3URI2',
                    'majorityLabel': 'not cell',
                    'workerAnnotations': [
                        {
                            "workerId": "idA2",
                            "roiLabel": "not cell"
                            },
                        {
                            "workerId": "idA3",
                            "roiLabel": "not cell"
                            }]
                        },
                {
                    'sourceData': 's3URI1,s3URI2',
                    'majorityLabel': 'cell',  # just inherits from 1st one
                    'workerAnnotations': [
                        {
                            "workerId": "idA1",
                            "roiLabel": "cell"
                            },
                        {
                            "workerId": "idA2",
                            "roiLabel": "not cell"
                            },
                        {
                            "workerId": "idA3",
                            "roiLabel": "not cell"
                            }]
                        })])
def test_merge_record(project1, project2, expected):
    """keeping this file short and testing 2 functions at
    once with the same parameters
    """
    assert expected == mu.merge_projects(project1, project2)

    record1 = {
            'roi-id': 1234,
            'project1': project1,
            'project1-metadata': {'job-name': 'job1'}
            }
    record2 = {
            'roi-id': 1234,
            'project2': project2,
            'project2-metadata': {'job-name': 'job2'}
            }
    expected_record = {
            'roi-id': 1234,
            'project1': expected,
            'project1-metadata': {'job-name': 'job1'}
            }
    new_record = mu.merge_records(record1, record2)
    assert new_record == expected_record


@pytest.fixture(scope='module')
def two_jobs(tmpdir_factory):
    """makes a list of jsons representative of 2 jobs
       labels are "cell" and "not cell" and None if missing
    """
    x = -1  # easier to read tables below, indicates no label from worker
    workers_job1 = [
            [0, 1, x, 0, 1, x, 0, 1],
            [1, x, 0, 1, x, 0, 1, 1],
            [0, 0, 1, x, 1, 1, 0, 1]]
    job1_majority = \
            [0, x, x, x, x, x, 0, 1]  # noqa
    workers_job2 = [
            [x, x, 0, x, x, 1, x, x],
            [x, 1, x, x, 1, x, x, x],
            [x, x, x, 0, x, x, x, x]]
    job2_majority = \
            [x, x, x, x, x, x, x, x]  # noqa
    expected = [
            [0, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1]]
    expected_majority = \
            [0, 1, 0, 0, 1, 1, 0, 1]  # noqa

    # labels coming out of our jobs are str
    translator = {1: "cell", 0: "not cell", -1: None}
    for label in [workers_job1, workers_job2, expected]:
        for i, v1 in enumerate(label):
            for j, v2 in enumerate(v1):
                label[i][j] = translator[v2]
    for label in [job1_majority, job2_majority, expected_majority]:
        for i, v in enumerate(label):
            label[i] = translator[v]

    job1 = []
    job2 = []
    expected_job = []
    for irecord in range(8):
        annotations1 = []
        annotations2 = []
        expected_annotations = []
        for iworker in range(3):
            w1 = workers_job1[iworker][irecord]
            if w1 is not None:
                annotations1.append(
                    mu.WorkerAnnotation(workerId=iworker, roiLabel=w1))
            w2 = workers_job2[iworker][irecord]
            if w2 is not None:
                annotations2.append(
                    mu.WorkerAnnotation(workerId=iworker, roiLabel=w2))
            expected_annotations.append(
                   mu.WorkerAnnotation(workerId=iworker,
                                       roiLabel=expected[iworker][irecord]))
        if len(annotations1) != 0:
            record1 = {
                'roi-id': irecord,
                'myproject': mu.Project(
                    sourceData='123',
                    majorityLabel=job1_majority[irecord],
                    workerAnnotations=annotations1),
                'myproject-metadata': {'job-name': 'job1'}
                }
            job1.append(record1)

        if len(annotations2) != 0:
            record2 = {
                'roi-id': irecord,
                'myproject': mu.Project(
                    sourceData='123',
                    majorityLabel=job2_majority[irecord],
                    workerAnnotations=annotations2),
                'myproject-metadata': {'job-name': 'job2'}
                }
            job2.append(record2)

        expected_job.append({
            'roi-id': irecord,
            'merged-project': mu.Project(
                sourceData='123',
                majorityLabel=expected_majority[irecord],
                workerAnnotations=expected_annotations),
            'merged-project-metadata': {'job-name': 'merged-job'}})

    tdir = tmpdir_factory.mktemp('job_outputs')
    jpath1 = tdir / "manifest1.jsonl"
    with open(jpath1, "w") as fp:
        w = jsonlines.Writer(fp)
        w.write_all(job1)

    jpath2 = tdir / "manifest2.jsonl"
    with open(jpath2, "w") as fp:
        w = jsonlines.Writer(fp)
        w.write_all(job2)

    yield jpath1, jpath2, expected_job


def test_merge_outputs(two_jobs):
    """tests that the merge merges as expected. Output not sent to disk or bucket
    """
    jpath1, jpath2, expected = two_jobs
    import json
    for jp in [jpath1, jpath2]:
        jl = list(mu.read_jsonlines(jp))
        print(json.dumps(jl, indent=2))

    merged = mu.merge_outputs(src_uris=[jpath1, jpath2])

    assert len(merged) == len(expected)

    # sort into order
    ids = [i['roi-id'] for i in merged]
    merged = [i for _, i in sorted(zip(ids, merged))]
    ids = [i['roi-id'] for i in expected]
    expected = [i for _, i in sorted(zip(ids, expected))]

    for i, (m, e) in enumerate(zip(merged, expected)):
        assert m.keys() == e.keys()
        for k in ['roi-id', 'merged-project-metadata']:
            assert m[k] == e[k]
        for k in ['sourceData', 'majorityLabel']:
            assert m['merged-project'][k] == e['merged-project'][k]

        # sort the labels by worker ids to align for comparison
        mwa = m['merged-project']['workerAnnotations']
        ids = [i['workerId'] for i in mwa]
        mlabels = [imwa['roiLabel'] for _, imwa in sorted(zip(ids, mwa))]

        ewa = e['merged-project']['workerAnnotations']
        ids = [i['workerId'] for i in ewa]
        elabels = [imwa['roiLabel'] for _, imwa in sorted(zip(ids, ewa))]

        assert mlabels == elabels


@mock_s3
@pytest.mark.parametrize("src_type", ["s3", "local"])
@pytest.mark.parametrize("dst_type", ["s3", "local"])
def test_merge_outputs_dst(two_jobs, src_type, dst_type, tmp_path):
    """check that the function can r/w with local or s3
    """
    jpath1, jpath2, expected = two_jobs

    bucket = "mybucket"

    if (dst_type == "s3") | (src_type == "s3"):
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=bucket)

    if src_type == "s3":
        s3.upload_file(str(jpath1), bucket, "job1.jsonl")
        jpath1 = f"s3://{bucket}/job1.jsonl"
        s3.upload_file(str(jpath2), bucket, "job2.jsonl")
        jpath2 = f"s3://{bucket}/job2.jsonl"

    if dst_type == "s3":
        dst_uri = f"s3://{bucket}/output.jsonl"
    else:
        dst_uri = tmp_path / "output.jsonl"

    merged = mu.merge_outputs(src_uris=[jpath1, jpath2], dst_uri=dst_uri)
    assert len(merged) == len(expected)

    # exists with content? separate test checks merging
    dst = list(mu.read_jsonlines(dst_uri))
    assert len(dst) == len(expected)
