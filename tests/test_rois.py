import json
import sqlite3
import pytest
import numpy as np
from scipy.sparse import coo_matrix
import segmentation_labeling_app.rois.rois as roi_module


@pytest.mark.parametrize("use_coo", [True, False])
@pytest.mark.parametrize(
        ("weighted", "expected", "athresh", "quantile"),
        [
            (
                # quantile of 0.2 will
                # eliminate the 0.5's in this example
                np.array([
                    [0.0, 0.5, 1.0],
                    [0.0, 2.0, 2.0],
                    [2.0, 1.0, 0.5]]),
                np.array([
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0]]),
                None,
                0.2),
            (
                # absolute threshold will only keep
                # values > 1.5
                np.array([
                    [0.0, 0.5, 1.0],
                    [0.0, 2.0, 2.0],
                    [2.0, 1.0, 0.5]]),
                np.array([
                    [0, 0, 0],
                    [0, 1, 1],
                    [1, 0, 0]]),
                1.5,
                None),
                ])
def test_binary_mask_from_threshold(
        weighted, expected, athresh, quantile, use_coo):
    if use_coo:
        weighted = coo_matrix(weighted)
    binary = roi_module.binary_mask_from_threshold(
            weighted,
            absolute_threshold=athresh,
            quantile=quantile)
    assert np.all(binary == expected)


@pytest.mark.parametrize("use_coo", [True, False])
@pytest.mark.parametrize("mask, full, shape, expected", [
    (
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        None,
        np.array([
            [1.0, 1.0],
            [1.0, 1.0]])),
    (
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        True,
        None,
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        ),
    (
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        (5, 5),
        np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
    (
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        (6, 6),
        np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
    ])
def test_sized_mask(mask, full, shape, expected, use_coo):
    if use_coo:
        mask = coo_matrix(mask)
    sized = roi_module.sized_mask(mask, shape=shape, full=full)
    assert np.all(sized == expected)


@pytest.mark.parametrize("mask, full, shape, , trace, expected", [
    (
        # full=False and shape=None will just
        # crop to the data
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        None,
        [1.234, 2.345, 3.456, 4.567, 6.789],
        np.array([
            [1.0, 1.0],
            [1.0, 1.0]])),
    (
        # full = True will give back the entire frame
        # in this case, the frame is only 4x4
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        True,
        None,
        [1.234, 2.345, 3.456, 4.567, 6.789],
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        ),
    (
        # shape=(5, 5) should give back a
        # padded window of that shape
        # this is an asymmetric padding case
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        (5, 5),
        [1.234, 2.345, 3.456, 4.567, 6.789],
        np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
    (
        # shape=(6, 6) also should come back
        # this is a symmetric padding case
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]]),
        False,
        (6, 6),
        [1.234, 2.345, 3.456, 4.567, 6.789],
        np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
    ])
def test_roi_generate_mask(mask, shape, full, trace, expected):
    coo = coo_matrix(mask)
    roi = roi_module.ROI(
            coo.row,
            coo.col,
            coo.data,
            image_shape=(4, 4),
            experiment_id=1234,
            roi_id=4567,
            trace=trace)
    roi_mask = roi.generate_ROI_mask(shape=shape, full=full)
    assert np.all(roi_mask == expected)
    assert np.all(trace == roi.trace)


@pytest.mark.parametrize(
        ("weighted", "expected", "athresh", "quantile", "full"),
        [
            (
                # basic test of making an outline mask
                np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 1.0, 0.0, 0.0],
                    [0.0, 2.0, 2.0, 1.5, 0.0],
                    [2.0, 1.0, 1.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                    ]),
                np.array([
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0]
                    ]),
                0.7,
                None,
                True),
            (
                # basic test where the ROI produces two distinct
                # contours in the outline mask
                np.array([
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.5, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 2.0, 1.0, 0.5, 1.0],
                    [2.0, 0.0, 1.0, 1.5, 0.5, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ]),
                np.array([
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1]
                    ]),
                0.7,
                None,
                True),
            ])
def test_roi_generate_outline(weighted, full, expected, athresh, quantile):
    """this test is not exhaustive
    """
    coo = coo_matrix(weighted)
    roi = roi_module.ROI(
            coo.row,
            coo.col,
            coo.data,
            image_shape=weighted.shape,
            experiment_id=1234,
            roi_id=4567,
            trace=[1.234, 2.345, 3.456, 4.567, 6.789])
    outline = roi.generate_ROI_outline(
            full=full,
            absolute_threshold=athresh,
            quantile=quantile)
    assert np.all(outline == expected)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, source_ref,"
                          "video_source_ref, max_source_ref, avg_source_ref,"
                          "trace_source_ref, roi_data_source_ref,"
                          "expected_manifest"), [
                        ([0, 0, 1, 1], [0, 1, 0, 1],
                         [0.75, 0.8, 0.9, 0.85], (3, 3), 1, 1, 'test_uri',
                         'test_uri', 'test_uri', 'test_uri', 'test_uri',
                         'test_uri', json.dumps(
                            {'source_ref': 'test_uri',
                             'video_source_ref': 'test_uri',
                             'max_source_ref': 'test_uri',
                             'avg_source_ref': 'test_uri',
                             'trace_source_ref': 'test_uri',
                             'roi_data_source-ref': 'test_uri',
                             'roi_id': 1,
                             'experiment_id': 1}))
])
def test_roi_manifest_json(coo_rows, coo_cols, coo_data, video_shape,
                           segmentation_id, roi_id, source_ref,
                           video_source_ref, max_source_ref,
                           avg_source_ref, trace_source_ref,
                           roi_data_source_ref, expected_manifest):
    test_roi = roi_module.ROI(
            coo_rows,
            coo_cols,
            coo_data,
            video_shape,
            segmentation_id,
            roi_id,
            trace=[1, 2, 3.1415, 4, 5, 6])
    test_manifest = test_roi.create_manifest_json(
            source_ref=source_ref,
            video_source_ref=video_source_ref,
            max_source_ref=max_source_ref,
            avg_source_ref=avg_source_ref,
            trace_source_ref=trace_source_ref,
            roi_data_source_ref=roi_data_source_ref)
    assert test_manifest == expected_manifest


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, transform_hash,"
                          "ophys_segmentation_commit_hash, creation_date,"
                          "upload_date, manifest"), [
    ([0, 0, 1, 1], [0, 1, 0, 1],
     [0.75, 0.8, 0.9, 0.85], (3, 3), 1, 1, 'test_hash', 'test_hash',
     'test_date', 'test_date', json.dumps('test_manifest'))
])
def test_roi_db_table_write(coo_rows, coo_cols, coo_data, video_shape,
                            segmentation_id, roi_id, transform_hash,
                            ophys_segmentation_commit_hash, creation_date,
                            upload_date, manifest, tmp_path):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    database_file = tmp_path / 'test_db.db'

    test_roi.write_roi_to_db(
            database_file,
            transform_hash=transform_hash,
            ophys_segmentation_commit_hash=ophys_segmentation_commit_hash,
            creation_date=creation_date,
            upload_date=upload_date,
            manifest=manifest)

    db_connection = sqlite3.connect(database_file.as_posix())
    curr = db_connection.cursor()
    curr.execute("SELECT * FROM rois_prelabeling")
    rois = curr.fetchall()
    db_roi = rois[0]

    assert db_roi[1] == transform_hash
    assert db_roi[2] == ophys_segmentation_commit_hash
    assert db_roi[3] == creation_date
    assert db_roi[4] == upload_date
    assert db_roi[5] == manifest
    assert db_roi[6] == test_roi.experiment_id
    assert db_roi[7] == test_roi.roi_id
