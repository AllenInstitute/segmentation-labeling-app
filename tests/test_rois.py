import json

import sqlite3
import pytest
import numpy as np

import segmentation_labeling_app.rois.rois as roi_module


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold, expected_mask"), [
    ([0, 0, 1, 1], [0, 1, 0, 1],
     [0.75, 0.8, 0.9, 0.85], (3, 3), 1, 1, 0.7,
     np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
    ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2],
     [0.75, 0.8, 0.5, 0.9, 0.85, 0.4], (3, 3), 1, 1, 0.7,
     np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))
])
def test_binary_mask(coo_rows, coo_cols, coo_data, video_shape,
                     segmentation_id, roi_id, threshold, expected_mask):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    generated_mask = test_roi.generate_binary_mask_from_threshold(threshold=threshold)
    assert np.array_equal(generated_mask, expected_mask)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold, expected_edges"),
                         [([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2],
                                                        [1, 3], [2, 1],
                                                        [2, 3], [3, 1],
                                                        [3, 2], [3, 3]])),
                          ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                           [0.75, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[0, 2], [1, 1],
                                                        [1, 3], [2, 1],
                                                        [2, 3], [3, 1],
                                                        [3, 2], [3, 3]])),
                          ([1], [1], [0.75], (2, 2), 1, 1, 0.7,
                           np.array([[1, 1]])),
                          ([], [], [], (2, 2), 1, 1, 0.7,
                           np.array([])),
                          ([1, 1, 1, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2],
                                                        [1, 3], [2, 1],
                                                        [2, 3],
                                                        [3, 1], [3, 2],
                                                        [3, 3]])),
                          ([1, 1, 2, 2], [1, 2, 1, 2], [0.5, 0.5, 0.5, 0.5],
                           (5, 5), 1, 1, 0.7, np.array([])),
                          ([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.4, 0.75, 0.5, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2],
                                                        [1, 3], [2, 1],
                                                        [2, 3],
                                                        [3, 2], [3, 3]])),
                          ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 2, 3, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.75, 0.84, 0.9, 0.87],
                           (4, 4), 1, 1, 0.7, np.array([[0, 0], [0, 1],
                                                        [1, 0], [1, 1],
                                                        [2, 2], [2, 3],
                                                        [3, 2], [3, 3]]))])
def test_edge_detection_edges(coo_rows, coo_cols, coo_data, video_shape,
                              segmentation_id, roi_id, threshold,
                              expected_edges):
    """
    Test Cases:
        1) 3 x 3 object in 5 x 5 video
        2) 3 x 3 object with out cropping in 5 x 5 video
        3) 1 x 1 object in 5 x 5 video
        4) 0 x 0 object in 5 x 5 video
        5) 3 x 3 object with a hole in center in a 5 x 5 box
        6) 2 x 2 object with all values below threshold in a 5 x 5 box
        7) 3 x 3 object with one pixel below threshold in 5 x 5 box
        8) 2 x 2 object in top left corner and 2 x 2 in bottom right corner
           not touching each other in a 4 x 4 box
    """
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    edge_points = test_roi.get_edge_points(threshold=threshold)
    assert np.array_equal(expected_edges, edge_points)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold, expected_mask"),
                         [([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [0, 1, 0, 1, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [0, 0, 0, 0, 0]])),
                          ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                           [0.75, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [1, 0, 0, 1, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [0, 0, 0, 0, 0]])),
                          ([], [], [], (5, 5), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0],
                                                                    [0, 0, 0, 0, 0]])),
                          ([1], [1], [1], (3, 3), 1, 1, 0.75, np.array([[0, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 0]]))])
def test_edge_detection_mask(coo_rows, coo_cols, coo_data, video_shape,
                             segmentation_id, roi_id, threshold,
                             expected_mask):
    """
    Test Case:
        1) 3 x 3 object in 5 x 5 in frame, edge mask gets rid of center
        2) 3 x 3 object with overhang at (2, 0) in a 5 x 5 frame, edge mask eliminates pixels (2, 1) at (2, 2)
        3) 0 x 0 object in a 5 x 5 frame, edge mask gets nothing
        4) 1 x 1 object in a 5 x 5 frame, edge mask return 1 x 1 object
    """
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    edge_mask = test_roi.get_edge_mask(threshold=threshold)
    assert np.array_equal(expected_mask, edge_mask)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold, stroke_size,"
                          "expected_mask"),
                         [([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, 1, np.array([[0, 0, 0, 0, 0],
                                                          [0, 1, 1, 1, 0],
                                                          [0, 1, 0, 1, 0],
                                                          [0, 1, 1, 1, 0],
                                                          [0, 0, 0, 0, 0]])),
                          ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                           [0.75, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, 1, np.array([[0, 0, 0, 0, 0],
                                                          [0, 1, 1, 1, 0],
                                                          [1, 0, 0, 1, 0],
                                                          [0, 1, 1, 1, 0],
                                                          [0, 0, 0, 0, 0]])),
                          ([2, 2, 3, 3], [2, 3, 2, 3], [0.85, 0.9, 0.89, 0.72],
                           (5, 5), 1, 1, 0.7, 2, np.array([[0, 0, 0, 0, 0],
                                                           [0, 1, 1, 1, 1],
                                                           [0, 1, 1, 1, 1],
                                                           [0, 1, 1, 1, 1],
                                                           [0, 1, 1, 1, 1]])),
                          ([2, 2, 2, 3, 3, 4, 4, 4],
                           [2, 3, 4, 2, 4, 2, 3, 4],
                           [1, 1, 1, 1, 1, 1, 1, 1],
                           (7, 7), 1, 1, 0.7, 2, np.array(
                              [[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0]]
                          )),
                          ([], [], [], (3, 3), 1, 1, 0.7, 3,
                           np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
                          ([1], [1], [1], (3, 3), 1, 1, 0.7, 2,
                           np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
                          ([0, 0, 1, 1, 3, 3, 4, 4], [0, 1, 0, 1, 3, 4, 3, 4],
                           [1, 1, 1, 1, 1, 1, 1, 1], (5, 5), 1, 1, 0.5, 2,
                           np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0],
                                     [1, 1, 0, 1, 1], [0, 0, 1, 1, 1],
                                     [0, 0, 1, 1, 1]])),
                          ([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.75, 0.89, 0.92, 1, 0.43, 0.56, 0.98, 0.3, 0.1],
                           (5, 5), 1, 1, 0.5, 2, np.array([[1, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 1],
                                                           [1, 1, 0, 1, 1],
                                                           [1, 1, 0, 1, 1],
                                                           [1, 1, 1, 0, 0]]))
                          ])
def test_edge_dilated_mask(coo_rows, coo_cols, coo_data, video_shape,
                           segmentation_id, roi_id, threshold, stroke_size,
                           expected_mask):
    """
    Test Cases:
        1) 3 x 3 object, dilated once to removes middle
        2) 3 x 3 object with overhang at row 2 col 0, dilated once removes all but edges
        3) 2 x 2 object centered in the bottom right quadrant, dilated once doubles width and height
        4) 3 x 3 object centered in the middle of frame with hole at center, dilated twice
        5) 0 x 0 object, dilated three times maintains no valid pixels
        6) 1 x 1 object in 3 x 3 frame, dilated twice to fill frame
        7) Two 2 x 2 objects in 5 x 5 frame, disjoint and not touching, dilated once
        8) 3 x 3 object with certain pixels below threshold, dilated twice
    """
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    dilated_mask = test_roi.create_dilated_contour_mask(threshold=threshold,
                                                        stroke_size=stroke_size)
    assert np.array_equal(expected_mask, dilated_mask)


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
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    test_manifest = test_roi.create_manifest_json(source_ref=source_ref,
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

    _ = test_roi.write_roi_to_db(database_file,
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
