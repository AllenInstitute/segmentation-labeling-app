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
    subtraction_mask = generated_mask - expected_mask
    assert not subtraction_mask.any()


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
                                                        [1, 1, 0, 1, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [0, 0, 0, 0, 0]]))])
def test_edge_detection_mask(coo_rows, coo_cols, coo_data, video_shape,
                             segmentation_id, roi_id, threshold,
                             expected_mask):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    edge_mask = test_roi.get_edge_mask(threshold=threshold)
    assert np.array_equal(edge_mask, expected_mask)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold,"
                          "expected_edge_points"), [
                        ([1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [1, 2, 3, 1, 2, 3, 1, 2, 3],
                         [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                         (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2], [1, 3],
                                                      [2, 1], [2, 3], [3, 1],
                                                      [3, 2], [3, 3]])),
                         ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                          [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                          [0.75, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                          (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2], [1, 3],
                                                       [2, 0], [2, 1], [2, 3],
                                                       [3, 1], [3, 2], [3, 3]]))])
def test_edge_detection_points(coo_rows, coo_cols, coo_data, video_shape,
                               segmentation_id, roi_id, threshold,
                               expected_edge_points):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    edge_points = test_roi.get_roi_edge_coordinates(threshold=threshold)
    assert np.array_equal(edge_points, expected_edge_points)


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold,"
                          "expected_center"),
                         [([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, (2, 2)),
                          ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                           [0.85, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, (1, 2)),
                          ([0, 0, 1, 1], [0, 1, 0, 1], [0.8, 1, 0.9, 0.75],
                           (3, 3), 1, 1, 0.7, (0, 0))])
def test_roi_get_center(coo_rows, coo_cols, coo_data, video_shape,
                        segmentation_id, roi_id, threshold, expected_center):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    centroid = test_roi.get_centroid_of_thresholded_mask(threshold)
    assert centroid == expected_center


@pytest.mark.parametrize(("coo_rows, coo_cols, coo_data, video_shape,"
                          "segmentation_id, roi_id, threshold,"
                          "expected_edge_mask, expected_edges"), [
                        ([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                         [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1, 0.95, 0.85, 0.7, 0.82, 0.74, 0.82, 0.9],
                         (6, 6), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0],
                                                      [0, 0, 1, 1, 0, 0],
                                                      [0, 0, 1, 1, 0, 0],
                                                      [0, 0, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0]]),
                         np.array([[2, 2], [2, 3], [3, 2], [3, 3]])),
                         ([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                          [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                          [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1, 0.95, 0.85, 0.7],
                          (6, 6), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0],
                                                       [0, 0, 1, 1, 0, 0],
                                                       [0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0]]),
                          np.array([[2, 2], [2, 3]])),
                         ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                          [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                          [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1, 0.95, 0.85, 0.7],
                          (6, 6), 1, 1, 0.7, np.array([[0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 0]]),
                          np.array([[2, 2], [3, 2]]))
])
def test_roi_get_inner_outline(coo_rows, coo_cols, coo_data, video_shape,
                               segmentation_id, roi_id, threshold,
                               expected_edge_mask, expected_edges):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data, video_shape,
                              segmentation_id, roi_id)
    generated_edge_mask, genereated_edge_points = test_roi.generate_roi_inner_edge_coordinates(stroke_size=1,
                                                                                               threshold=threshold)
    assert np.array_equal(generated_edge_mask, expected_edge_mask)
    assert np.array_equal(genereated_edge_points, expected_edges)
