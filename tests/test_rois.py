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
                          "segmentation_id, roi_id, threshold, expected_edges"),
                         [([1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3],
                           [0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[1, 1], [1, 2],
                                                        [1, 3], [2, 3],
                                                        [3, 3], [3, 2],
                                                        [3, 1], [2, 1]])),
                          ([1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
                           [0.75, 0.8, 0.9, 0.85, 0.75, 0.8, 0.82, 0.9, 0.85, 1],
                           (5, 5), 1, 1, 0.7, np.array([[1, 1], [0, 2],
                                                        [1, 3], [2, 3],
                                                        [3, 3], [3, 2],
                                                        [3, 1], [2, 1]]))])
def test_edge_detection_edges(coo_rows, coo_cols, coo_data, video_shape,
                              segmentation_id, roi_id, threshold,
                              expected_edges):
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
                           (5, 5), 1, 1, 0.7, np.array([[0, 0, 1, 0, 0],
                                                        [0, 1, 0, 1, 0],
                                                        [0, 1, 0, 1, 0],
                                                        [0, 1, 1, 1, 0],
                                                        [0, 0, 0, 0, 0]]))])
def test_edge_detection_mask(coo_rows, coo_cols, coo_data, video_shape,
                             segmentation_id, roi_id, threshold,
                             expected_mask):
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
                          ([2, 2, 2, 3, 3, 3, 4, 4, 4],
                           [2, 3, 4, 2, 3, 4, 2, 3, 4],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           (7, 7), 1, 1, 0.7, 1, np.array(
                              [[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0],
                               [0, 0, 1, 0, 1, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]]
                          )),
                          ([2, 2, 2, 3, 3, 3, 4, 4, 4],
                           [2, 3, 4, 2, 3, 4, 2, 3, 4],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           (7, 7), 1, 1, 0.7, 2, np.array(
                              [[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0]]
                          )),
                          ([2, 2, 2, 3, 3, 3, 4, 4, 4],
                           [2, 3, 4, 2, 3, 4, 2, 3, 4],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1],
                           (7, 7), 1, 1, 0.7, 3, np.array(
                              [[1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 0, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1]]
                          ))
                          ])
def test_edge_dilated_mask(coo_rows, coo_cols, coo_data, video_shape,
                           segmentation_id, roi_id, threshold, stroke_size,
                           expected_mask):
    test_roi = roi_module.ROI(coo_rows, coo_cols, coo_data,
                              video_shape, segmentation_id, roi_id)
    dilated_mask = test_roi.create_dilated_contour_mask(threshold=threshold,
                                                        stroke_size=stroke_size)
    assert np.array_equal(expected_mask, dilated_mask)
