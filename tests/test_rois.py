import pytest
import numpy as np
from scipy.sparse import coo_matrix
import slapp.rois as roi_module


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
                # expected is 0 on 255 background
                255 * np.uint8([
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1]
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
                255 * np.uint8([
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0]
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


@pytest.mark.parametrize(
        "mask_matrix, xoff, yoff, shape, expected",
        [
            (
                [[True, False, True, False],
                 [False, True, True, False],
                 [False, True, True, False],
                 [False, True, True, False]],
                0, 0, None,
                [[True, False, True],
                 [False, True, True],
                 [False, True, True],
                 [False, True, True]]
                ),
            (
                [[True, False, True, False],
                 [False, True, True, False],
                 [False, True, True, False],
                 [False, True, True, False]],
                0, 0, (6, 5),
                [[True, False, True, False, False],
                 [False, True, True, False, False],
                 [False, True, True, False, False],
                 [False, True, True, False, False],
                 [False, False, False, False, False],
                 [False, False, False, False, False]]
                ),
            (
                [[True, False, True, False],
                 [False, True, True, False],
                 [False, True, True, False],
                 [False, True, True, False]],
                2, 1, (6, 5),
                [[False, False, False, False, False],
                 [False, False, True, False, True],
                 [False, False, False, True, True],
                 [False, False, False, True, True],
                 [False, False, False, True, True],
                 [False, False, False, False, False]]
                ),
            ])
def test_coo_from_lims_style(mask_matrix, xoff, yoff, shape, expected):
    coo = roi_module.coo_from_lims_style(mask_matrix, xoff, yoff, shape)
    np.testing.assert_array_equal(coo.toarray(), np.array(expected))
