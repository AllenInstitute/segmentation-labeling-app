import pytest
from scipy.sparse import coo_matrix
import numpy as np

from segmentation_labeling_app.transforms import array_utils as au


@pytest.mark.parametrize(
    "arr,expected",
    [
        (
            coo_matrix(
                (np.array([3, 2, 1]),
                 (np.array([0, 1, 0]), np.array([0, 1, 2]))),
                shape=(4, 4)),
            np.array([[3, 0, 1], [0, 2, 0]]),
        ),
        (
            np.ones((5, 5)),
            np.ones((5, 5)),
        ),
        (
            np.array([[1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1]])
        ),
        (
            np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1]]),
            np.array([[1]])
        )
    ]
)
def test_crop_array(arr, expected):
    np.testing.assert_array_equal(expected, au.crop_2d_array(arr))


@pytest.mark.parametrize(
    "arr",
    [
        (np.zeros((3, 8))),
        (np.zeros(0)),
    ]
)
def test_crop_array_raises_error(arr):
    with pytest.raises(ValueError):
        au.crop_2d_array(arr)


@pytest.mark.parametrize(
    "arr,shape,value,expected",
    [
        (   # Can perfectly center, unit value
            np.array([[1]]),
            (3, 3),
            99,
            np.array([[99, 99, 99],
                      [99, 1, 99],
                      [99, 99, 99]]),
        ),
        (   # Uneven centers
            np.array([[1], [2]]),
            (3, 4),
            0,
            np.array([[0, 1, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 0, 0]]),
        ),
        (
            np.zeros((0, 0)),
            (4, 4),
            1,
            np.ones((4, 4)),
        ),
    ]
)
def test_center_pad_2d(arr, shape, value, expected):
    np.testing.assert_array_equal(expected,
                                  au.center_pad_2d(arr, shape, value))
