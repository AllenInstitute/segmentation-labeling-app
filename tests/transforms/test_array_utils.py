import pytest
from scipy.sparse import coo_matrix
import numpy as np

from slapp.transforms import array_utils as au


@pytest.mark.parametrize(
    "arr,expected",
    [
        (   # Typical
            coo_matrix(
                (np.array([3, 2, 1]),
                 (np.array([0, 1, 0]), np.array([0, 1, 2]))),
                shape=(4, 4)),
            (0, 2, 0, 3)
        ),
        (   # Full data
            np.ones((10, 10)),
            (0, 10, 0, 10),
        ),
        (   # No data
            np.zeros((5, 5)),
            (0, 0, 0, 0),
        ),
        (   # Corners
            coo_matrix(
                (np.array([9, 9]),
                 (np.array([0, 6]), np.array([0, 6]))),
                shape=(7, 7)),
            (0, 7, 0, 7)
        ),
    ]
)
def test_content_boundary_2d(arr, expected):
    assert expected == au.content_boundary_2d(arr)


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
    "arr,shape,value,allow_overflow,expected",
    [
        (   # Can perfectly center, unit value
            np.array([[1]]),
            (3, 3),
            99,
            True,
            np.array([[99, 99, 99],
                      [99, 1, 99],
                      [99, 99, 99]]),
        ),
        (   # Uneven centers
            np.array([[1], [2]]),
            (3, 4),
            0,
            True,
            np.array([[0, 1, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 0, 0]]),
        ),
        (
            np.zeros((0, 0)),
            (4, 4),
            1,
            True,
            np.ones((4, 4)),
        ),
        (   # Too big, let it go
            np.ones((5, 5)),
            (4, 4),
            0,
            True,
            np.ones((5, 5))
        ),
        (   # Too big, no-go
            np.ones((5, 5)),
            (4, 4),
            0,
            False,
            None
        )
    ]
)
def test_center_pad_2d(arr, shape, value, allow_overflow, expected):
    np.testing.assert_equal(
        expected, au.center_pad_2d(arr, shape, value, allow_overflow))
