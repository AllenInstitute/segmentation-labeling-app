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


@pytest.mark.parametrize(
        ("arr", "shape", "expected"),
        [
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (5, 5),
                ((1, 6, 1, 6), ((0, 0), (0, 0)))),
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (5, 5),
                ((1, 6, 1, 6), ((0, 0), (0, 0)))),
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (5, 5),
                ((2, 7, 2, 7), ((0, 0), (0, 0)))),
            (
                np.array([
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (5, 5),
                ((0, 3, 0, 4), ((2, 0), (1, 0)))),
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (2, 2),
                ((3, 5, 3, 5), ((0, 0), (0, 0)))),
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (3, 3),
                ((2, 5, 2, 5), ((0, 0), (0, 0)))),
            ])
def test_content_extents(arr, shape, expected):
    bounds = au.content_extents(arr, shape)
    assert np.all(bounds == expected)


@pytest.mark.parametrize(
        ("arr", "shape"),
        [
            (
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 2, 2, 0, 0],
                    [0, 0, 1, 1, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]),
                (5, 5)),
            ])
def test_compare_crop_pad_and_extents(arr, shape):
    """check that cropping and padding an image
    is the same as getting the extents and
    indexing the array (i.e. as would be applied to video frame)
    """
    cropped_padded = au.center_pad_2d(au.crop_2d_array(arr), shape)
    extents, pad_width = au.content_extents(arr, shape)
    indexed = np.pad(
            arr[extents[0]:extents[1], extents[2]:extents[3]],
            pad_width)
    assert np.all(cropped_padded == indexed)


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, random_seed, strategy, expected"),
        [
            (
                # random downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'random',
                np.array([2, 5])),
            (
                # random downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'random',
                np.array([[2, 1], [5, 8]])),
            (
                # first downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'first',
                np.array([1, 3])),
            (
                # random downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'first',
                np.array([[1, 3], [3, 2]])),
            (
                # last downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'last',
                np.array([2, 11])),
            (
                # last downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'last',
                np.array([[2, 1], [11, 12]])),
            (
                # average downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'average',
                np.array([13/4, 19/3])),
            (
                # average downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'average',
                np.array([[13/4, 4], [19/3, 22/3]])),
            (
                # maximum downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'maximum',
                np.array([6, 11])),
            ])
def test_downsample(array, input_fps, output_fps, random_seed, strategy,
                    expected):
    array_out = au.downsample_array(
            array=array,
            input_fps=input_fps,
            output_fps=output_fps,
            strategy=strategy,
            random_seed=random_seed)
    assert np.array_equal(expected, array_out)


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, random_seed, strategy, expected"),
        [
            (
                # upsampling not defined
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 11, 0, 'maximum',
                np.array([6, 11])),
            (
                # maximum downsample ND array
                # not defined
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 1234], [11, 12]]),
                7, 2, 0, 'maximum',
                np.array([[6, 8], [11, 12]])),
            ])
def test_downsample_exceptions(array, input_fps, output_fps, random_seed,
                               strategy, expected):
    with pytest.raises(ValueError):
        au.downsample_array(
                array=array,
                input_fps=input_fps,
                output_fps=output_fps,
                strategy=strategy,
                random_seed=random_seed)


@pytest.mark.parametrize(
        "array, lower_cutoff, upper_cutoff, expected",
        [
            (
                np.array([
                    [0.0, 100.0, 200.0],
                    [300.0, 400.0, 500.0],
                    [600.0, 700.0, 800.0]]),
                250, 650,
                np.uint8([
                    [0, 0, 0],
                    [31, 95, 159],
                    [223, 255, 255]]))
                ]
        )
def test_normalize_array(array, lower_cutoff, upper_cutoff, expected):
    normalized = au.normalize_array(array, lower_cutoff, upper_cutoff)
    np.testing.assert_array_equal(normalized, expected)
