import pytest
import numpy as np
from segmentation_labeling_app.transforms.transformations import (
    get_centered_coordinate_box_video
)

test_frame = np.arange(25).reshape(5, 5)
test_video = np.stack([test_frame]*2, axis=0)


@pytest.mark.parametrize(
    "arr,coord,box,expected",
    [
        (
            test_video, (0, 0), (3, 3),
            np.stack(
                [np.array([[0, 1, 2],
                           [5, 6, 7],
                           [10, 11, 12]])]*2)
        ),
        (
            test_video, (2, 2), (3, 3),
            np.stack(
                [np.array([[6, 7, 8],
                           [11, 12, 13],
                           [14, 15, 16]])]*2)
        ),
    ]
)
def test_get_centered_coordinate_box_video(arr, coord, box, expected):
    result = get_centered_coordinate_box_video(coord, box, arr)
    np.testing.assert_array_equal(expected, result)
