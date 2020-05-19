import pytest
import numpy as np
import slapp.transforms.image_utils as image_utils


@pytest.mark.parametrize(
        "test_image, scale_position, um_per_pixel, scale_size_um, "
        "thickness_um, expected",
        [
            (
                # text out of frame, scale is 1
                np.uint8([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
                (0, 3), 1.0, 4.0, 1.0,
                np.uint8([[1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 1, 1, 1]])),
            (
                # scale is 1, overwrites image
                np.uint8([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 8, 7, 6],
                          [5, 4, 3, 2]]),
                (0, 3), 1.0, 4.0, 1.0,
                np.uint8([[1, 2, 3, 4],
                          [1, 6, 7, 8],
                          [1, 8, 7, 6],
                          [1, 1, 1, 1]])),
            (
                # text out of frame, scale is 1
                np.uint8([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
                (1, 2), 1.0, 3.0, 1.0,
                np.uint8([[0, 1, 0, 0],
                          [0, 1, 0, 0],
                          [0, 1, 1, 1],
                          [0, 0, 0, 0]])),
            (
                # text out of frame, changing resolution
                np.uint8([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]),
                (2, 4), 0.5, 2.0, 0.5,
                np.uint8([[0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0]]))
                ])
def test_add_scale_array(test_image, scale_position, um_per_pixel,
                         scale_size_um, thickness_um, expected):
    array_with_scale = image_utils.add_scale(
            test_image,
            scale_position,
            um_per_pixel,
            scale_size_um,
            1,
            thickness_um)

    assert test_image.dtype == array_with_scale.dtype
    np.testing.assert_array_equal(array_with_scale, expected)
