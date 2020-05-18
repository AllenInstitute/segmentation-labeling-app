from pathlib import Path

import pytest
import numpy as np
import imageio

import slapp.transforms.image_utils as image_utils


def png_image(shape, tmp_path):
    image_array = np.full(shape=shape, fill_value=255)
    image_path = tmp_path / 'image.png'
    imageio.imsave(image_path, image_array)
    return image_path


@pytest.mark.parametrize("img_size, scale_position, scale_size, width,"
                         "expected", [
                          ((5, 5), (0, 4), 3, 1,
                           np.array([[0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 0, 0, 0, 0]]))])
def test_add_scale_png(img_size, scale_position, scale_size, width, expected,
                       tmp_path):
    # create dummy png image
    test_image = png_image(shape=img_size, tmp_path=tmp_path)
    font_file = Path(__file__).parents[1] / 'resources' / 'arial.ttf'
    png_with_scale = image_utils.add_scale(test_image,
                                           scale_position=scale_position,
                                           font_file=font_file,
                                           scale_size=scale_size,
                                           width=width)
    # convert image from RGBA to gray scale for testing
    png_with_scale = png_with_scale.convert('LA')
    # convert to numpy array
    png_scale_matrix = np.array(png_with_scale)
    # test if black frame equivalent to expected
    np.testing.assert_array_equal(png_scale_matrix[:, :, 0], expected)


@pytest.mark.parametrize("img_size, fill_value, scale_position, scale_size,"
                         "width, expected", [
                          ((5, 5), 255, (0, 4), 3, 1,
                           np.array([[0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 255, 255, 255, 255],
                                     [0, 0, 0, 0, 0]]))])
def test_add_scale_array(img_size, fill_value, scale_position, scale_size,
                         width, expected, tmp_path):
    # create dummy png image
    test_image = np.full(shape=img_size, fill_value=fill_value, dtype=np.uint8)
    font_file = Path(__file__).parents[1] / 'resources' / 'arial.ttf'
    png_with_scale = image_utils.add_scale(test_image,
                                           scale_position=scale_position,
                                           font_file=font_file,
                                           scale_size=scale_size,
                                           width=width)
    # convert image from RGBA to gray scale for testing
    png_with_scale = png_with_scale.convert('LA')
    # convert to numpy array
    png_scale_matrix = np.array(png_with_scale)
    # test if black frame equivalent to expected
    np.testing.assert_array_equal(png_scale_matrix[:, :, 0], expected)
