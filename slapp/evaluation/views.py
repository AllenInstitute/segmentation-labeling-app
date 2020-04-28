from typing import Optional, Tuple

from scipy.sparse import coo_matrix
import numpy as np

from slapp.transforms.array_utils import (
    center_pad_2d, crop_2d_array)


def postage_stamp(data: coo_matrix,
                  canvas_size: Optional[Tuple[int, int]] = None):
    """
    Generate a "postage stamp" image data for a matrix in coo_format.

    Parameters
    ==========
    coo_matrix: scipy.sparse.coo_matrix representation of an image to
        represent as a png
    canvas_size: tuple of canvas size in pixels, where each row and
        column is one pixel (rows, columns). If None, will fit tight to
        outer boundaries of the data. If larger than the image data,
        will place the image in the center of the canvas (as closely as
        possible given dimensions).

    Returns
    =======
    2-dimensional array representing PNG data normalized for grayscale
    (values 0-255).
    """
    # Crop image down to minimum-sized rectangle containing all nonzero data
    arr = crop_2d_array(data)
    img_size = arr.shape

    # Normalize to grayscale based on data values
    gray = (arr / arr.max() * 255).astype(np.uint8)

    # Don't need to do any transforms if fitting tightly to image data
    if canvas_size is None or canvas_size == img_size:
        return gray
    else:
        return center_pad_2d(gray, canvas_size, 0)
