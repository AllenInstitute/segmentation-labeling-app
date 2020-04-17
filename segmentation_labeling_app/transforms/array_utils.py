import math
import logging
from typing import Tuple, Type, Union
import numpy as np
from scipy.sparse import coo_matrix


def content_boundary_2d(arr: Union[np.ndarray, coo_matrix]) -> np.ndarray:
    """
    Get the minimal row/column boundaries for content in a 2d array.

    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.

    Returns
    =======
    4-tuple of row/column boundaries that define the minimal rectangle
    around nonzero array content. Note that the maximum side boundaries
    follow python indexing rules, so the value returned is the actual
    max index + 1:
        top_bound: smallest row index
        bot_bound: largest row index + 1
        left_bound: smallest column index
        right_bound: largest column index + 1
    """
    if isinstance(arr, coo_matrix):
        col = arr.col
        row = arr.row
    else:
        if not isinstance(arr, np.ndarray):
            arr = arr.toarray()
        row, col = np.nonzero(arr)
    if not row.size:
        logging.warning("No content found. Either array is empty or all "
                        "elements equal zero.")
        return (0, 0, 0, 0)
    left_bound = col.min()
    right_bound = col.max() + 1
    top_bound = row.min()
    bot_bound = row.max() + 1
    return top_bound, bot_bound, left_bound, right_bound


def crop_2d_array(arr: Union[np.ndarray, coo_matrix]) -> np.ndarray:
    """
    Crop a 2d array to a rectangle surrounding all nonzero elements.

    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.

    Raises
    ======
    ValueError if all elements are nonzero.
    """
    boundaries = content_boundary_2d(arr)
    if not isinstance(arr, np.ndarray):
        arr = arr.toarray()
    if sum(boundaries) == 0:
        raise ValueError("Cannot crop an empty array, or an array where all "
                         "elements are zero.")
    top_bound, bot_bound, left_bound, right_bound = boundaries
    return arr[top_bound:bot_bound, left_bound:right_bound]


def center_pad_2d(arr: np.ndarray, shape: Tuple[int, int],
                  value: Type[np.dtype] = 0,
                  allow_overflow: bool = True) -> np.ndarray:
    """
    Add padding around a numpy array such that the original array data
    stays in the center. If padding cannot be evenly applied due to the
    size of array data and desired shape, then the extra padding will
    be applied after (on the bottom for the row axis and right side
    for the column axis).

    Parameters
    ==========
    arr: (np.ndarray) 2d array of data
    shape: (Tuple[int,int]) Desired final shape after padding is applied.
        If smaller than the input array, will return the input array
        without any changes.
    value: (inherit from np.dtype) Any valid numpy dtype value. Should
        be homogenous with the input array.
    allow_overflow: (bool) If true, will return array unchanged if the
        array size is larger than the value for `shape`. If false,
        will return None instead.
    """
    if arr.size == 0:
        return np.full(shape, value)
    img_size = arr.shape
    vertical_pad = shape[0] - img_size[0]
    horizontal_pad = shape[1] - img_size[1]

    if (img_size[0] > shape[0]) or (img_size[1] > shape[1]):
        logging.warning("Specified shape after padding is too small. "
                        "Returning input array without padding.")
        if allow_overflow:
            return arr
        else:
            return None

    if vertical_pad % 2 == 0:
        top_pad = bottom_pad = int(vertical_pad / 2)
    else:
        top_pad = math.floor(vertical_pad / 2)
        bottom_pad = top_pad + 1
    if horizontal_pad % 2 == 0:
        left_pad = right_pad = int(horizontal_pad / 2)
    else:
        left_pad = math.floor(horizontal_pad / 2)
        right_pad = left_pad + 1

    return np.pad(arr, ((top_pad, bottom_pad), (left_pad, right_pad)),
                  mode="constant", constant_values=(value,))
