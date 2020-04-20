import pytest
from scipy.sparse import coo_matrix
import numpy as np

from segmentation_labeling_app.evaluation import views


@pytest.mark.parametrize(
    "data,canvas_size,expected",
    [
        (
            coo_matrix(
                (np.array([1, 1, 1]),
                 (np.array([0, 1, 0]), np.array([0, 1, 2]))),
                shape=(4, 4)),
            (5, 5),
            np.array([[0, 0, 0, 0, 0],
                      [0, 255, 0, 255, 0],
                      [0, 0, 255, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        ),
        (   # Tight fit
            coo_matrix(
                (np.array([1, 1, 1]),
                 (np.array([0, 1, 0]), np.array([0, 1, 2]))),
                shape=(2, 3)),
            (2, 3),
            np.array([[255, 0, 255],
                      [0, 255, 0]]),
        ),
    ]
)
def test_postage_stamp(data, canvas_size, expected):
    np.testing.assert_array_equal(expected,
                                  views.postage_stamp(data, canvas_size))
