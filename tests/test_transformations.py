import os

import pytest
import numpy as np
import cv2
import imageio

import segmentation_labeling_app.transforms.transformations as transformations

test_frame = np.arange(25).reshape(5, 5)
test_video = np.stack([test_frame]*2, axis=0)


@pytest.fixture()
def video_fixture():
    np.random.seed(0)
    random_video = np.array(
        [[[5, 5], [5, 5]], [[2, 2], [2, 2]],
         [[4, 4], [4, 4]], [[0, 0], [0, 0]],
         [[3, 2], [10, 7]], [[11, 1], [0, 5]],
         [[8, 2], [12, 1]], [[6, 3], [2, 8]],
         [[11, 3], [1, 17]], [[8, 3], [9, 0]]]
    )
    yield random_video


@pytest.mark.parametrize(("input_fps, output_fps, random_seed, strategy, "
                          "expected_video"), [(10, 2, 10, 'random',
                                              np.array([[[3, 2], [10, 7]],
                                                        [[11, 1], [0, 5]]])),
                                              (10, 2, 10, 'maximum',
                                               np.array([[[5, 5], [10, 7]],
                                                         [[11, 3], [12, 17]]])),
                                              (10, 2, 10, 'average',
                                               np.array([[[2.8, 2.6], [4.2, 3.6]],
                                                         [[8.8, 2.4], [4.8, 6.2]]])),
                                              (10, 2, 10, 'first',
                                               np.array([[[5, 5], [5, 5]],
                                                         [[11, 1], [0, 5]]])),
                                              (10, 2, 10, 'last',
                                               np.array([[[3, 2], [10, 7]],
                                                         [[8, 3], [9, 0]]])),
                                               (10, 3, 10, 'random',
                                                np.array([[[5, 5], [5, 5]],
                                                          [[11, 1], [0, 5]],
                                                          [[11, 3], [1, 17]]])),
                                               (10, 3, 10, 'maximum',
                                                np.array([[[5, 5], [5, 5]],
                                                          [[11, 2], [12, 7]],
                                                          [[11, 3], [9, 17]]])),
                                               (10, 6, 10, 'average',
                                                np.array([[[3.5, 3.5], [3.5, 3.5]],
                                                          [[2, 2], [2, 2]],
                                                          [[7, 1.5], [5, 6]],
                                                          [[7, 2.5], [7, 4.5]],
                                                          [[11, 3], [1, 17]],
                                                          [[8, 3], [9, 0]]])),
                                               (10, 3, 10, 'first',
                                                np.array([[[5, 5], [5, 5]],
                                                          [[3, 2], [10, 7]],
                                                          [[6, 3], [2, 8]]])),
                                               (10, 3, 10, 'last',
                                                np.array([[[0, 0], [0, 0]],
                                                          [[8, 2], [12, 1]],
                                                          [[8, 3], [9, 0]]]))])
def test_video_downsampling(input_fps, output_fps, random_seed, strategy,
                            expected_video, video_fixture):
    """
    Test Cases:
        1) Random selection of uniform sized subsets
        2) Maximum selection of uniform sized subsets
        3) Average selection of uniform sized subsets
        4) First selection of uniform sized subsets
        5) Last selection of uniform sized subsets
        6) Random selection of non uniform sized subsets
        7) Maximum selection of non uniform sized subsets
        8) Average selection of non uniform sized subsets
        9) First selection of non uniform sized subsets
        10) Last selection of non uniform sized subsets
    """
    new_video = transformations._downsample_array(full_array=video_fixture,
                                                  input_fps=input_fps,
                                                  output_fps=output_fps,
                                                  strategy=strategy,
                                                  random_seed=random_seed)
    assert np.array_equal(expected_video, new_video)


# @pytest.mark.parametrize(("coordinate_pair, window_size, video_shape,"
#                           "expected_pair"), [
#                         ((0, 9), (10, 10), (512, 512), (4, 9)),
#                         ((9, 0), (10, 10), (512, 512), (9, 4)),
#                         ((511, 4), (10, 10), (512, 512), (506, 4)),
#                         ((4, 511), (10, 10), (512, 512), (4, 506)),
#                         ((5, 5), (3, 3), (5, 5), (3, 3)),
#                         ((-1, 2), (3, 3), (5, 5), (1, 2))])
# def test_coordinate_window_shift(coordinate_pair, window_size, video_shape,
#                                  expected_pair):
#     transformed_center = transformations.get_transformed_center(coordinate_pair,
#                                                                      window_size,
#                                                                      video_shape)
#     assert transformed_center == expected_pair


# @pytest.mark.parametrize(("coordinate_pair, window_size, video_shape,"
#                           "expected_pair"),[
#                         ((3, 3), (600, 600), (512, 512), None)
#                         ])
# def test_coordinate_shift_exception(coordinate_pair, window_size, video_shape,
#                                     expected_pair):
#     with pytest.raises(ValueError):
#         transformed_center = transformations.get_transformed_center(coordinate_pair,
#                                                                     window_size,
#                                                                     video_shape)
#         assert transformed_center == expected_pair


def test_mp4_conversion(video_fixture, tmp_path):
    output_path = tmp_path / 'video.mp4'

    fps = 30

    transformations.transform_to_mp4(video=video_fixture,
                                     output_path=output_path.as_posix(),
                                     fps=fps),

    frames = []
    reader = imageio.get_reader(output_path.as_posix(), mode='I', fps=fps,
                                size=(2, 2), pixelformat="gray")
    for frame in reader:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(gray_frame)
    frames = np.stack(frames)

    # codec for mp4 encoding changes the video slightly between h5 and mp4
    # and vice versa, so numbers are not exactly the same. Instead general
    # structure of video array loaded. Cv color conversion does not rectify
    # this error unfortunately, this has been attempted and small errors
    # persist

    assert frames.shape == video_fixture.shape
