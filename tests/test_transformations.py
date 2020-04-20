from pathlib import Path
import math
import random
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
                                                         [[8, 3], [9, 0]]]))
                                              ])
def test_video_downsampling(input_fps, output_fps, random_seed, strategy,
                            expected_video, video_fixture):
    new_video = transformations._downsample_array(full_array=video_fixture,
                                                  input_fps=input_fps,
                                                  output_fps=output_fps,
                                                  strategy=strategy,
                                                  random_seed=random_seed)
    assert np.array_equal(expected_video, new_video)


@pytest.mark.parametrize(("coordinate_pair, window_size, video_shape,"
                          "expected_pair"), [
                        ((0, 9), (10, 10), (512, 512), (4, 9)),
                        ((9, 0), (10, 10), (512, 512), (9, 4)),
                        ((511, 4), (10, 10), (512, 512), (506, 4)),
                        ((4, 511), (10, 10), (512, 512), (4, 506)),
                        ((5, 5), (3, 3), (5, 5), (3, 3))])
def test_coordinate_window_shift(coordinate_pair, window_size, video_shape,
                                 expected_pair):
    transformed_center = transformations.get_transformed_center(coordinate_pair,
                                                                     window_size,
                                                                     video_shape)
    assert transformed_center == expected_pair


def test_mp4_conversion(video_fixture):
    output_path = Path(__file__).parent
    output_path = output_path / 'video.mp4'

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

    os.remove(output_path)


@pytest.mark.parametrize("video,coord,box_size,expected",
    [(test_video, (0, 0), (3, 3),
      np.stack([np.array([[0, 1, 2],
                          [5, 6, 7],
                          [10, 11, 12]])]*2)),
     (test_video, (2, 2), (3, 3), np.stack([np.array([[6, 7, 8],
                                                      [11, 12, 13],
                                                      [16, 17, 18]])]*2)),
     (test_video, (-1, -1), (3, 3), np.stack([np.array([[0, 1, 2], [5, 6, 7],
                                                       [10, 11, 12]])]*2)),
     (test_video, (10, 10), (3, 3), np.stack([np.array([[12, 13, 14], [17, 18, 19],
                                                        [22, 23, 24]])] * 2)),
     (test_video, (0, 10), (3, 3), np.stack([np.array([[10, 11, 12], [15, 16, 17],
                                                       [20, 21, 22]])] * 2)),
     (test_video, (10, 0), (3, 3), np.stack([np.array([[2, 3, 4], [7, 8, 9],
                                                       [12, 13, 14]])] * 2)),
     (test_video, (2, -1), (3, 3), np.stack([np.array([[1, 2, 3], [6, 7, 8],
                                                        [11, 12, 13]])] * 2)),
     (test_video, (-1, 2), (3, 3), np.stack([np.array([[5, 6, 7], [10, 11, 12],
                                                       [15, 16, 17]])]*2)),
     (test_video, (0, 0), (5, 5), test_video),
     (test_video, (2, 2), (0, 0), np.zeros(shape=(2, 0, 0)))]
)
def test_get_centered_coordinate_box_video(video, coord, box_size, expected):
    result = transformations.get_centered_coordinate_box_video(coord, box_size,
                                                               video)
    np.testing.assert_array_equal(expected, result)
