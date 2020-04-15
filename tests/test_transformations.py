from pathlib import Path
import math
import random
import os

import pytest
import numpy as np
import cv2
import imageio

import segmentation_labeling_app.transforms.transformations as transformations


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
                        ((4, 511), (10, 10), (512, 512), (4, 506))])
def test_coordinate_window_shift(coordinate_pair, window_size, video_shape,
                                 expected_pair):
    transformed_center = transformations.get_transformed_center(coordinate_pair,
                                                                     window_size,
                                                                     video_shape)
    assert transformed_center == expected_pair


@pytest.mark.parametrize("coordinate_pair, window_size, frame_cnt, video_width,"
                         "video_height",
                         [((4, 4), (6, 6), 20, 512, 512),
                          ((10, 10), (12, 12), 20, 512, 512),
                          ((510, 510), (10, 10), 20, 512, 512),
                          ((10, 510), (10, 10), 20, 512, 512),
                          ((510, 10), (10, 10), 20, 512, 512)])
def test_video_subset_coord_pair(coordinate_pair, window_size, frame_cnt,
                                 video_width, video_height):
    np.random.seed(0)
    video = np.random.randint(0, 255, size=(frame_cnt, video_width,
                                            video_height))
    transformed_coordinates = transformations.get_transformed_center(coordinate_pair,
                                                                     window_size,
                                                                     video.shape)
    left_column = transformed_coordinates[0] - math.ceil(window_size[0] / 2)
    right_column = transformed_coordinates[0] + math.ceil(window_size[0] / 2)
    up_row = transformed_coordinates[1] - math.ceil(window_size[1] / 2)
    down_row = transformed_coordinates[1] + math.ceil(window_size[1] / 2)

    # transform video
    transformed_video = transformations.get_centered_coordinate_box_video(coordinate_pair,
                                                                          window_size,
                                                                          video)
    # random sample frame to test subset
    random_index = random.randint(0, len(video) - 1)
    random_transformed_frame = transformed_video[random_index]
    random_regular_frame = video[random_index]
    regular_subset = random_regular_frame[up_row:down_row,
                                          left_column:right_column]

    assert np.array_equal(regular_subset, random_transformed_frame)


def test_mp4_conversion(video_fixture):
    output_path = Path(__file__).parent
    output_path = output_path / 'video.mp4'

    norm_video = transformations.normalize_video(video_fixture)

    transformations.transform_to_mp4(video=video_fixture,
                                     output_path=output_path.as_posix())

    frames = []
    reader = imageio.get_reader(output_path.as_posix(), mode='I', fps=30,
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

    random.seed(0)
    rand_idx = random.randint(0, len(norm_video) - 1)

    assert frames.shape == video_fixture.shape

    os.remove(output_path)
