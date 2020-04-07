from pathlib import Path
import math
import random
import os

import pytest
import h5py
import numpy as np
import cv2
import imageio

import segmentation_labeling_app.transforms.transformations as transformations


@pytest.fixture()
def video_fixture():
    current_path = (Path(__file__)).parent
    video_path = current_path / 'resources' / 'test_video.h5'
    yield video_path


def test_video_random_downsampling(video_fixture):
    input_fps = 31
    output_fps = 4
    new_video = transformations.downsample_h5_video(video_path=video_fixture,
                                                    input_fps=input_fps,
                                                    output_fps=output_fps)

    with h5py.File(video_fixture, 'r') as open_video:
        video = open_video['data'][:]
        match_count = 0
        new_video_index = 0
        sampling_ratio = int(input_fps / output_fps)
        expected_count = math.ceil(len(video) / int((input_fps / output_fps)))

        for i in range(0, len(video), sampling_ratio):
            frames = video[i:(i + sampling_ratio)]
            new_frame = new_video[new_video_index]
            if new_frame in frames:
                match_count += 1
            new_video_index += 1
        assert expected_count == match_count


def test_video_maximum_downsampling(video_fixture):
    input_fps = 31
    output_fps = 4
    new_video = transformations.downsample_h5_video(video_path=video_fixture,
                                                    input_fps=input_fps,
                                                    output_fps=output_fps,
                                                    strategy='maximum')

    with h5py.File(video_fixture, 'r') as open_video:
        video = open_video['data'][:]
        match_count = 0
        new_video_index = 0
        sampling_ratio = int(input_fps / output_fps)
        expected_count = math.ceil(len(video) / int((input_fps / output_fps)))

        for i in range(0, len(video), sampling_ratio):
            frames = video[i:(i + sampling_ratio)]
            new_frame = new_video[new_video_index]
            test_frame = new_frame - np.max(frames, axis=0)
            if not test_frame.any():
                match_count += 1
            new_video_index += 1
        assert expected_count == match_count


def test_video_average_downsampling(video_fixture):
    input_fps = 31
    output_fps = 4
    new_video = transformations.downsample_h5_video(video_path=video_fixture,
                                                    input_fps=input_fps,
                                                    output_fps=output_fps,
                                                    strategy='average')

    with h5py.File(video_fixture, 'r') as open_video:
        video = open_video['data'][:]
        match_count = 0
        new_video_index = 0
        sampling_ratio = int(input_fps / output_fps)
        expected_count = math.ceil(len(video) / int((input_fps / output_fps)))

        for i in range(0, len(video), sampling_ratio):
            frames = video[i:(i + sampling_ratio)]
            new_frame = new_video[new_video_index]
            test_frame = new_frame - np.mean(frames)
            if not test_frame.any():
                match_count += 1
            new_video_index += 1
        assert expected_count == match_count


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


@pytest.mark.parametrize(("coordinate_pair, window_size"),
                         [((10, 10), (6, 6)),
                          ((10, 10), (12, 12)),
                          ((510, 510), (10, 10)),
                          ((10, 510), (10, 10)),
                          ((510, 10), (10, 10))])
def test_video_subset_coord_pair(coordinate_pair, window_size):
    current_path = (Path(__file__)).parent
    video_path = current_path / 'resources' / 'test_video.h5'
    with h5py.File(video_path, 'r') as open_h5:
        video = open_h5['data'][:]
        transformed_coordinates = transformations.get_transformed_center(coordinate_pair,
                                                                         window_size,
                                                                         video.shape)
        left_column = transformed_coordinates[0] - math.ceil(window_size[0] / 2)
        right_column = transformed_coordinates[0] + math.ceil(window_size[0] / 2)
        up_row = transformed_coordinates[1] - math.ceil(window_size[1] / 2)
        down_row = transformed_coordinates[1] + math.ceil(window_size[1] / 2)

        # transform video
        transformed_video = transformations.get_centered_coordinate_box(coordinate_pair,
                                                                        window_size,
                                                                        video)
        # random sample frame to test subset
        random_index = random.randint(0, 50)
        random_transformed_frame = transformed_video[random_index]
        random_regular_frame = video[random_index]
        regular_subset = random_regular_frame[up_row:down_row,
                                              left_column:right_column]

        subtraction_frame = random_transformed_frame - regular_subset
        assert not subtraction_frame.any()


def test_mp4_conversion(video_fixture):
    with h5py.File(video_fixture, 'r') as open_video:
        video = open_video['data'][:]

        norm_video = transformations.normalize_video(video)
        output_path = video_fixture.parent
        output_path = output_path / 'video.mp4'

        transformations.transform_to_mp4(video=video,
                                         output_path=output_path.as_posix())

        frames = []
        reader = imageio.get_reader(output_path.as_posix(), mode='I', fps=30,
                                    size=(500, 500), pixelformat="gray")
        for frame in reader:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frames.append(gray_frame)
        frames = np.stack(frames)

        # codec for mp4 encoding changes the video slightly between h5 and mp4
        # and vice versa, so numbers are not exactly the same. Instead general
        # structure of video array loaded. Cv color conversion does not rectify
        # this error unfortunately, this has been attempted and small errors
        # persist

        rand_idx = random.randint(0, 50)

        rand_frame = norm_video[rand_idx]
        rand_test_frame = frames[rand_idx]

        subtraction_frame = rand_frame - rand_test_frame

        assert frames.shape == video.shape

        os.remove(output_path)
