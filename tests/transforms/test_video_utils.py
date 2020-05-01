import pytest
import numpy as np
import cv2
import imageio
import h5py
import slapp.transforms.video_utils as transformations
from slapp.transforms.array_utils import normalize_array

test_frame = np.arange(25).reshape(5, 5)
test_video = np.stack([test_frame]*2, axis=0)


@pytest.fixture()
def video_fixture():
    video = np.array(
        [[[5, 5], [5, 5]], [[2, 2], [2, 2]],
         [[4, 4], [4, 4]], [[0, 0], [0, 0]],
         [[3, 2], [10, 7]], [[11, 1], [0, 5]],
         [[8, 2], [12, 1]], [[6, 3], [2, 8]],
         [[11, 3], [1, 17]], [[8, 3], [9, 0]]]
    )
    yield video


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, strategy, expected"),
        [
            (
                # average downsample video file with this dataset:
                np.array([
                    [[1, 1], [1, 1]],
                    [[2, 2], [2, 2]],
                    [[3, 3], [3, 3]],
                    [[4, 4], [4, 4]],
                    [[5, 5], [5, 5]],
                    [[6, 6], [6, 6]],
                    [[7, 7], [7, 7]]]),
                7, 2, 'average',
                np.array([
                    [[2.5, 2.5], [2.5, 2.5]],
                    [[6.0, 6.0], [6.0, 6.0]]])),
                ])
def test_video_downsample(
        array, input_fps, output_fps, strategy, expected, tmp_path):

    video_file = tmp_path / "sample_video_file.h5"
    with h5py.File(video_file, "w") as h5f:
        h5f.create_dataset('data', data=array)

    downsampled_video = transformations.downsample_h5_video(
            video_file,
            input_fps,
            output_fps,
            strategy)

    assert np.array_equal(downsampled_video, expected)


def test_mp4_conversion(video_fixture, tmp_path):
    output_path = tmp_path / 'video.mp4'

    fps = 30

    normalized_video = normalize_array(
            video_fixture, 0, video_fixture.max())

    transformations.transform_to_mp4(video=normalized_video,
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
