import pytest
import numpy as np
import imageio
import h5py
import slapp.transforms.video_utils as transformations


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


def test_mp4_conversion(tmp_path):
    # make the test video a size of at least 16x16
    # otherwise, need to mess with macro_block_size arg
    nframes_written = 25
    output_path = tmp_path / 'video.mp4'
    fps = 30
    rng = np.random.default_rng(0)
    video = np.array(
            [rng.integers(0, 256, size=(16, 16), dtype='uint8')
             for i in range(nframes_written)])

    transformations.transform_to_mp4(video=video,
                                     output_path=output_path.as_posix(),
                                     fps=fps),

    reader = imageio.get_reader(output_path.as_posix(), mode='I', fps=fps,
                                pixelformat="gray")
    meta = reader.get_meta_data()
    nframes_read = int(np.round(meta['duration'] * meta['fps']))

    assert nframes_read == nframes_written

    read_video = np.zeros((nframes_read, *meta['size']), dtype='uint8')
    for i in range(nframes_read):
        read_video[i] = reader.get_data(i)[:, :, 0]
    reader.close()

    assert read_video.shape == video.shape
    # the default settings for imageio-ffmpeg are not lossless
    # so can't test for exact match
    np.testing.assert_allclose(read_video, video, atol=25)
