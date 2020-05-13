from pathlib import Path
from typing import Union
import h5py
import numpy as np
import imageio_ffmpeg as mpg
from slapp.transforms.array_utils import downsample_array


def downsample_h5_video(
        video_path: Union[Path],
        input_fps: int = 31,
        output_fps: int = 4,
        strategy: str = 'average',
        random_seed: int = 0) -> np.ndarray:
    """Opens an h5 file and downsamples dataset 'data'
    along axis=0

    Parameters
    ----------
        video_path: pathlib.Path
            path to an h5 video. Should have dataset 'data'. For video,
            assumes dimensions [time, width, height] and downsampling
            applies to time.
        input_fps: int
            frames-per-second of the input array
        output_fps: int
            frames-per-second of the output array
        strategy: str
            downsampling strategy. 'random', 'maximum', 'average',
            'first', 'last'. Note 'maximum' is not defined for
            multi-dimensional arrays
        random_seed: int
            passed to numpy.random.default_rng if strategy is 'random'

    Returns:
        video_out: numpy.ndarray
            array downsampled along axis=0
    """
    with h5py.File(video_path, 'r') as h5f:
        video_out = downsample_array(
                h5f['data'],
                input_fps,
                output_fps,
                strategy,
                random_seed)
    return video_out


def transform_to_webm(video: np.ndarray, output_path: str,
                      fps: float, bitrate: str = "192k"):
    """
    Function to transform 2p gray scale video into a webm
    video using imageio_ffmpeg.
    Args:
        video: Video to be transformed with shape (time, row, col)
        output_path: Output path for the transformed video
        fps: desired fps of the output video

    Returns:

    """

    # ffmpeg expects the video shape in width, height not row, col
    # have to reverse shape when inputting
    # gray8 is uint8 format

    writer = mpg.write_frames(output_path,
                              video[0].shape[::-1],
                              pix_fmt_in="gray8",
                              pix_fmt_out="yuv420p",
                              codec="libvpx-vp9",
                              fps=fps,
                              bitrate=bitrate,
                              output_params=['-row-mt', '1'])
    writer.send(None)
    for frame in video:
        writer.send(frame)
    writer.close()
