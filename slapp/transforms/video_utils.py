from pathlib import Path
from typing import Union
import h5py
import numpy as np
import imageio_ffmpeg as mpg
from slapp.transforms.array_utils import downsample_array
import multiprocessing
import tempfile
import subprocess


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


def transform_job(fargs):
    video, output_path, fps, bitrate = fargs

    writer = mpg.write_frames(output_path,
                              video[0].shape[::-1],
                              pix_fmt_in="gray8",
                              pix_fmt_out="yuv420p",
                              codec="libvpx-vp9",
                              fps=fps,
                              bitrate=bitrate)
    writer.send(None)
    for frame in video:
        writer.send(frame)
    writer.close()
    return output_path


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

    ncpu = multiprocessing.cpu_count()
    split_video = np.array_split(video, ncpu)
    outputs = [tempfile.NamedTemporaryFile(suffix='.webm')
               for i in range(ncpu)]
    args = []
    for vid, outpath in zip(split_video, outputs):
        args.append([vid, outpath.name, fps, bitrate])

    with multiprocessing.Pool(ncpu) as pool:
        results = pool.map(transform_job, args)

    listfile = tempfile.NamedTemporaryFile(suffix=".txt")
    with open(listfile.name, 'w') as fp:
        for fname in results:
            fp.write(f"file '{fname}'\n")

    sub_args = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i',
                listfile.name, '-c', 'copy', output_path]
    subprocess.run(sub_args)

    listfile.close()
    [o.close() for o in outputs]
