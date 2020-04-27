from typing import Union
import h5py
import numpy as np
import imageio_ffmpeg as mpg


def downsample_array(
        array: Union[h5py.Dataset, np.ndarray],
        input_fps: int = 31,
        output_fps: int = 4,
        strategy: str = 'average',
        random_seed: int = 0) -> np.ndarray:
    """Downsamples an array-like object along axis=0

    Parameters
    ----------
        array: h5py.Dataset or numpy.ndarray
            the input array
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
        new_video: A new video down-sampled to desired FPS with selected
        strategy (time, row, col)
    """
    if output_fps > input_fps:
        raise ValueError('Output FPS cannot be greater than input FPS')
    if (strategy == 'maximum') & (len(array.shape) > 1):
        raise ValueError("downsampling with strategy 'maximum' is not defined")

    npts_in = array.shape[0]
    npts_out = int(npts_in * output_fps / input_fps)
    bin_list = np.array_split(np.arange(npts_in), npts_out)

    array_out = np.zeros((npts_out, *array.shape[1:]))

    if strategy == 'random':
        rng = np.random.default_rng(random_seed)

        def sampler(arr, idx):
            return arr[rng.choice(idx)]
    elif strategy == 'maximum':

        def sampler(arr, idx):
            return arr[idx].max(axis=0)
    elif strategy == 'average':

        def sampler(arr, idx):
            return arr[idx].mean(axis=0)
    elif strategy == 'first':

        def sampler(arr, idx):
            return arr[idx[0]]
    elif strategy == 'last':

        def sampler(arr, idx):
            return arr[idx[-1]]

    for i, bin_indices in enumerate(bin_list):
        array_out[i] = sampler(array, bin_indices)

    return array_out


def normalize_video(video: np.ndarray):
    """
    Function to normalize video by its global max and maximum 8 bit value
    Args:
        video: Video to be normalized with shape (time, row, col)
    Returns:
        norm_frames: normalized video (time, row, col)
    """
    return np.uint8(video / video.max() * 255)


def transform_to_mp4(video: np.ndarray, output_path: str,
                     fps: int):
    """
    Function to transform 2p gray scale video into a mp4
    video using imageio_ffmpeg.
    Args:
        video: Video to be transformed with shape (time, row, col)
        output_path: Output path for the transformed video
        fps: desired fps of the output video

    Returns:

    """
    norm_video = normalize_video(video)
    writer = mpg.write_frames(output_path,
                              video[0].shape,
                              pix_fmt_in="gray",
                              pix_fmt_out="gray",
                              fps=fps)
    writer.send(None)
    for frame in norm_video:
        writer.send(frame)
    writer.close()
