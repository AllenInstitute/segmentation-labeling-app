from pathlib import Path
import random
from typing import Union, Tuple
import math

import h5py
import numpy as np
import imageio_ffmpeg as mpg


def downsample_h5_video(video_path: Union[Path], input_fps: int = 31,
                        output_fps: int = 4, strategy: str = 'average',
                        random_seed: int = 0) -> np.ndarray:
    """
    Function to downsample an array stored in h5 format to a desired FPS.
    Follows the strategy specified in the function call, current strategies are
    (random, maximum, average). Random takes a random sample of one frame from
    window, maximum takes the maximum frame in window, average takes the mean
    of window.
    Args:
        video_path: The path to the input video in Path format
        input_fps: The FPS or Hz of the input video
        output_fps: The desired output FPS or Hz
        strategy: The down-sampling strategy to follow
        random_seed: The seed for generating random variables if random strategy
        is followed

    Returns:
        new_video: A new video down-sampled to desired FPS with selected
        strategy (time, row, col)
    """
    if not video_path.exists():
        raise FileNotFoundError('Path specified doesnt exist or is not accessible')
    if not video_path.suffix == '.h5':
        raise AttributeError('Video path provided not an h5 file, please'
                             'provide an h5 file')

    with h5py.File(video_path, 'r') as open_video:
        video = open_video['data']
        return _downsample_array(full_array=video,
                                 input_fps=input_fps,
                                 output_fps=output_fps,
                                 strategy=strategy,
                                 random_seed=random_seed)


def _downsample_array(full_array: h5py.Dataset, input_fps: int,
                      output_fps: int, strategy: str = 'random',
                      random_seed: int = 0) -> np.ndarray:
    """
    Function to down-sample a numpy array to a desired FPS. Follows the
    strategy specified in the function call, current strategies are (random,
    maximum, average, first, and last). Random takes a random sample of one
    frame from window, maximum takes the maximum frame in window, average takes
    the mean of window, first takes the first frame, last takes the last frame.
    Args:
        full_array: The video array in h5 dataset format (time, row, col)
        input_fps: The FPS or Hz of the input video
        output_fps: The desired output FPS or Hz
        strategy: The down-sampling strategy to follow
        random_seed: If random strategy is followed, seed controls the random
        variable generation

    Returns:
        new_video: A new video down-sampled to desired FPS with selected
        strategy (time, row, col)
    """
    if output_fps > input_fps:
        raise ValueError('Output FPS cannot be greater than input FPS')
    downsampled_ratio = input_fps / output_fps
    downsampled_bin_cnt = len(full_array) / downsampled_ratio

    downsampled_bins = np.array_split(full_array, downsampled_bin_cnt)
    new_frames = np.zeros(shape=(len(downsampled_bins), full_array.shape[1],
                                 full_array.shape[2]))
    if strategy == 'random':
        random.seed(random_seed)

    for i, downsampled_bin in enumerate(downsampled_bins):
        if strategy == 'random':
            rand_idx = random.randint(0, len(downsampled_bin) - 1)
            new_frames[i] = downsampled_bin[rand_idx]
        elif strategy == 'maximum':
            new_frames[i] = np.max(downsampled_bin, axis=0)
        elif strategy == 'average':
            new_frames[i] = np.mean(downsampled_bin, axis=0)
        elif strategy == 'first':
            new_frames[i] = downsampled_bin[0]
        elif strategy == 'last':
            new_frames[i] = downsampled_bin[len(downsampled_bin) - 1]
    return new_frames


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
