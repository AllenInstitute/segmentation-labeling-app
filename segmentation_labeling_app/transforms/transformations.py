from pathlib import Path
import random
from typing import Union, Tuple
import math

import h5py
import numpy as np
import imageio_ffmpeg as mpg


def downsample_h5_video(video_path: Union[Path], input_fps: int,
                        output_fps: int, strategy: str = 'random',
                        random_seed: int = 0) -> np.ndarray:
    """
    Function to down-sample an array stored in h5 format to a desired FPS.
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


def get_transformed_center(coordinate_pair: Tuple[int, int],
                           box_size: Tuple[int, int],
                           video_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Returns a coordinate pair that has been transformed to fit as closely to
    original position as possible but still with box contained totally in video.
    This is made to shift the coordinate if the subset box goes out of bound of
    the video shape. The shift moves the box fully into the frame while maintaining
    the center of the visual box is as close as possible to the original center.
    (example coordinate pair at 0, with box size 3 x 3)
    x x x x    x x x x
    x x x x    x x x x
    x x x x -> x x x x
    x x x x    x 0 x x
    x 0 x x    x x x x
    Args:
        coordinate_pair: the original coordinate location for centering (row, col)
        box_size: the size of the box for subset (row, col)
        video_shape: the x and y dimensions of the video (time, row, col)

    Returns:
        transformed_center: returns the best fitting center of the box for out
        of bounds coordinates (row, col)

    """
    if box_size[0] > video_shape[0] or box_size[1] > video_shape[1]:
        raise ValueError('Box is larger or equal in size in one or more'
                         'dimensions to original video shape. Provide a '
                         'box size smaller than original video.')
    else:
        # if the box size is less than frame size it can only be out of bounds in
        # direction left and right and one direction up and down
        transformed_col = coordinate_pair[1]
        transformed_row = coordinate_pair[0]
        # can't be centered left
        if coordinate_pair[0] < (math.ceil(box_size[0]/2) - 1):
            transformed_row = math.ceil(box_size[0] / 2) - 1

        # can't be centered up
        if coordinate_pair[1] < (math.ceil(box_size[1]/2) - 1):
            transformed_col = math.ceil(box_size[1] / 2) - 1

        # can't be centered right
        if (coordinate_pair[0] + (math.ceil(box_size[0]/2))) > video_shape[0] - 1:
            transformed_row = video_shape[0] - math.floor(box_size[0] / 2) - 1

        # can't be centered down
        if (coordinate_pair[1] + (math.ceil(box_size[1]/2))) > video_shape[1] - 1:
            transformed_col = video_shape[1] - math.floor(box_size[1] / 2) - 1

    return transformed_row, transformed_col


def get_centered_coordinate_box_video(coordinate_pair: Tuple[int, int],
                                      box_size: Tuple[int, int],
                                      video_array: np.ndarray):
    """
    Function to get video subset centered around coordinate pair with
    size specified by box size tuple. Function takes a subset of each frame
    and stacks together to get final video.
    Args:
        coordinate_pair: the coordinate on which to center the subset (row, col)
        box_size: the size of the subset box (row, col)
        video_array: the video to take the subset from array should be of
                     form (time, row, col)

    Returns:

    """
    if not len(video_array[0].shape) == 2:
        raise ValueError('Video does not have correct shape')
    transformed_coordinates = get_transformed_center(coordinate_pair, box_size,
                                                     video_array[0].shape)
    print(transformed_coordinates)
    left_column = transformed_coordinates[1] - math.floor(box_size[1] / 2)
    right_column = transformed_coordinates[1] + math.ceil(box_size[1] / 2)
    up_row = transformed_coordinates[0] - math.floor(box_size[0] / 2)
    down_row = transformed_coordinates[0] + math.ceil(box_size[0] / 2)
    transformed_video = np.zeros(shape=(video_array.shape[0], box_size[0], box_size[1]))
    for i, frame in enumerate(video_array):
        transformed_video[i] = frame[up_row:down_row, left_column:right_column]
    return np.uint8(transformed_video)


def generate_max_ave_proj_image(video: np.ndarray,
                                projection_type: str = 'average'):
    """
    Returns a maximum projection or an average projection of a video in
    numpy array format
    Args:
        video: The video to generate the projection with shape (time, row, col)
        projection_type: maximum or average, what type of projection

    Returns: A numpy array generated with the specified strategy (time, row, col)

    """
    if projection_type == 'average':
        return np.mean(video)
    elif projection_type == 'maximum':
        return np.max(video, axis=0)


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

