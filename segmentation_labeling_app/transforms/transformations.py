from pathlib import Path
import random
from typing import Union, Tuple
import os.path
import math
import copy

import h5py
import cv2
import numpy as np
import imageio_ffmpeg as mpg


def downsample_h5_video(video_path: Union[str, Path], input_fps: int,
                     output_fps: int, strategy: str = 'random') -> np.ndarray:
    """
    Function to down-sample an array stored in h5 format to a desired FPS.
    Follows the strategy specified in the function call, current strategies are
    (random, maximum, average). Random takes a random sample of one frame from
    window, maximum takes the maximum frame in window, average takes the mean
    of window.
    Args:
        video_path: The path to the input video in str or Path format
        input_fps: The FPS or Hz of the input video
        output_fps: The desired output FPS or Hz
        strategy: The down-sampling startegy to follow

    Returns:
        new_video: A new video down-sampled to desired FPS with selected
        strategy
    """
    if os.path.exists(video_path):
        path_parts = str(video_path).split('.')
        extension = path_parts[len(path_parts) - 1]
        if extension == 'h5':
            with h5py.File(video_path, 'r') as open_video:
                video = open_video['data'][:]
                return _down_sample_array(full_array=video,
                                          input_fps=input_fps,
                                          output_fps=output_fps,
                                          strategy=strategy)
        else:
            raise AttributeError('Video path provided not an h5 file, please'
                                 'provide an h5 file')
    else:
        raise FileNotFoundError('Path specified doesnt exist or is not accessible')


def _down_sample_array(full_array: np.ndarray, input_fps: int,
                       output_fps: int, strategy: str = 'random') -> np.ndarray:
    """
    Function to down-sample a numpy array to a desired FPS. Follows the
    strategy specified in the function call, current strategies are (random,
    maximum, average). Random takes a random sample of one frame from window,
    maximum takes the maximum frame in window, average takes the mean of window.
    Args:
        full_array: The video array in numpy format
        input_fps: The FPS or Hz of the input video
        output_fps: The desired output FPS or Hz
        strategy: The down-sampling startegy to follow

    Returns:
        new_video: A new video down-sampled to desired FPS with selected
        strategy
    """
    if output_fps > input_fps:
        raise ValueError('Output FPS cannot be greater than input FPS')
    sampling_ratio = int(input_fps / output_fps)
    new_frames = []
    for i in range(0, len(full_array), sampling_ratio):
        frames = full_array[i:(i+sampling_ratio)]
        if strategy == 'random':
            new_frames.append(frames[random.randint(0, len(frames) - 1)])
        elif strategy == 'maximum':
            new_frames.append(np.max(frames, axis=0))
        elif strategy == 'average':
            new_frames.append(np.mean(frames))
    return np.stack(new_frames)


def get_transformed_center(coordinate_pair: Tuple[int, int],
                           box_size: Tuple[int, int],
                           video_shape: Tuple[int, int]):
    """
    Returns a coordinate pair that has been transformed to fit as closely to
    original position as possible but still with box contained totally in video.
    This is made to shift the coordinate if the subset box goes out of bound of
    the video shape.
    Args:
        coordinate_pair: the original coordinate location for centering
        box_size: the size of the box for subset
        video_shape: the shape of the original video

    Returns:

    """
    if box_size[0] < video_shape[0] and box_size[1] < video_shape[1]:
        # if the box size is less than frame size it can only be out of bounds in
        # direction left and right and one direction up and down
        transformed_x = coordinate_pair[0]
        transformed_y = coordinate_pair[1]
        # can't be centered left
        if coordinate_pair[0] < (math.ceil(box_size[0]/2) - 1):
            transformed_x = math.ceil(box_size[0] / 2) - 1

        # can't be centered up
        if coordinate_pair[1] < (math.ceil(box_size[1]/2) - 1):
            transformed_y = math.ceil(box_size[1] / 2) - 1

        # can't be centered right
        if (coordinate_pair[0] + (math.ceil(box_size[0]/2))) > video_shape[0] - 1:
            transformed_x = video_shape[0] - math.ceil(box_size[0] / 2) - 1

        # can't be centered down
        if (coordinate_pair[1] + (math.ceil(box_size[1]/2))) > video_shape[1] - 1:
            transformed_y = video_shape[1] - math.ceil(box_size[1] / 2) - 1

        return transformed_x, transformed_y

    else:
        raise ValueError('Box is larger or equal in size in one or more'
                         'dimensions to original video shape. Provide a '
                         'box size smaller than original video.')


def get_centered_coordinate_box(coordinate_pair: Tuple[int, int],
                                box_size: Tuple[int, int],
                                video_array: np.ndarray):
    """
    Function to get video subset centered around coordinate pair with
    size specified by box size tuple. Function takes a subset of each frame
    and stacks together to get final video.
    Args:
        coordinate_pair: the coordinate on which to center the subset
        box_size: the size of the subset box
        video_array: the video to take the subset from

    Returns:

    """
    transformed_coordinates = get_transformed_center(coordinate_pair, box_size,
                                                     video_array.shape)
    left_column = transformed_coordinates[0] - math.ceil(box_size[0] / 2)
    right_column = transformed_coordinates[0] + math.ceil(box_size[0] / 2)
    up_row = transformed_coordinates[1] - math.ceil(box_size[1] / 2)
    down_row = transformed_coordinates[1] + math.ceil(box_size[1] / 2)
    transformed_stack = []
    for frame in video_array:
        transformed_frame = frame[up_row:down_row, left_column:right_column]
        transformed_stack.append(transformed_frame)
    transformed_video = np.stack(transformed_stack)
    return transformed_video


def generate_max_ave_proj_image(video: np.ndarray, projection_type: str= 'average'):
    """
    Returns a maximum projection or an average projection of a video in
    numpy array format
    Args:
        video: The video to generate the projection
        projection_type: maximum or average, what type of projection

    Returns: A numpy array generated with the specified strategy

    """
    if projection_type == 'average':
        return np.mean(video)
    elif projection_type == 'maximum':
        return np.max(video, axis=0)


def normalize_video(video: np.ndarray):
    """
    Function to normalize video by its global max and maximum 8 bit value
    Args:
        video: Video to be normalized
    Returns:
        norm_frames: normalized video
    """
    global_max = video.max()
    norm_frames = []
    for frame in video:
        norm_frame = copy.deepcopy(frame)
        norm_frame = np.uint8(norm_frame / global_max * 255)
        norm_frames.append(norm_frame)
    norm_frames = np.stack(norm_frames)
    return norm_frames


def transform_to_mp4(video: np.ndarray, output_path: str):
    """
    Function to transform 2p gray scale video into a mp4
    video using opencv.
    Args:
        video: Video to be transformed
        output_path: Output path for the transformed video

    Returns:

    """
    writer = mpg.write_frames(output_path,
                              video[0].shape,
                              pix_fmt_in="gray",
                              pix_fmt_out="gray",
                              fps=30)
    writer.send(None)
    norm_video = normalize_video(video)
    for frame in video:
        frame = np.uint8(frame / frame.max() * 255)
        writer.send(frame)
    writer.close()

