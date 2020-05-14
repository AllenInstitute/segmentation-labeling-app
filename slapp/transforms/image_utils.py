from typing import Tuple, Union
from pathlib import Path
from math import ceil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_scale(image: Union[Path, np.ndarray],
              scale_position: Tuple[int, int],
              font_file: Path,
              resolution: float = 400 / 512,
              scale_size: int = 10,
              fill: Tuple[int, int, int] = (0, 0, 0),
              width: int = 5,
              font_size: int = 8) -> Image:
    """
    Adds a scale bar onto a image passed in the form of a image path. Uses
    PIL library ImageDraw and ImageFont to draw scale bar onto images.
    Parameters
    ----------
    image: Path
        path to the image to add scale bar
    scale_position: Tuple
        the position at which to set the bottom left corner of the scale. In
        x, y format, this is the format PIL expects.
    font_file: Path
        the font file to be used as denoted by a path object. On windows can
        just be specified as 'arial.ttf' on linux must be a full path.
    resolution: float = 10/13
        resolution of the video in microns per pixel, defaulted to
        scientifica general which is 10/13~.78 micros/pixel
    scale_size: int=10
        how many micro meters the scale bar should display, defaulted to 10
    fill: Tuple=(0, 0, 0)
        This parameter is passed to line draw in PIL libary and name reflects
        name in that API. The fill color in rgb tuple to draw the text and
        scale, defaults to black
    width: Int
        This parameter is passed to line draw in PIL libary and name reflects
        name in that API. The width of the lines of the scale in pixels
    font_size: Int
        the font size of the description text

    Returns
    -------
    scaled_image: Image
        PIL image object with the appended scale bar drawn on top
    """
    # open image as PIL Image object and create draw object
    if isinstance(image, Path):
        image = Image.open(image.as_posix()).convert('RGBA')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGBA')
    else:
        raise ValueError('Supplied image is not a matrix or a path to '
                         'an image file.')
    draw = ImageDraw.Draw(image)

    pixel_cnt = ceil(scale_size / resolution)
    # image draws down and to the right of given point
    scale_points = [(scale_position[0] + pixel_cnt, scale_position[1]),
                    scale_position,
                    (scale_position[0], scale_position[1] - pixel_cnt)]

    # draw the scale
    draw.line(xy=scale_points, fill=fill, width=width)

    font = ImageFont.truetype(font_file.as_posix(), font_size, encoding='unic')

    draw.text((scale_position[0] + pixel_cnt + 5,
               scale_position[1] - font_size),
              f"{scale_size} \u03BCm", font=font, fill=fill)

    return image
