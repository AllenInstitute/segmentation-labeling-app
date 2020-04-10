from typing import Tuple
import math


def get_magnitude(coord_one: Tuple[int, int],
                  coord_two: Tuple[int, int]):
    """
    Returns a distance between two points
    Args:
        coord_one: Tuple containing x and y, (x, y)
        coord_two: Tuple containing x and y, (x, y)
    Returns:
         d: Distance between the two points
    """
    distance = math.sqrt(math.pow(coord_one[0] - coord_two[0], 2) +
                         math.pow(coord_one[1] - coord_two[1], 2))
    return distance

