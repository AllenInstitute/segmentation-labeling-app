from typing import List, Tuple, Union

from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import numpy as np
import cv2

import segmentation_labeling_app.utils.query_utils as query_utils

stroke_weight = 1.125


class ROI:
    """Class is used for manipulating ROI from LIMs for serving to labeling app

    This class is used for loading the ROIs from LIMs DB tables and
    contains pre processing methods for ROIs loaded from LIMs. These methods
    are used to define drawing parameters as well as other ROI class
    methods useful for post processing required to display to end user.
    Attributes:
        image_shape: the shape of the image the roi is contained within
        segmentation_id: the unique id for the segmentation run
        roi_id: the unique id for the ROI in the segmentation run
        _sparse_coo: the sparse matrix containing the probability mask
        for the ROI
    """

    def __init__(self,
                 coo_rows: Union[np.array, List[int]],
                 coo_cols: Union[np.array, List[int]],
                 coo_data: Union[np.array, List[float]],
                 image_shape: Tuple[int, int],
                 segmentation_id: int,
                 roi_id: int):
        self.image_shape = image_shape
        self.segmentation_id = segmentation_id
        self.roi_id = roi_id
        self._sparse_coo = coo_matrix((coo_data, (coo_rows, coo_cols)),
                                      shape=image_shape)

    @classmethod
    def roi_from_query(cls, segmentation_id: int,
                       roi_id: int) -> "ROI":
        """
        Queries and builds ROI object by querying LIMS table for
        produced labeling ROIs.
        Args:
            segmentation_id: Id of the segmentation run
            roi_id: Unique Id of the ROI to be loaded

        Returns: ROI object for the given segmentation_id and roi_id
        """
        label_vars = query_utils.get_labeling_env_vars()

        shape = query_utils.query(
            f"SELECT * FROM public.segmentation_runs WHERE id={segmentation_id}",
            user=label_vars.user,
            host=label_vars.host,
            database=label_vars.database,
            port=label_vars.port,
            password=label_vars.password)[0]['video_shape']

        roi = query_utils.query(
            f"SELECT * FROM public.rois WHERE "
            f"segmentation_run_id={segmentation_id} AND id={roi_id}",
            user=label_vars.user,
            host=label_vars.host,
            database=label_vars.database,
            port=label_vars.port,
            password=label_vars.password)
        return ROI(coo_rows=roi['coo_rows'][0],
                   coo_cols=roi['coo_cols'][0],
                   coo_data=roi['coo_data'][0],
                   image_shape=shape,
                   segmentation_id=segmentation_id,
                   roi_id=roi_id)

    def generate_binary_mask_from_threshold(self, threshold: float) -> np.array:
        """
        Simple binary mask from a provided threshold.
        Args:
            threshold: The threshold to compare values

        Returns:
            2D numpy array with values greater than threshold == 1 less than
            threshold == 0
        Notes:
            This function will likely become deprecated as we work on more
            applicable binary thresh holding
        """
        return np.where(self._sparse_coo.toarray() >= threshold, 1, 0)

    def generate_roi_inner_edge_coordinates(self, stroke_size: int,
                                            threshold: float):
        """
        Returns a list of coordinates in a video frame for the inner outline
        of an ROI given a desired stroke size. Looks at each border pixel and
        moves away from edge by stroke specified pixels. This process is completed
        by defining the center of the roi and creating a unit vector from the
        center to the edge point. This unit vector is then multiplied by the stroke
        and a slight weight to correct for non int centers. The movement vector
        is added to the edge point and the new point is then set to true in
        the return mask.
        Args:
            stroke_size: Value in pixels for the size of the stroke
            threshold: Float to threshold every pixel against to generate
            binary mask

        Returns:
            eroded_inner_mask: A binary mask for the new edge coordinates
            that has been slightly eroded
            coordinate_list: A list of coordinates for the edge of the ROI
            pushed in by stroke pixel count
        """
        edge_coordinates = self.get_edge_points(threshold=threshold)
        center = self.get_centroid_of_thresholded_mask(threshold=threshold)
        inner_edge_mask = np.zeros(self.image_shape)
        distance_vectors = np.transpose(cdist(center, edge_coordinates, 'euclidean'))
        vectors = np.array(list(zip(edge_coordinates, distance_vectors)))
        for vector_pair in vectors:
            # get distance to center
            if vector_pair[1][0] >= stroke_size:
                # get vector from edge to center
                vector_from_center_to_edge = np.array([(center[0][0] - vector_pair[0][0]),
                                                      (center[0][1] - vector_pair[0][1])])
                # get unit vector
                unit_vector = vector_from_center_to_edge / vector_pair[1][0]
                """
                This is a little iffy, you overweight the movement vector a
                tiny bit, this is due to moving to much in one direction over the
                other when you're not at a 45 degree angle from the center. So
                we multiply it by a defined weight to overweight the movement
                just slightly to compensate for imperfect stroke movement.
                (with correct weighting)
                x x x x x x     x x x x x x
                x o o o o x     x x x x x x
                x o o o o x     x x o o x x
                x o o o o x - > x x o o x x
                x o o o o x     x x x x x x
                x x x x x x     x x x x x x
                (without correct weighting)
                x x x x x x     x x x x x x
                x o o o o x     x x x o x x
                x o o o o x     x x o o x x
                x o o o o x - > x o o o x x
                x o o o o x     x x x x x x
                x x x x x x     x x x x x x
                It's caused by some angles from center generating a unit vector
                that is more pointed in one direction over the other, just under
                the rounding value. So a unit vector will be (0.7, 0.44) and this
                becomes (1, 0) when rounded, we want (1, 1) in order to eliminate
                outer parts
                """
                # multiply unit vector by movement amount
                movement_vector = stroke_weight * stroke_size * unit_vector
                # create new position by adding movement vector a edge position
                new_position_x = int(round(vector_pair[0][0] + movement_vector[0]))
                new_position_y = int(round(vector_pair[0][1] + movement_vector[1]))
                new_position_vec = np.array([new_position_x, new_position_y])
                # add new position to mask by making value 1
                inner_edge_mask[new_position_vec[0]][new_position_vec[1]] = 1
            else:
                raise ValueError("Stroke size too large, cannot get inner edge "
                                 "coordinate with distance from center: "
                                 "%i, and stroke size: %i" % (vector_pair[1][0],
                                                              stroke_size))
        inner_edge_points = np.argwhere(inner_edge_mask == 1)
        return inner_edge_points

    def get_edge_points(self, threshold: float):
        """
        Returns a mask of edge points where edges of an roi are represented
        as 1 and non edges are 0. Computes edges using binary erosion and
        exclusive or operation. Generates edges from a thresholded binary mask.
        Args:
            threshold: a float to threshold all values against to create binary
            mask
        Returns:
            Contours: a list of coordinates that contain edge points
        """
        binary_mask = self.generate_binary_mask_from_threshold(threshold)
        contours, hierarchy = cv2.findContours(np.uint8(binary_mask),
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
        return np.squeeze(contours[0], axis=1)

    def get_centroid_of_thresholded_mask(self, threshold: float):
        """
        Returns a tuple containing the center x and center y of an
        object in 2d picture. Calculates the moments of the image and
        uses the ratios to calculate the center.
        Args:
            threshold: Float to threshold pixels against to create binary mask
        Returns:
            tuple[c_x, c_y], a tuple containing the center of the roi from the
            thresholded binary mask
        """
        threshold_mask = self.generate_binary_mask_from_threshold(threshold)
        moments = cv2.moments(np.float32(threshold_mask))

        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])

        return np.array([[center_x, center_y]])
