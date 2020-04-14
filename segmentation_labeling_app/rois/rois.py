from typing import List, Tuple, Union

from scipy.sparse import coo_matrix
import numpy as np
import cv2

import segmentation_labeling_app.utils.query_utils as query_utils

stroke_weight = 1.125


class ROI:
    """
    This module is used for loading the ROIs from LIMs DB tables and
    contains pre processing methods for ROIs loaded from LIMs. These methods
    are used to define drawing parameters as well as other ROI class
    methods useful for post processing required to display to end user.
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
    def roi_from_query(cls, segmentation_id:int,
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

    def generate_binary_mask_from_threshold(self, threshold: float):
        """
        Simple binary mask from a provided threshold.
        Args:
            threshold: The threshold to compare values

        Returns:

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
        inner_edge_mask = np.zeros(shape=self.image_shape)
        for edge_coordinate in edge_coordinates[0]:
            # get distance to center
            vector = np.array([(center[0] - edge_coordinate[0][0]),
                               (center[1] - edge_coordinate[0][1])])
            vector_magnitude = np.linalg.norm(vector)
            if vector_magnitude >= stroke_size:
                unit_vector = vector / vector_magnitude
                """
                This is a little iffy, you overweight the movement vector a
                tiny bit, this is due to moving to much in one direction over the
                other when you're not at a 45 degree angle from the center. So
                we multiply it by a defined weight to overweight the movement
                just slightly to compensate for imperfect stroke movement.
                """
                movement_vector = stroke_weight * stroke_size * unit_vector
                new_position_x = int(round(edge_coordinate[0][0] + movement_vector[0]))
                new_position_y = int(round(edge_coordinate[0][1] + movement_vector[1]))
                new_position_vec = np.array([new_position_x, new_position_y])
                inner_edge_mask[new_position_vec[0]][new_position_vec[1]] = 1
            else:
                raise ValueError("Stroke size too large, cannot get inner edge "
                                 "coordinate with distance from center: "
                                 "%i, and stroke size: %i" % (vector_magnitude, stroke_size))
        # do edge finding again in case of inner edges existing
        # don't use defined functions to preserve more simple structure
        return inner_edge_mask, np.argwhere(inner_edge_mask == 1)

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
        return contours

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

        return center_x, center_y
