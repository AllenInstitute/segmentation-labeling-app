"""
This module is used for loading the ROIs from LIMs DB tables and
contains pre processing methods for ROIs loaded from LIMs. These methods
are used to define drawing parameters as well as other ROI class
methods useful for post processing required to display to end user.
"""
from typing import List, Tuple, Union

from scipy.sparse import coo_matrix
import scipy.ndimage
import numpy as np
import cv2

import segmentation_labeling_app.utils.query_utils as query_utils
from segmentation_labeling_app.utils.math_utils import get_magnitude

stroke_weight = 1.125


class ROI:

    def __init__(self,
                 coo_rows: Union[np.array, List[int]],
                 coo_cols: Union[np.array, List[int]],
                 coo_data: Union[np.array, List[float]],
                 image_shape: Tuple[int, int],
                 segmentation_id: int,
                 roi_id: int):
        self.coo_rows = coo_rows
        self.coo_cols = coo_cols
        self.coo_data = coo_data
        self.image_shape = image_shape
        self.segmentation_id = segmentation_id
        self.roi_id = roi_id
        self._sparse_coo = None
        self._dense_mat = None
        self._edges = None

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

    @property
    def sparse_coo(self):
        """
        Produces a sparse coo matrix scipy object from the
        Returns: A sparse_csc scipy object for the data
        """
        if self._sparse_coo is None:
            self._sparse_coo = coo_matrix((
                self.coo_data,
                (self.coo_rows,
                 self.coo_cols)),
                shape=self.image_shape)
        return self._sparse_coo

    @property
    def dense_matrix(self):
        """
        Produces dense matrix from spare_coo
        Returns: Dense matrix for ROI
        """
        if self._dense_mat is None:
            self._dense_mat = self.sparse_coo.toarray()
        return self._dense_mat

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
        return np.where(self.dense_matrix >= threshold, 1, 0)

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
        edge_coordinates = self.get_roi_edge_coordinates(threshold=threshold)
        center = self.get_centroid_of_thresholded_mask(threshold=threshold)
        inner_edge_mask = np.zeros(shape=self.image_shape)
        for edge_coordinate in edge_coordinates:
            # get distance to center
            vector_magnitude = get_magnitude(center, edge_coordinate)
            if vector_magnitude >= stroke_size:
                vector = np.array([(center[0] - edge_coordinate[0]),
                                   (center[1] - edge_coordinate[1])])
                unit_vector = vector / vector_magnitude
                """
                This is a little iffy, you overweight the movement vector a
                tiny bit, this is due to moving to much in one direction over the
                other when you're not at a 45 degree angle from the center. So
                we multiply it by a defined weight to overweight the movement
                just slightly to compensate for imperfect stroke movement.
                """
                movement_vector = stroke_weight * stroke_size * unit_vector
                new_position_x = int(round(edge_coordinate[0] + movement_vector[0]))
                new_position_y = int(round(edge_coordinate[1] + movement_vector[1]))
                new_position_vec = np.array([new_position_x, new_position_y])
                inner_edge_mask[new_position_vec[0]][new_position_vec[1]] = 1
            else:
                raise ValueError("Stroke size too large, cannot get inner edge "
                                 "coordinate with distance from center: "
                                 "%i, and stroke size: %i" % (vector_magnitude, stroke_size))
        # do edge finding again in case of inner edges existing
        # don't use defined functions to preserve more simple structure
        edge_struct = scipy.ndimage.generate_binary_structure(2, 2)
        inner_edge_erode = scipy.ndimage.binary_erosion(inner_edge_mask, edge_struct)
        eroded_inner_edge_mask = np.int8(inner_edge_mask) ^ inner_edge_erode
        return eroded_inner_edge_mask, np.argwhere(inner_edge_mask == 1)

    def get_edge_mask(self, threshold: float):
        """
        Returns a mask of edge points where edges of an roi are represented
        as 1 and non edges are 0. Computes edges using binary erosion and
        exclusive or operation. Generates edges from a thresholded binary mask.
        Args:
            Threshold to generate binary mask from
        Returns:
            Edges: a list of coordinates that contain edge points
        """
        edge_struct = scipy.ndimage.generate_binary_structure(2, 2)
        binary_mask = self.generate_binary_mask_from_threshold(threshold)
        erode = scipy.ndimage.binary_erosion(binary_mask, edge_struct)
        edge_mask = binary_mask ^ erode
        return edge_mask

    def get_roi_edge_coordinates(self, threshold: float):
        """
        Returns a list of edge points against a binary mask of an roi.
        Creates the edge points using binary erosion and exclusive or
        operations.
        Args:
            threshold: Float to threshold pixels against to create binary mask
        Returns:
            Edges: a list of coordinates that contain edge points
        """
        edge_mask = self.get_edge_mask(threshold=threshold)
        return np.argwhere(edge_mask == 1)

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
