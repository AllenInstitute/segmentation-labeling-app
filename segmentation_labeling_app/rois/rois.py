from typing import List, Tuple, Union

from scipy.sparse import coo_matrix
import numpy as np
import cv2

import segmentation_labeling_app.utils.query_utils as query_utils


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

    def create_dilated_contour_mask(self, stroke_size: int,
                                    threshold: float) -> np.array:
        """
        Returns a mask for a stroke size and threshold. An edge mask is
        calculated from the threshold. This mask is then dilated with dilation
        iterations for all values from 0 to stroke_size. All but the edges
        are masked away and added to the previous edge mask.
        This mask is then binarized where each pixel above or equal to 1
        becomes 1. This produces a border of size stroke_size around the
        original edge mask.
        Args:
            stroke_size: Value in pixels for the size of the stroke
            threshold: Float to threshold every pixel against to generate
            binary mask

        Returns:
            final_mask: the edge binary mask around the ROI where 1 is border
            and 0 is non border
        """
        edge_mask = self.get_edge_mask(threshold=threshold)
        final_mask = np.zeros(self.image_shape)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    ksize=(3, 3))
        for i in range(0, stroke_size):
            dilated_mask = cv2.dilate(edge_mask, dilation_kernel,
                                      iterations=i)
            dilated_mask = self.get_edge_mask(threshold=1,
                                              optional_array=dilated_mask)
            final_mask = final_mask + dilated_mask
        return np.where(final_mask >= 1, 1, 0)

    def get_edge_points(self, threshold: float,
                        optional_array: np.array = None):
        """
        Returns a list of edge points. Computes edges using binary erosion and
        exclusive or operation. Generates edges from a thresholded binary mask.
        Args:
            threshold: a float to threshold all values against to create binary
                       mask
            optional_array: an optional array to overload the function and
            create edge points from different mask
        Returns:
            Contours: a list of coordinates that contain edge points
        """
        if optional_array is None:
            optional_array = self.generate_binary_mask_from_threshold(threshold)
        contours, hierarchy = cv2.findContours(np.uint8(optional_array),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        return np.squeeze(contours[0], axis=1)

    def get_edge_mask(self, threshold: float,
                      optional_array: np.array = None) -> np.array:
        """
        Returns a mask of edge points where edges of an roi are represented
        as 1 and non edges are 0. Computes edges using binary erosion and
        exclusive or operation. Generates edges from a thresholded binary mask.
        Args:
            threshold: a float to threshold all values against to create binary
                       mask
            optional_array: an optional array to overload the function and
            create edge points from different mask

        Returns:
            mask_matrix: a binary mask of the edges given a threshold

        """
        edge_coordinates = self.get_edge_points(threshold=threshold,
                                                optional_array=optional_array)
        mask_matrix = np.zeros(shape=self.image_shape)
        for edge_coordinate in edge_coordinates:
            mask_matrix[edge_coordinate[0]][edge_coordinate[1]] = 1
        return mask_matrix
