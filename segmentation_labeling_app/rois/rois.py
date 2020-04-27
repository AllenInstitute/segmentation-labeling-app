from typing import List, Tuple, Union
import json
from pathlib import Path
from scipy.sparse import coo_matrix
import numpy as np
import cv2
import sqlite3

import segmentation_labeling_app.utils.query_utils as query_utils
from segmentation_labeling_app.transforms.array_utils import (
        center_pad_2d, crop_2d_array)


sql_create_rois_table = """ CREATE TABLE IF NOT EXISTS rois_prelabeling (
                            id integer PRIMARY KEY,
                            transform_hash text NOT NULL,
                            ophys_segmentation_commit_hash text NOT NULL,
                            creation_date text NOT NULL,
                            upload_date text,
                            manifest text NOT NULL,
                            experiment_id integer NOT NULL,
                            roi_id integer NOT NULL);"""

sql_insert_roi = """ INSERT INTO rois_prelabeling (transform_hash,
                     ophys_segmentation_commit_hash,
                     creation_date, upload_date, manifest, experiment_id,
                     roi_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?) """


def binary_mask_from_threshold(
            arr: Union[np.ndarray, coo_matrix],
            absolute_threshold: float = None,
            quantile: float = 0.1) -> np.array:
    """Binarize an array

    Parameters
    ----------
    arr: numpy.ndarray or scipy.sparse.coo_matrix
        2D array of weighted floating-point values
    absolute_threshold: float
        weighted entries above this value=1, below=0
        over-ridden if quantile is set
    quantile: float
        if specified, set absolute threshold np.quantile(arr.data, quantile)

    Returns
    -------
    binary: numpy.ndarray
        binarized mask

    """
    wmask = arr
    if isinstance(arr, coo_matrix):
        wmask = arr.toarray()
    vals = wmask[np.nonzero(wmask)]

    if quantile is not None:
        absolute_threshold = np.quantile(vals, quantile)

    binary = np.uint8(wmask > absolute_threshold)

    return binary


def sized_mask(
        arr: Union[np.ndarray, coo_matrix], shape: Tuple[int, int] = None,
        full: bool = False):
    """return a 2D dense array representation of the mask, optionally
    cropped and padded

    Parameters
    ----------
    arr: numpy.ndarray or scipy.sparse.coo_matrix:
        a representation of the mask
    shape: tuple(int, int)
        [h, w] for padded shape. If None, cropped to existing values
    full: bool
        if True, the full-frame array is returned

    Returns
    -------
    mask: numpy.ndarray
        2D dense matrix representation of the mask

    """
    if isinstance(arr, coo_matrix):
        mask = arr.toarray()
    else:
        mask = arr
    if not full:
        mask = crop_2d_array(mask)
        if shape is not None:
            mask = center_pad_2d(mask, shape)
    return mask


class ROI:
    """Class is used for manipulating ROI from LIMs for serving to labeling app

    This class is used for loading the ROIs from LIMs DB tables and
    contains pre processing methods for ROIs loaded from LIMs. These methods
    are used to define drawing parameters as well as other ROI class
    methods useful for post processing required to display to end user.
    Attributes:
        image_shape: the shape of the image the roi is contained within
        experiment_id: the unique id for the segmentation run
        roi_id: the unique id for the ROI in the segmentation run
        _sparse_coo: the sparse matrix containing the probability mask
        for the ROI
    """

    def __init__(self,
                 coo_rows: Union[np.array, List[int]],
                 coo_cols: Union[np.array, List[int]],
                 coo_data: Union[np.array, List[float]],
                 image_shape: Tuple[int, int],
                 experiment_id: int,
                 roi_id: int,
                 trace: Union[np.array, List[float]] = None,
                 ):
        self.image_shape = image_shape
        self.experiment_id = experiment_id
        self.roi_id = roi_id
        self._sparse_coo = coo_matrix((coo_data, (coo_rows, coo_cols)),
                                      shape=image_shape)
        self.trace = trace

    @classmethod
    def roi_from_query(cls, roi_id: int,
                       db_conn: query_utils.DbConnection) -> "ROI":
        """
        Queries and builds ROI object by querying LIMS table for
        produced labeling ROIs.
        Args:
            roi_id: Unique Id of the ROI to be loaded

        Returns: ROI object for the given segmentation_id and roi_id
        """

        roi = db_conn.query(f"SELECT * FROM rois WHERE id={roi_id}")[0]

        segmentation_run = db_conn.query(
            ("SELECT * FROM segmentation_runs WHERE "
             f"id={roi['segmentation_run_id']}"))[0]

        # NOTE: temporary legacy support for no traces
        # will remove and delete legacy traceless ROI
        # entries when all agreed.
        trace = None
        if 'trace' in roi:
            trace = roi['trace']

        return ROI(coo_rows=roi['coo_row'],
                   coo_cols=roi['coo_col'],
                   coo_data=roi['coo_data'],
                   image_shape=segmentation_run['video_shape'],
                   experiment_id=segmentation_run['ophys_experiment_id'],
                   roi_id=roi_id,
                   trace=trace
                   )

    def generate_ROI_mask(
            self, shape: Tuple[int, int] = None, full: bool = False):
        """return a 2D dense representation of the mask

        Parameters
        ----------
        shape: tuple(int, int)
            [h, w] for padded shape. If None, cropped to existing values
        full: bool
            if True, the full-frame array is returned

        Returns
        -------
        mask: numpy.ndarray
            2D dense representation of the mask

        """
        mask = sized_mask(self._sparse_coo.toarray(), shape=shape, full=full)

        return mask

    def generate_ROI_outline(
            self, shape: Tuple[int, int] = None,
            full: bool = False,
            absolute_threshold: float = None,
            quantile: float = 0.1,
            dilation_kernel_size: int = 1, inner_outline: bool = True):
        """return a 2D dense representation of the mask outline

        Parameters
        ----------
        shape: tuple(int, int)
            [h, w] for padded shape. If None, cropped to existing values
        full: bool
            if True, the full-frame array is returned
        absolute_threshold: float
            weighted entries above this value=1, below=0
            over-ridden if quantile is set
        quantile: float
            if specified, set absolute threshold by using np.quantile()
            with this value as the quantile arg
        dilation_kernel_size: int
            passed as size to cv2.getStructuringElement()
        inner_outline: bool
            whether to retain only outline points interior to the mask

        Returns
        -------
        mask: numpy.ndarray
            2D dense representation of the mask outline

        """
        binary = binary_mask_from_threshold(
            self._sparse_coo,
            absolute_threshold=absolute_threshold,
            quantile=quantile)

        contours, _ = cv2.findContours(binary,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        xy = np.concatenate(contours).squeeze()

        mask = np.zeros_like(binary)
        mask[xy[:, 1], xy[:, 0]] = 1.0

        kernel = np.ones((dilation_kernel_size, dilation_kernel_size))

        mask = cv2.dilate(mask, kernel)

        if inner_outline:
            mask = mask & binary

        mask = sized_mask(mask, shape=shape, full=full)

        return mask

    def create_manifest_json(self, source_ref: str,
                             video_source_ref: str, max_source_ref: str,
                             avg_source_ref: str, trace_source_ref: str,
                             roi_data_source_ref: str) -> str:
        """
        Function to make the manifest json string required for sagemaker
        ground truth to read the data in successfully. The strings are
        local URIs for each of the required data.
        Args:
            source_ref: Location of ROI mask png
            video_source_ref: Location of 2P video for the ROI
            max_source_ref: Location of the maximum projection for the ROI
            avg_source_ref: Location of the average projection for the ROI
            trace_source_ref: Location of the trace data for the ROI
            roi_data_source_ref: Location of the ROI coordinate data

        Returns:
            dictionary: returns dictionary as json string
        """
        dictionary = {'source_ref': source_ref,
                      'video_source_ref': video_source_ref,
                      'max_source_ref': max_source_ref,
                      'avg_source_ref': avg_source_ref,
                      'trace_source_ref': trace_source_ref,
                      'roi_data_source-ref': roi_data_source_ref,
                      'roi_id': self.roi_id,
                      'experiment_id': self.experiment_id}
        return json.dumps(dictionary)

    def write_roi_to_db(self, database_file_path: Path,
                        transform_hash: str,
                        ophys_segmentation_commit_hash: str,
                        creation_date: str,
                        manifest: str,
                        upload_date: str = None):
        # connect to the database file
        db_connection = sqlite3.connect(database_file_path.as_posix())

        # create table, sql query handles the checking if already exists
        curr = db_connection.cursor()
        curr.execute(sql_create_rois_table)

        roi_task = (transform_hash, ophys_segmentation_commit_hash,
                    creation_date, upload_date, manifest,
                    self.experiment_id, self.roi_id)
        # add the roi to the table
        curr.execute(sql_insert_roi, roi_task)
        unique_id = curr.lastrowid
        db_connection.commit()
        db_connection.close()

        return unique_id
