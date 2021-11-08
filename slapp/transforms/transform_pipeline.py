import datetime
import h5py
import json
import jsonlines
import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple

import argschema
import imageio
import marshmallow as mm
import matplotlib.pyplot as plt
import numpy as np

import slapp.utils.query_utils as query_utils
from slapp.rois import ROI, coo_from_lims_style
from slapp.transforms.video_utils import (downsample_h5_video,
                                          transform_to_webm)
from slapp.transforms.array_utils import (
        content_extents, downsample_array, normalize_array)
from slapp.transforms.image_utils import (
    add_scale)


insert_str_template = (
        "INSERT INTO roi_manifests "
        "(manifest, transform_hash, roi_id) "
        "VALUES ('{}', '{}', {})")


class TransformPipelineException(Exception):
    pass


class TransformPipelineSchema(argschema.ArgSchema):
    segmentation_run_id = argschema.fields.Int(
        required=False,
        description=("which segmentation_run_id to transform. "
                     "If not provided, will attempt to query using args: "
                     "ophys_experiment_id, ophys_segmentation_commit_hash."))
    prod_segmentation_run_manifest = argschema.fields.InputFile(
        required=True,
        description=("A field which allows for a slapp manifest to be created "
                     "from a production segmentation run.")
    )
    output_manifest = argschema.fields.OutputFile(
        required=False,
        description="output path for jsonlines manifest contents")
    ophys_experiment_id = argschema.fields.Int(
        required=False,
        description=("used as a query filter if "
                     "segmentation_run_id is not provided."))
    ophys_segmentation_commit_hash = argschema.fields.Str(
        required=False,
        description=("used as a query filter if "
                     "segmentation_run_id is not provided."))
    artifact_basedir = argschema.fields.OutputDir(
        required=True,
        description=("artifacts will be written to "
                     "artifact_basedir/segmentation_run_id."))
    cropped_shape = argschema.fields.List(
        argschema.fields.Int,
        cli_as_single_argument=True,
        required=True,
        default=[128, 128],
        description="[h, w] of bounding box around ROI images/videos")
    quantile = argschema.fields.Float(
        required=True,
        default=0.1,
        description=("quantile threshold for outlining an ROI. "))
    input_fps = argschema.fields.Float(
        required=False,
        default=31.0,
        description="frames per second of input movie")
    output_fps = argschema.fields.Float(
        required=False,
        default=4.0,
        description="frames per second of downsampled movie")
    playback_factor = argschema.fields.Float(
        required=False,
        default=1.0,
        description=("webm FPS and trace pointInterval will adjust by this "
                     "factor relative to real time."))
    downsampling_strategy = argschema.fields.Str(
        required=False,
        default="average",
        validator=mm.validate.OneOf(['average', 'first', 'last', 'random']),
        description="what downsampling strategy to apply to movie and trace")
    random_seed = argschema.fields.Int(
        required=False,
        default=0,
        description="random seed to use if downsampling strategy is 'random'")
    movie_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.1,
        description=("lower quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    movie_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=("upper quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    projection_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.2,
        description=("lower quantile threshold for projection "
                     "histogram adjustment"))
    projection_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.99,
        description=("upper quantile threshold for projection "
                     "histogram adjustment"))
    webm_bitrate = argschema.fields.Str(
        required=False,
        default="0",
        description="passed as bitrate to imageio-ffmpeg.write_frames()")
    webm_quality = argschema.fields.Int(
        required=False,
        default=30,
        description=("Governs encoded video perceptual quality. "
                     "Can be from 0-63. Lower values mean higher quality. "
                     "Passed as crf to ffmpeg")
    )
    webm_parallelization = argschema.fields.Int(
        required=False,
        default=1,
        description=("Number of parallel processes to use for video encoding. "
                     "A value of -1 results in "
                     "using multiprocessing.cpu_count()")
    )
    scale_offset = argschema.fields.Int(
        required=False,
        default=3,
        description=("number of pixels scale corner is offset from "
                     "lower left in cropped field-of-view ROI outline"))
    full_scale_offset = argschema.fields.Int(
        required=False,
        default=12,
        description=("number of pixels scale corner is offset from "
                     "lower left in full field-of-view ROI outline"))
    scale_size_um = argschema.fields.Float(
        required=False,
        default=10.0,
        description=("length of scale bars in um in cropped field-of-view "
                     "ROI outline"))
    full_scale_size_um = argschema.fields.Float(
        required=False,
        default=40.0,
        description=("length of scale bars in um in full field-of-view "
                     "ROI outline"))
    um_per_pixel = argschema.fields.Float(
        required=False,
        default=0.78125,
        description="microns per pixel in the 2P source video")
    skip_movies = argschema.fields.Bool(
        required=False,
        default=False,
        description="for generating CNN inputs, can skip movies to speed up.")
    skip_traces = argschema.fields.Bool(
        required=False,
        default=False,
        description='Skip producing trace artifacts'
    )
    all_ROIs = argschema.fields.Bool(
        required=False,
        default=False,
        description=("if generating from prod manifest, makes artifacts for "
                     "all ROIs. For disk space reasons, probably only want "
                     "to do this with skip_movies=True."))

    @mm.pre_load
    def set_segmentation_run_id(self, data, **kwargs):
        """if segmentation_run_id not provided, query the database
        using the other provided args
        """
        # If we already have a prod manifest no need to query DB
        if "prod_segmentation_run_manifest" in data:
            return data

        if "segmentation_run_id" not in data:
            if not all(i in data for i in [
                    "ophys_experiment_id",
                    "ophys_segmentation_commit_hash"]):
                raise TransformPipelineException(
                        "if omitting arg segmentation_run_id, must "
                        "include args ophys_experiment_id and "
                        "ophys_segmentation_commit_hash")

            db_credentials = query_utils.get_db_credentials(
                    env_prefix="LABELING_",
                    **query_utils.label_defaults)
            db_connection = query_utils.DbConnection(**db_credentials)
            query_string = (
                "SELECT id FROM segmentation_runs WHERE "
                f"ophys_experiment_id={data['ophys_experiment_id']} AND "
                "ophys_segmentation_commit_hash="
                f"'{data['ophys_segmentation_commit_hash']}'")
            entries = db_connection.query(query_string)
            if len(entries) != 1:
                raise TransformPipelineException(
                    f"{query_string} did not return exactly 1 result")
            data['segmentation_run_id'] = entries[0]['id']
        return data

    @mm.post_load
    def set_webm_parallelization(self, data, **kwargs):
        if data["webm_parallelization"] == -1:
            data["webm_parallelization"] = multiprocessing.cpu_count()
        return data


def xform_from_slapp_db(db_conn: query_utils.DbConnection,
                        segmentation_run_id: int) -> Tuple[List[ROI], Path]:

    # get all ROI ids from this segmentation run
    query_string = ("SELECT id FROM rois WHERE segmentation_run_id="
                    f"{segmentation_run_id}")
    entries = db_conn.query(query_string)
    roi_ids = [i['id'] for i in entries]

    rois = [ROI.roi_from_query(roi_id, db_conn) for roi_id in roi_ids]

    query_string = ("SELECT * FROM segmentation_runs "
                    f"WHERE id={segmentation_run_id}")
    seg_query = db_conn.query(query_string)[0]

    return (rois, Path(seg_query['source_video_path']))


class ProdSegmentationRunManifestSchema(mm.Schema):
    experiment_id = mm.fields.Int(required=True)
    binarized_rois_path = argschema.fields.InputFile(required=True)
    # TODO: Consider using H5InputFile field from ophys_etl_pipelines
    traces_h5_path = mm.fields.Str(required=False)
    # TODO: Consider using H5InputFile field from ophys_etl_pipelines
    movie_path = argschema.fields.Str(required=True)
    local_to_global_roi_id_map = mm.fields.Dict(required=False,
                                                keys=mm.fields.Int(),
                                                values=mm.fields.Int())
    correlation_projection_path = argschema.fields.InputFile(
        required=True, description='Path to correlation projection pngs')

    @mm.post_load
    def load_rois(self, data, **kwargs) -> dict:
        with open(data['binarized_rois_path'], 'r') as f:
            data['binarized_rois'] = json.load(f)
        return data

    @mm.post_load
    def get_movie_frame_shape(self, data, **kwargs) -> dict:
        with h5py.File(data['movie_path'], 'r') as h5f:
            data['movie_frame_shape'] = h5f['data'].shape[1:]
        return data


def xform_from_prod_manifest(prod_manifest_path: str,
                             all_ROIs: bool,
                             include_trace=True) -> Tuple[List[ROI], Path]:
    with open(prod_manifest_path, 'r') as f:
        prod_manifest = json.load(f)
    prod_manifest = ProdSegmentationRunManifestSchema().load(prod_manifest)
    id_map = prod_manifest['local_to_global_roi_id_map']
    if all_ROIs:
        id_map = {roi['id']: roi['id']
                  for roi in prod_manifest['binarized_rois']}

    rois = []
    for roi in prod_manifest['binarized_rois']:
        if roi['id'] not in id_map:
            # only make manifests for listed ROIs
            continue

        roi_stamp = coo_from_lims_style(
                mask_matrix=roi['mask_matrix'],
                xoffset=roi['x'],
                yoffset=roi['y'],
                shape=prod_manifest['movie_frame_shape'])
        converted_roi_id = id_map[roi['id']]

        if include_trace:
            with h5py.File(prod_manifest['traces_h5_path'], 'r') as h5f:
                traces_id_order = list(h5f['roi_names'][:].astype(int))
                roi_trace = h5f['data'][traces_id_order.index(roi['id'])]
        else:
            roi_trace = None

        converted_roi = ROI(coo_rows=roi_stamp.row,
                            coo_cols=roi_stamp.col,
                            coo_data=roi_stamp.data.astype('uint8'),
                            image_shape=prod_manifest['movie_frame_shape'],
                            experiment_id=prod_manifest['experiment_id'],
                            roi_id=converted_roi_id,
                            trace=roi_trace,
                            is_binary=True)
        rois.append(converted_roi)

    return rois, Path(prod_manifest['movie_path'])


class TransformPipeline(argschema.ArgSchemaParser):
    default_schema = TransformPipelineSchema

    def run(self):
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        rois, video_path = xform_from_prod_manifest(
            prod_manifest_path=self.args['prod_segmentation_run_manifest'],
            all_ROIs=self.args['all_ROIs'],
            include_trace=not self.args['skip_traces']
        )
        output_dir = Path(self.args['artifact_basedir']) / \
            self.timestamp
        os.makedirs(output_dir, exist_ok=True)

        downsampled_video = downsample_h5_video(
                video_path,
                self.args['input_fps'],
                self.args['output_fps'],
                self.args['downsampling_strategy'],
                self.args['random_seed'])

        # strategy for normalization: normalize entire video and projections
        # on quantiles of average projection before per-ROI processing
        avg_projection = np.mean(downsampled_video, axis=0)
        max_projection = np.max(downsampled_video, axis=0)
        correlation_projection = plt.imread(self.args[
                                                'correlation_projection_path'])
        movie_quantiles = [self.args['movie_lower_quantile'],
                           self.args['movie_upper_quantile']]
        proj_quantiles = [self.args['projection_lower_quantile'],
                          self.args['projection_upper_quantile']]
        # normalize movie according to avg quantiles
        lower_cutoff, upper_cutoff = np.quantile(
                avg_projection.flatten(), movie_quantiles)
        if not self.args['skip_movies']:
            downsampled_video = normalize_array(
                    downsampled_video, lower_cutoff, upper_cutoff)
        # normalize avg projection
        lower_cutoff, upper_cutoff = np.quantile(
                avg_projection.flatten(), proj_quantiles)
        avg_projection = normalize_array(
                avg_projection, lower_cutoff, upper_cutoff)
        # normalize max projection
        lower_cutoff, upper_cutoff = np.quantile(
                max_projection.flatten(), proj_quantiles)
        max_projection = normalize_array(
                max_projection, lower_cutoff, upper_cutoff)

        playback_fps = self.args['output_fps'] * self.args['playback_factor']

        # experiment-level artifact
        if not self.args['skip_movies']:
            full_video_path = output_dir / "full_video.webm"
            transform_to_webm(
                video=downsampled_video, output_path=str(full_video_path),
                fps=playback_fps, ncpu=self.args['webm_parallelization'],
                bitrate=self.args['webm_bitrate'],
                crf=self.args['webm_quality'])

        # where to position the scales for the outlines
        scale_position = (
                self.args['scale_offset'],
                self.args['cropped_shape'][1] - self.args['scale_offset'])
        full_scale_position = (
                self.args['full_scale_offset'],
                max_projection.shape[1] - self.args['full_scale_offset'])

        # create the per-ROI artifacts
        insert_statements = []
        manifests = []
        for roi in rois:
            # mask and outline from ROI class
            mask_path = output_dir / f"mask_{roi.roi_id}.png"
            outline_path = output_dir / f"outline_{roi.roi_id}.png"
            full_outline_path = output_dir / f"full_outline_{roi.roi_id}.png"
            sub_video_path = output_dir / f"video_{roi.roi_id}.webm"
            max_proj_path = output_dir / f"max_{roi.roi_id}.png"
            avg_proj_path = output_dir / f"avg_{roi.roi_id}.png"
            corr_proj_path = output_dir / f"corr_{roi.roi_id}.png"
            trace_path = output_dir / f"trace_{roi.roi_id}.json"

            mask = roi.generate_ROI_mask(
                    shape=self.args['cropped_shape'])
            mask = np.uint8(mask * 255 / mask.max())
            outline = roi.generate_ROI_outline(
                shape=self.args['cropped_shape'],
                quantile=self.args['quantile'])
            full_outline = roi.generate_ROI_outline(
                shape=self.args['cropped_shape'],
                quantile=self.args['quantile'],
                full=True)

            imageio.imsave(mask_path, mask, transparency=0)

            outline = add_scale(
                    outline,
                    scale_position,
                    self.args['um_per_pixel'],
                    self.args['scale_size_um'],
                    color=0,
                    fontScale=0.3)
            full_outline = add_scale(
                    full_outline,
                    full_scale_position,
                    self.args['um_per_pixel'],
                    self.args['full_scale_size_um'],
                    color=0,
                    thickness_um=1.5,
                    fontScale=0.8)

            imageio.imsave(outline_path, outline, transparency=255)
            imageio.imsave(full_outline_path, full_outline, transparency=255)

            # video sub-frame
            inds, pads = content_extents(
                    roi._sparse_coo,
                    shape=self.args['cropped_shape'],
                    target_shape=tuple(downsampled_video.shape[1:]))
            if not self.args['skip_movies']:
                sub_video = np.pad(
                        downsampled_video[:, inds[0]:inds[1], inds[2]:inds[3]],
                        ((0, 0), *pads))
                transform_to_webm(
                    video=sub_video, output_path=str(sub_video_path),
                    fps=playback_fps, ncpu=self.args['webm_parallelization'],
                    bitrate=self.args['webm_bitrate'],
                    crf=self.args['webm_quality'])

            # sub-projections
            sub_max = np.pad(
                    max_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)
            sub_ave = np.pad(
                    avg_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)
            sub_corr = np.pad(
                correlation_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)
            imageio.imsave(max_proj_path, sub_max)
            imageio.imsave(avg_proj_path, sub_ave)
            imageio.imsave(corr_proj_path, sub_corr)

            if not self.args['skip_movies']:
                # trace
                trace = downsample_array(
                        np.array(roi.trace),
                        self.args['input_fps'],
                        self.args['output_fps'],
                        self.args['downsampling_strategy'],
                        self.args['random_seed']).tolist()
                trace_json = {
                        "pointStart": 0,
                        "pointInterval": 1.0 / playback_fps,
                        "dataLength": len(trace),
                        "trace": trace}
                with open(trace_path, "w") as fp:
                    json.dump(trace_json, fp)

            # manifest entry creation
            manifest = {}
            manifest['experiment-id'] = roi.experiment_id
            manifest['roi-id'] = roi.roi_id
            manifest['source-ref'] = str(outline_path)
            manifest['roi-mask-source-ref'] = str(mask_path)
            manifest['max-source-ref'] = str(max_proj_path)
            manifest['avg-source-ref'] = str(avg_proj_path)
            manifest['full-outline-source-ref'] = str(full_outline_path)
            if not self.args['skip_movies']:
                manifest['trace-source-ref'] = str(trace_path)
                manifest['full-video-source-ref'] = str(full_video_path)
                manifest['video-source-ref'] = str(sub_video_path)

            if 'output_manifest' in self.args:
                manifests.append(manifest)
            else:
                insert_str = insert_str_template.format(
                        json.dumps(manifest),
                        os.environ['TRANSFORM_HASH'],
                        roi.roi_id)

                insert_statements.append(insert_str)

        if 'output_manifest' in self.args:
            with open(self.args['output_manifest'], "w") as fp:
                jsonlines.Writer(fp).write_all(manifests)
        else:
            db_conn.bulk_insert(insert_statements)


if __name__ == "__main__":  # pragma: no cover
    pipeline = TransformPipeline()
    pipeline.run()
