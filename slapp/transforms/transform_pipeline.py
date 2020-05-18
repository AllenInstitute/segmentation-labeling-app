import datetime
import json
import multiprocessing
import os
from pathlib import Path

import argschema
import imageio
import marshmallow as mm
import numpy as np

import slapp.utils.query_utils as query_utils
from slapp.rois import ROI
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
        required=True,
        description=("which segmentation_run_id to transform. "
                     "If not provided, will attempt to query using args: "
                     "ophys_experiment_id, ophys_segmentation_commit_hash."))
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
    image_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.2,
        description=("lower quantile threshold for avg projection "
                     "histogram adjustment"))
    image_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=("upper quantile threshold for avg projection "
                     "histogram adjustment"))
    webm_bitrate = argschema.fields.Str(
        required=False,
        default="0",
        description="passed as bitrate to imageio-ffmpeg.write_frames()")
    font_file = argschema.fields.InputFile(
        required=False,
        default="/allen/aibs/informatics/isaak/fonts/arial.ttf",
        description=("Path to font file to be used in constructing the "
                     "videos and images")
    )
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

    @mm.pre_load
    def set_segmentation_run_id(self, data, **kwargs):
        """if segmentation_run_id not provided, query the database
        using the other provided args
        """
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


class TransformPipeline(argschema.ArgSchemaParser):
    default_schema = TransformPipelineSchema

    def run(self, db_conn: query_utils.DbConnection):
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = Path(self.args['artifact_basedir']) / \
            f"seg_run_id_{self.args['segmentation_run_id']}" / \
            self.timestamp
        os.makedirs(output_dir, exist_ok=True)

        # get all ROI ids from this segmentation run
        query_string = ("SELECT id FROM rois WHERE segmentation_run_id="
                        f"{self.args['segmentation_run_id']}")
        entries = db_conn.query(query_string)
        roi_ids = [i['id'] for i in entries]

        # TODO: here could be a good place to put a pre-filtering stepw
        rois = [ROI.roi_from_query(roi_id, db_conn) for roi_id in roi_ids]

        # load and downsample the source video
        query_string = ("SELECT * FROM segmentation_runs "
                        f"WHERE id={self.args['segmentation_run_id']}")
        seg_query = db_conn.query(query_string)[0]
        downsampled_video = downsample_h5_video(
                Path(seg_query['source_video_path']),
                self.args['input_fps'],
                self.args['output_fps'],
                self.args['downsampling_strategy'],
                self.args['random_seed'])

        # strategy for normalization: normalize entire video and projections
        # on quantiles of average projection before per-ROI processing
        avg_projection = np.mean(downsampled_video, axis=0)
        max_projection = np.max(downsampled_video, axis=0)
        quantiles = [self.args['image_lower_quantile'],
                     self.args['image_upper_quantile']]
        # normalize movie and avg according to avg quantiles
        lower_cutoff, upper_cutoff = np.quantile(
                avg_projection.flatten(), quantiles)
        avg_projection = normalize_array(
                avg_projection, lower_cutoff, upper_cutoff)
        downsampled_video = normalize_array(
                downsampled_video, lower_cutoff, upper_cutoff)
        # normalize max on its own quantiles, to avoid saturation
        lower_cutoff, upper_cutoff = np.quantile(
                max_projection.flatten(), quantiles)
        max_projection = normalize_array(
                max_projection, lower_cutoff, upper_cutoff)

        playback_fps = self.args['output_fps'] * self.args['playback_factor']

        # experiment-level artifact
        full_video_path = output_dir / "full_video.webm"
        transform_to_webm(
            video=downsampled_video, output_path=str(full_video_path),
            fps=playback_fps, ncpu=self.args['webm_parallelization'],
            bitrate=self.args['webm_bitrate'], crf=self.args['webm_quality'])

        # create the per-ROI artifacts
        insert_statements = []
        for roi in rois:
            # mask and outline from ROI class
            mask_path = output_dir / f"mask_{roi.roi_id}.png"
            outline_path = output_dir / f"outline_{roi.roi_id}.png"
            full_outline_path = output_dir / f"full_outline_{roi.roi_id}.png"
            sub_video_path = output_dir / f"video_{roi.roi_id}.webm"
            max_proj_path = output_dir / f"max_{roi.roi_id}.png"
            avg_proj_path = output_dir / f"avg_{roi.roi_id}.png"
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

            # add the scale to the outlines they will toggle with the outline
            # element on the web app
            scale_position = (3, outline.shape[1] - 3)
            scale_position_full = (12, full_outline.shape[1] - 12)
            font_file_path = Path(self.args['font_file'])

            outline_with_scale = add_scale(outline, scale_position,
                                           font_file_path)
            outline_with_scale.save(outline_path, 'PNG', transparency=0)

            full_outline_with_scale = add_scale(full_outline,
                                                scale_position_full,
                                                font_file_path,
                                                scale_size=40)
            full_outline_with_scale.save(full_outline_path, 'PNG',
                                         transparency=0)

            # video sub-frame
            inds, pads = content_extents(
                    roi._sparse_coo,
                    shape=self.args['cropped_shape'],
                    target_shape=tuple(downsampled_video.shape[1:]))
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
            imageio.imsave(max_proj_path, sub_max)
            imageio.imsave(avg_proj_path, sub_ave)

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
            manifest['experiment-id'] = seg_query['ophys_experiment_id']
            manifest['roi-id'] = roi.roi_id
            manifest['source-ref'] = str(outline_path)
            manifest['roi-mask-source-ref'] = str(mask_path)
            manifest['full-video-source-ref'] = str(full_video_path)
            manifest['video-source-ref'] = str(sub_video_path)
            manifest['max-source-ref'] = str(max_proj_path)
            manifest['avg-source-ref'] = str(avg_proj_path)
            manifest['trace-source-ref'] = str(trace_path)
            manifest['full-outline-source-ref'] = str(full_outline_path)

            insert_str = insert_str_template.format(
                    json.dumps(manifest),
                    os.environ['TRANSFORM_HASH'],
                    roi.roi_id)

            insert_statements.append(insert_str)

        db_conn.bulk_insert(insert_statements)


if __name__ == "__main__":  # pragma: no cover
    db_credentials = query_utils.get_db_credentials(
            env_prefix="LABELING_",
            **query_utils.label_defaults)
    db_connection = query_utils.DbConnection(**db_credentials)

    pipeline = TransformPipeline()
    pipeline.run(db_connection)
