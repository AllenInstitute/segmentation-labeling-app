import argschema
import marshmallow as mm
from pathlib import Path
import os
import imageio
import numpy as np
import segmentation_labeling_app.utils.query_utils as query_utils
from segmentation_labeling_app.rois.rois import ROI
from segmentation_labeling_app.transforms.transformations import (
        downsample_h5_video, transform_to_mp4)
from segmentation_labeling_app.transforms.array_utils import (
        content_extents)


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
        required=True,
        default=[128, 128],
        description="[h, w] of bounding box around ROI images/videos")
    quantile = argschema.fields.Float(
        required=True,
        default=0.1,
        description=("quantile threshold for outlining an ROI. "))

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
            data["segmentation_run_id"] = 12345
            label_vars = query_utils.get_labeling_env_vars()
            query_string = (
                "SELECT id FROM segmentation_runs WHERE "
                f"ophys_experiment_id={data['ophys_experiment_id']} AND "
                "ophys_segmentation_commit_hash="
                f"'{data['ophys_segmentation_commit_hash']}'")
            entries = query_utils.query(
                    query_string,
                    label_vars.user,
                    label_vars.host,
                    label_vars.database,
                    label_vars.password,
                    label_vars.port)
            if len(entries) != 1:
                raise TransformPipelineException(
                    f"{query_string} did not return exactly 1 result")
            data['segmentation_run_id'] = entries[0]['id']
        return data


class TransformPipeline(argschema.ArgSchemaParser):
    default_schema = TransformPipelineSchema

    def run(self):
        output_dir = Path(self.args['artifact_basedir']) / \
                     f"segmentation_run_id_{self.args['segmentation_run_id']}"
        os.makedirs(output_dir, exist_ok=True)

        # get all ROI ids from this segmentation run
        dbvars = query_utils.get_labeling_env_vars()
        query_string = ("SELECT id FROM rois WHERE segmentation_run_id="
                        f"{self.args['segmentation_run_id']}")
        entries = query_utils.query(query_string, *dbvars)
        roi_ids = [i['id'] for i in entries]

        # NOTE: here could be a good place to put a pre-filtering step

        rois = [ROI.roi_from_query(roi_id) for roi_id in roi_ids]

        # load, downsample and project the source video
        query_string = ("SELECT * FROM segmentation_runs "
                        f"WHERE id={self.args['segmentation_run_id']}")
        seg_query = query_utils.query(query_string, *dbvars)[0]
        source_path = Path(seg_query['source_video_path'])
        downsampled_video = downsample_h5_video(source_path)
        max_projection = np.max(downsampled_video, axis=0)
        ave_projection = np.mean(downsampled_video, axis=0)

        # create the per-ROI artifacts
        for roi in rois:
            # mask and outline from ROI class
            mask_path = output_dir / f"mask_{roi.roi_id}.png"
            outline_path = output_dir / f"outline_{roi.roi_id}.png"
            sub_video_path = output_dir / f"video_{roi.roi_id}.mp4"
            max_proj_path = output_dir / f"max_{roi.roi_id}.png"
            ave_proj_path = output_dir / f"ave_{roi.roi_id}.png"

            mask = roi.generate_ROI_mask(
                    shape=self.args['cropped_shape'])
            outline = roi.generate_ROI_outline(
                shape=self.args['cropped_shape'],
                quantile=self.args['quantile'])

            imageio.imsave(mask_path, mask, transparency=0)
            imageio.imsave(outline_path, outline, transparency=0)

            # video sub-frame
            inds, pads = content_extents(
                    roi._sparse_coo,
                    shape=self.args['cropped_shape'],
                    target_shape=tuple(downsampled_video.shape[1:]))
            sub_video = np.pad(
                    downsampled_video[:, inds[0]:inds[1], inds[2]:inds[3]],
                    ((0, 0), *pads))
            transform_to_mp4(sub_video, sub_video_path, 10)

            # sub-projections
            sub_max = np.pad(
                    max_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)
            sub_ave = np.pad(
                    ave_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)
            imageio.imsave(max_proj_path, sub_max)
            imageio.imsave(ave_proj_path, sub_ave)

            # manifest entry creation
            manifest = {}
            manifest['experiment-id'] = seg_query['ophys_experiment_id']
            manifest['roi-id'] = roi.roi_id
            manifest['source-ref'] = outline_path
            manifest['roi-mask-source-ref'] = mask_path
            manifest['video-source-ref'] = sub_video_path
            manifest['max-source-ref'] = max_proj_path
            manifest['avg-source-ref'] = ave_proj_path
            # manifest['trace-source-ref'] =


if __name__ == "__main__":  # pragma: no cover
    pipeline = TransformPipeline()
    pipeline.run()
