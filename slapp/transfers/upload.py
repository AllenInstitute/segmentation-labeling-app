import argschema
import datetime
import tempfile
import slapp.transfers.utils as utils
import slapp.utils.query_utils as query_utils
import numpy as np
import pathlib


class UploadSchema(argschema.ArgSchema):
    roi_manifests_ids = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description=("specifies the values of roi_manifests.ids "
                     "to include in the upload"))
    s3_bucket_name = argschema.fields.Str(
        required=True,
        description="destination bucket name")
    prefix = argschema.fields.Str(
        required=False,
        default=None,
        allow_none=True,
        description="key prefix for manifest and contents")
    timestamp = argschema.fields.Bool(
        required=False,
        missing=True,
        description=("whether to append a timestamp "
                     "to the key prefix"))


class LabelDataUploader(argschema.ArgSchemaParser):
    default_schema = UploadSchema

    def run(self, db_conn: query_utils.DbConnection):
        self.logger.name = type(self).__name__

        # unique timestamp for this invocation
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # get the specified per-ROI manifests
        nrequested = len(self.args['roi_manifests_ids'])
        self.logger.info(
                f"Requesting {nrequested} roi manifests from postgres")

        idstr = repr(self.args['roi_manifests_ids'])[1:-1]
        query_string = ("SELECT id, manifest FROM roi_manifests "
                        f"WHERE id in ({idstr})")
        results = db_conn.query(query_string)
        manifests = [r['manifest'] for r in results]
        nman = len(manifests)
        if nman != nrequested:
            manifest_ids = [r['id'] for r in results]
            missing_ids = \
                set(self.args['roi_manifests_ids']) - set(manifest_ids)
            self.logger.warning(
                    f"Requested {nrequested}, received {nman}. "
                    f"Missing ids: {missing_ids}")

        # specify the URI
        prefix = self.args['prefix']
        if self.args['timestamp']:
            if prefix is None:
                prefix = self.timestamp
            else:
                prefix += '/' + self.timestamp

        uri = utils.s3_uri(self.args['s3_bucket_name'], prefix)
        self.logger.info(f"bucket destination is {uri}")

        # upload the per-experiment objects
        full_video_paths, uindex = np.unique(
                [m['full-video-source-ref'] for m in manifests],
                return_index=True)
        experiment_ids = [manifests[ui]['experiment-id'] for ui in uindex]
        self.logger.info(f"{full_video_paths.size} full videos to upload")
        s3_full_videos = {}
        for eid, video_path in zip(experiment_ids, full_video_paths):
            object_key = prefix + "/" + f"{eid}_"
            object_key += pathlib.PurePath(video_path).name
            s3_full_video = utils.upload_file(
                    video_path,
                    self.args['s3_bucket_name'],
                    object_key)
            s3_full_videos[video_path] = s3_full_video

        # upload the per-ROI manifests
        s3_manifests = []
        for nm, manifest in enumerate(manifests):
            s3_manifests.append(
                utils.upload_manifest_contents(
                    manifest,
                    self.args['s3_bucket_name'],
                    prefix,
                    skip_keys=['full-video-source-ref']))
            s3_manifests[-1]['full-video-source-ref'] = \
                s3_full_videos[manifest['full-video-source-ref']]
            if ((nm + 1) % 100 == 0) | (nm == nman - 1):
                self.logger.info(
                        f"uploaded source data for {nm + 1} / {nman} ROIs")

        # upload the manifest
        tfile = tempfile.NamedTemporaryFile()
        utils.manifest_file_from_jsons(tfile.name, s3_manifests)
        # NOTE: the docs for SageMaker GroundTruth specify a JSON Lines format
        # but, throws an error with the .jsonl extension
        # setting here to .json extension to resolve the error.
        s3_manifest = utils.upload_file(
                tfile.name,
                self.args['s3_bucket_name'],
                key=prefix + "/manifest.json")
        self.logger.info(f"uploaded {s3_manifest}")
        tfile.close()


if __name__ == "__main__":  # pragma: no cover
    db_credentials = query_utils.get_db_credentials(
            env_prefix="LABELING_",
            **query_utils.label_defaults)
    db_connection = query_utils.DbConnection(**db_credentials)

    ldu = LabelDataUploader()
    ldu.run(db_connection)
