import argschema
import datetime
import slapp.transfers.utils as utils
import slapp.utils.query_utils as query_utils
import numpy as np
import pathlib
import jsonlines
from multiprocessing.pool import ThreadPool
from functools import partial
import marshmallow as mm


class UploadSchema(argschema.ArgSchema):
    roi_manifests_ids = argschema.fields.List(
        argschema.fields.Int,
        required=False,
        cli_as_single_argument=True,
        missing=None,
        default=None,
        description=("specifies the values of roi_manifests.ids "
                     "to include in the upload"))
    manifest_file = argschema.fields.InputFile(
        required=False,
        missing=None,
        default=None,
        description="Manifest file path in jsonlines format."
    )
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
    parallelization = argschema.fields.Int(
        required=False,
        default=1,
        description="Number of parallel processes to use for uploading.")
    client_config = argschema.fields.Dict(
        required=False,
        missing={'retries': {'mode': 'standard', 'max_attempts': 10}},
        description=("passed as kwargs to botocore.config.Config() for "
                     "client configuration."))
    local_s3_manifest_copy = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("file to be written to S3 as overall manifest, "
                     "saved locally. If path not provided, will default "
                     "to <timestamp>_s3_manifest.jsonl in output_json dir"))

    @mm.pre_load
    def set_local_manifest_path(self, data, **kwargs):
        if data['local_s3_manifest_copy'] is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            data['local_s3_manifest_copy'] = str(
                    pathlib.PurePath(data['output_json']).parent /
                    f"{timestamp}_s3_manifest.jsonl")
        return data


class ResponseSchema(argschema.schemas.DefaultSchema):
    file_name = argschema.fields.InputFile(required=True)
    bucket = argschema.fields.Str(required=True)
    key = argschema.fields.Str(required=True)
    response = argschema.fields.Dict(required=True)


class UploadOutputSchema(argschema.ArgSchema):
    successful_uploads = argschema.fields.List(
        argschema.fields.Nested(ResponseSchema),
        required=True)
    failed_uploads = argschema.fields.List(
        argschema.fields.Nested(ResponseSchema),
        required=True)
    local_s3_manifest_copy = argschema.fields.OutputFile(required=True)


class LabelDataUploader(argschema.ArgSchemaParser):
    default_schema = UploadSchema
    default_output_schema = UploadOutputSchema

    def run(self, db_conn: query_utils.DbConnection):
        self.logger.name = type(self).__name__

        # unique timestamp for this invocation
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # get the specified per-ROI manifests
        if self.args["roi_manifests_ids"]:
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
        elif self.args["manifest_file"]:
            manifests = []
            with jsonlines.open(self.args["manifest_file"], "r") as reader:
                for obj in reader:
                    manifests.append(obj)
            nman = len(manifests)

        else:
            raise ValueError("Need to specify either manifest_file or "
                             "roi_manifests_ids.")
        # specify the URI
        prefix = self.args['prefix']
        if self.args['timestamp']:
            if prefix is None:
                prefix = self.timestamp
            else:
                prefix += '/' + self.timestamp

        uri = utils.s3_uri(self.args['s3_bucket_name'], prefix)
        self.logger.info(f"bucket destination is {uri}")

        # find unique experiments and full videos
        full_video_paths, uindex = np.unique(
                [m['full-video-source-ref'] for m in manifests],
                return_index=True)
        experiment_ids = [manifests[ui]['experiment-id'] for ui in uindex]
        self.logger.info(f"{full_video_paths.size} full videos to upload")

        args = []
        for eid, video_path in zip(experiment_ids, full_video_paths):
            object_key = prefix + "/" + f"{eid}_"
            object_key += pathlib.PurePath(video_path).name
            args.append({
                'file_name': video_path,
                'bucket': self.args['s3_bucket_name'],
                'key': object_key})
        chunked_args = [
                (
                    utils.ConfiguredUploadClient(**self.args['client_config']),
                    i.tolist())
                for i in np.array_split(args, self.args['parallelization'])]

        # track every server response for potential cleanup operations
        upload_responses = []

        # NOTE a reason to use ThreadPool instead of Pool is that
        # moto testing does not work with a process-based pool
        # multiprocessing docs do not detail ThreadPool:
        # https://github.com/python/cpython/blob/eb0d359b4b0e14552998e7af771a088b4fd01745/Lib/multiprocessing/pool.py#L918 # noqa
        with ThreadPool(self.args['parallelization']) as pool:
            results = pool.starmap(utils.upload_files, chunked_args)
        results = [i for r in results for i in r]
        s3_full_videos = {e: utils.s3_uri(r['bucket'], r['key'])
                          for e, r in zip(experiment_ids, results)}
        upload_responses.extend([r for r in results])

        # upload the per-ROI manifests
        s3_manifests = []
        upload_partial = partial(
                utils.upload_manifest_contents,
                utils.ConfiguredUploadClient(**self.args['client_config']),
                skip_keys=['full-video-source-ref'])
        args = []
        for manifest in manifests:
            args.append((manifest, self.args['s3_bucket_name'], prefix))
        with ThreadPool(self.args['parallelization']) as pool:
            results = pool.starmap(upload_partial, args)
        s3_manifests, responses = list(zip(*results))
        for r in responses:
            upload_responses.extend(r)

        for s3_manifest in s3_manifests:
            s3_manifest['full-video-source-ref'] = \
                    s3_full_videos[s3_manifest['experiment-id']]

        # upload the manifest
        utils.manifest_file_from_jsons(
                self.args['local_s3_manifest_copy'],
                s3_manifests)
        self.logger.info("wrote local s3 manifest copy "
                         f"{self.args['local_s3_manifest_copy']}")
        # NOTE: the docs for SageMaker GroundTruth specify a JSON Lines format
        # but, throws an error with the .jsonl extension
        # setting here to .json extension to resolve the error.
        client = utils.ConfiguredUploadClient(**self.args['client_config'])
        result = utils.upload_file(
                client,
                self.args['local_s3_manifest_copy'],
                self.args['s3_bucket_name'],
                key=prefix + "/manifest.json")
        self.logger.info(
                f"uploaded {utils.s3_uri(result['bucket'], result['key'])}")
        upload_responses.append(result)

        # cleanup attempt
        success, failed = utils.sort_upload_results(upload_responses)
        if len(failed) != 0:
            self.logger.warning(f"attempting to clean up {len(failed)} "
                                "failed uploads")
            cleanup_args = []
            for r in failed:
                cleanup_args.append(dict(r))
                cleanup_args[-1].pop('response')
            result = utils.upload_files(client, cleanup_args)
            upload_responses = success + result
            success, failed = utils.sort_upload_results(upload_responses)

        self.logger.info(f"{len(success)} uploads succeeded")
        if len(failed) != 0:
            self.logger.warning(f"{len(failed)} uploads failed")

        self.output(
                {
                    'successful_uploads': success,
                    'failed_uploads': success,
                    'local_s3_manifest_copy':
                        self.args['local_s3_manifest_copy']
                        },
                indent=2)


if __name__ == "__main__":  # pragma: no cover
    db_credentials = query_utils.get_db_credentials(
            env_prefix="LABELING_",
            **query_utils.label_defaults)
    db_connection = query_utils.DbConnection(**db_credentials)

    ldu = LabelDataUploader()
    ldu.run(db_connection)
