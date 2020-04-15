import argschema
import datetime
import tempfile
import segmentation_labeling_app.transfers.transfer_utils as utils


class UploadSchema(argschema.ArgSchema):
    sqlite_db_path = argschema.fields.InputFile(
        required=True,
        description="sqllite input db file")
    sql_table = argschema.fields.Str(
        required=False,
        missing="manifest_table",
        description="table where the manifests are stored")
    sql_filter = argschema.fields.Str(
        required=False,
        default="",
        missing="",
        description="SQL query filter, starting with 'WHERE'")
    s3_bucket_name = argschema.fields.Str(
        required=True,
        description="destination bucket name")
    contents_prefix = argschema.fields.Str(
        required=False,
        default="",
        missing="",
        description="key prefix for manifest contents")
    manifest_prefix = argschema.fields.Str(
        required=False,
        allow_none=True,
        default=None,
        missing=None,
        description=("key prefix for manifest. Will default "
                     "to contents_prefix"))
    timestamp = argschema.fields.Bool(
        required=False,
        missing=True,
        description=("whether to append a timestamp "
                     "to the key prefix"))


class LabelDataUploader(argschema.ArgSchemaParser):
    default_schema = UploadSchema

    def run(self):
        manifests = utils.get_manifests_from_db(
                self.args['sqlite_db_file'],
                self.args['sqlote_table'],
                self.args['sql_filter'])

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        contents_path = self.args['contents_prefix']
        if self.args['timestamp']:
            contents_path += '/' + timestamp

        s3_manifests = []
        for manifest in manifests:
            s3_manifests.append(
                utils.upload_manifest_contents(
                    manifest,
                    self.args['s3_bucket_name'],
                    contents_path))

        manifest_path = self.args['manifest_prefix']
        if manifest_path is None:
            manifest_path = contents_path

        if self.args['timestamp']:
            manifest_path += '/' + timestamp

        # with tempfile.NamedTemporaryFile() as tfile:
        #     utils.manifest_file_from_jsons(tfile.name, s3_manifests)
        #     utils.upload_manifest(
        #         tfile.name,
        #         self.args['s3_bucket_name'],
        #         key=manifest_path + f"{timestamp}_manifest.jsonl")


if __name__ == "__main__":
    ldu = LabelDataUploader()
    ldu.run()
