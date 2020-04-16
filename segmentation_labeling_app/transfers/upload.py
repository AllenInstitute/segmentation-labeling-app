import argschema
import datetime
import tempfile
import segmentation_labeling_app.transfers.utils as utils


class UploadSchema(argschema.ArgSchema):
    sqlite_db_file = argschema.fields.InputFile(
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
    prefix = argschema.fields.Str(
        required=False,
        default="",
        missing="",
        description="key prefix for manifest and contents")
    timestamp = argschema.fields.Bool(
        required=False,
        missing=True,
        description=("whether to append a timestamp "
                     "to the key prefix"))


class LabelDataUploader(argschema.ArgSchemaParser):
    default_schema = UploadSchema

    def run(self):
        # unique timestamp for this invocation
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # get the specified per-ROI manifests
        manifests = utils.get_manifests_from_db(
                self.args['sqlite_db_file'],
                self.args['sql_table'],
                sql_filter=self.args['sql_filter'])

        # upload the per-ROI manifests
        prefix = self.args['prefix']
        if self.args['timestamp']:
            prefix += '/' + self.timestamp
        s3_manifests = []
        for manifest in manifests:
            s3_manifests.append(
                utils.upload_manifest_contents(
                    manifest,
                    self.args['s3_bucket_name'],
                    prefix))

        # upload the manifest
        tfile = tempfile.NamedTemporaryFile()
        utils.manifest_file_from_jsons(tfile.name, s3_manifests)
        utils.upload_file(
                tfile.name,
                self.args['s3_bucket_name'],
                key=prefix + "/manifest.jsonl")
        tfile.close()


if __name__ == "__main__":  # pragma: no cover
    ldu = LabelDataUploader()
    ldu.run()
