import argschema
import datetime
import tempfile
import slapp.transfers.utils as utils
import slapp.utils.query_utils as query_utils


class UploadSchema(argschema.ArgSchema):
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

    def run(self, db_conn: query_utils.DbConnection):
        # unique timestamp for this invocation
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # get the specified per-ROI manifests
        query_string = ("SELECT manifest FROM roi_manifests "
                        f"{self.args['sql_filter']}")
        manifests = [r['manifest'] for r in db_conn.query(query_string)]

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
    db_credentials = query_utils.get_labeling_db_credentials()
    db_connection = query_utils.DbConnection(**db_credentials)

    ldu = LabelDataUploader()
    ldu.run(db_connection)
