import slapp.utils.query_utils as qu
import argschema
import h5py
import pathlib
from slapp.data_selection.utils import find_full_movie


class SegmentationManifestException(Exception):
    pass


class SegmentationManifestSchema(argschema.ArgSchema):
    experiment_selection_id = argschema.fields.Int(
        required=True,
        description="experiment_selection.id to query")
    bin_size = argschema.fields.Int(
        required=False,
        default=115,
        description=("number of frames for Suite2P to do temporal "
                     "binning. `nbinned` parameter for Suite2P will "
                     "be determined by nframes/bin_size on a per-"
                     "experiment basis."))


class ManifestEntrySchema(argschema.ArgSchema):
    experiment_id = argschema.fields.Int()
    nbinned = argschema.fields.Int()
    input_video = argschema.fields.InputFile()


class SegmentationManifestOutputSchema(argschema.ArgSchema):
    manifest = argschema.fields.List(
        argschema.fields.Nested(ManifestEntrySchema))


class SegmentationManifest(argschema.ArgSchemaParser):
    default_schema = SegmentationManifestSchema
    default_output_schema = SegmentationManifestOutputSchema

    def run(self, lims_dbconn, label_dbconn):
        self.logger.name = type(self).__name__

        # from labeling database, find the experiment ids we want
        experiments = label_dbconn.query(
                "SELECT sub_selected_ids FROM experiment_selection "
                f"WHERE id={self.args['experiment_selection_id']}")[0]
        experiments = experiments['sub_selected_ids']
        self.logger.info(f"selected {len(experiments)} experiments from "
                         "table experiment_selection")

        # find out some information about the experiments from LIMS
        lims_query_string = (
                "SELECT id, movie_number_of_frames as nframes, "
                "storage_directory FROM ophys_experiments WHERE "
                f"id in ({repr(experiments)[1:-1]})")
        lims_results = lims_dbconn.query(lims_query_string)
        lims_ids = [i['id'] for i in lims_results]
        if len(lims_ids) != len(experiments):
            raise SegmentationManifestException(
                    f"labeling database specified {len(experiments)} "
                    f"experiments. But LIMS only found {len(lims_ids)} "
                    f"with missing ids {set(experiments) - set(lims_ids)}")

        # create the manifest
        manifest = {'manifest': []}
        for result in lims_results:
            # get the full video path
            video_path = find_full_movie(
                    pathlib.Path(result['storage_directory']))
            with h5py.File(video_path, "r") as h5f:
                nframes_h5 = h5f['data'].shape[0]
            if nframes_h5 != result['nframes']:
                # LIMS is a little weird for not storing the actual movie path
                # check to be sure.
                raise SegmentationManifestException(
                        f"for experiment {result['id']} LIMS has "
                        f"nframes = {result['nframes']} "
                        f"but the found video file {video_path} "
                        f"has nframes = {nframes_h5}")

            manifest['manifest'].append({
                'experiment_id': result['id'],
                'nbinned': int(result['nframes'] / self.args['bin_size']),
                'input_video': str(video_path)
                })

        self.output(manifest, indent=2)
        self.logger.info(f"wrote {self.args['output_json']}")


if __name__ == "__main__":  # pragma: no cover
    lims_credentials = qu.get_db_credentials(
            env_prefix="LIMS_",
            **qu.lims_defaults)
    lims_connection = qu.DbConnection(**lims_credentials)
    label_credentials = qu.get_db_credentials(
            env_prefix="LABELING_",
            **qu.label_defaults)
    label_connection = qu.DbConnection(**label_credentials)

    sm = SegmentationManifest()
    sm.run(lims_connection, label_connection)
