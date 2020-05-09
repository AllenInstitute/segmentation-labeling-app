import slapp.utils.query_utils as qu
import base64
import os
import argschema
import marshmallow as mm
import numpy as np


insert_statement_template = (
        "INSERT INTO experiment_selection "
        "(base64_query_strings, slapp_commit_hash, "
        "ophys_experiment_ids, sub_selected_ids, random_seed) "
        "VALUES ({}, {}, {}, {}, {})")


class DataSelectorSchema(argschema.ArgSchema):
    query_strings = argschema.fields.List(
        argschema.fields.Str,
        required=True,
        cli_as_single_argument=True,
        description=("results will be the meta set of experiments. "
                     "queries must return 'exp_id' key as alias for "
                     "ophys_experiments.id"))
    sub_selection_counts = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description="how many to sub-select from each query")
    random_seed = argschema.fields.Int(
        required=False,
        default=42,
        description="random number generator seed")

    @mm.post_load
    def check_lengths(self, data, **kwargs):
        nstrings = len(data['query_strings'])
        ncounts = len(data['sub_selection_counts'])
        if nstrings != ncounts:
            raise mm.ValidationError(f"{nstrings} query strings and "
                                     f"{ncounts} sub_selection_counts "
                                     "provided. These should be the same len")
        return data


class DataSelector(argschema.ArgSchemaParser):
    default_schema = DataSelectorSchema

    def run(self, lims_dbconn, label_dbconn):
        experiment_ids = []
        b64_queries = []
        for qstring in self.args['query_strings']:
            experiment_ids.append(
                    [i['exp_id'] for i in lims_dbconn.query(qstring)])
            b64_queries.append(
                base64.b64encode(bytes(qstring, 'utf-8')).decode('utf-8'))

        rng = np.random.default_rng(self.args['random_seed'])

        sub_experiments = []
        for iq, elist in enumerate(experiment_ids):
            rng.shuffle(elist)
            sub_experiments.extend(
                elist[0:self.args['sub_selection_counts'][iq]])

        insert_statement = insert_statement_template.format(
            f"ARRAY{repr(b64_queries)}",
            f"'{os.environ['TRANSFORM_HASH']}'",
            f"ARRAY{repr(experiment_ids)}",
            f"ARRAY{repr(sub_experiments)}",
            f"{self.args['random_seed']}")

        label_dbconn.insert(insert_statement)


if __name__ == "__main__":  # pragma: no cover
    lims_credentials = qu.get_db_credentials(
            env_prefix="LIMS_",
            **qu.lims_defaults)
    lims_connection = qu.DbConnection(**lims_credentials)
    label_credentials = qu.get_db_credentials(
            env_prefix="LABELING_",
            **qu.label_defaults)
    label_connection = qu.DbConnection(**label_credentials)

    query_string_template = (
            "SELECT oe.id AS exp_id "
            "FROM ophys_experiments as oe "
            "JOIN ophys_sessions as os on oe.ophys_session_id=os.id "
            "JOIN projects on projects.id=os.project_id "
            "WHERE projects.code='{}' "
            "AND oe.workflow_state = 'passed' LIMIT 10")

    example_queries = [query_string_template.format(project) for project in
                       ['VisualBehaviorTask1B', 'VisualBehaviorMultiscope']]
    args = {
        "query_strings": example_queries,
        "sub_selection_counts": [2, 4]
        }

    selector = DataSelector(input_data=args)
    selector.run(lims_connection, label_connection)
