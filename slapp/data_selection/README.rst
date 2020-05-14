==============
Data Selection
==============
These modules are used to select and segment experiments. In context of the rest of this repo, a complete dataflow to a SagemakeGroundTruth labeling job would be:

1. select experiments
2. create segmentation manifest
3. segment experiments into ROIs from segmentation manifest
4. transform ROIs into data sources and manifests
5. upload data sources and manifest to S3
6. launch labeling job

This README is concerned with 1, 2 and 3.

Select experiments
------------------

::

  python -m slapp.data_selection.select_data --input_json example_selection_input.json

where example_selection_input.json has this structure:

::

  {
          "query_strings":[
                  "SELECT oe.id as exp_id, idep.depth, genotypes.name FROM ophys_experiments as oe JOIN ophys_sessions as os on oe.ophys_session_id=os.id JOIN projects on projects.id=os.project_id JOIN imaging_depths as idep on idep.id = oe.imaging_depth_id JOIN specimens on os.specimen_id=specimens.id JOIN donors on specimens.donor_id=donors.id JOIN donors_genotypes dg ON dg.donor_id=donors.id JOIN genotypes ON genotypes.id=dg.genotype_id JOIN genotype_types as gt on gt.id=genotypes.genotype_type_id WHERE gt.name='driver' AND oe.workflow_state = 'passed' AND projects.code='VisualBehaviorMultiscope' AND genotypes.name='Sst-IRES-Cre' AND idep.depth BETWEEN 100 and 300",
                  "SELECT oe.id as exp_id, idep.depth, genotypes.name FROM ophys_experiments as oe JOIN ophys_sessions as os on oe.ophys_session_id=os.id JOIN projects on projects.id=os.project_id JOIN imaging_depths as idep on idep.id = oe.imaging_depth_id JOIN specimens on os.specimen_id=specimens.id JOIN donors on specimens.donor_id=donors.id JOIN donors_genotypes dg ON dg.donor_id=donors.id JOIN genotypes ON genotypes.id=dg.genotype_id JOIN genotype_types as gt on gt.id=genotypes.genotype_type_id WHERE gt.name='driver' AND oe.workflow_state = 'passed' AND projects.code='VisualBehaviorTask1B' AND genotypes.name='Sst-IRES-Cre' AND idep.depth BETWEEN 100 and 300",
                  "SELECT oe.id as exp_id, idep.depth, genotypes.name FROM ophys_experiments as oe JOIN ophys_sessions as os on oe.ophys_session_id=os.id JOIN projects on projects.id=os.project_id JOIN imaging_depths as idep on idep.id = oe.imaging_depth_id JOIN specimens on os.specimen_id=specimens.id JOIN donors on specimens.donor_id=donors.id JOIN donors_genotypes dg ON dg.donor_id=donors.id JOIN genotypes ON genotypes.id=dg.genotype_id JOIN genotype_types as gt on gt.id=genotypes.genotype_type_id WHERE gt.name='driver' AND oe.workflow_state = 'passed' AND projects.code='VisualBehaviorMultiscope' AND genotypes.name='Slc17a7-IRES2-Cre' AND idep.depth BETWEEN 100 and 200",
                  "SELECT oe.id as exp_id, idep.depth, genotypes.name FROM ophys_experiments as oe JOIN ophys_sessions as os on oe.ophys_session_id=os.id JOIN projects on projects.id=os.project_id JOIN imaging_depths as idep on idep.id = oe.imaging_depth_id JOIN specimens on os.specimen_id=specimens.id JOIN donors on specimens.donor_id=donors.id JOIN donors_genotypes dg ON dg.donor_id=donors.id JOIN genotypes ON genotypes.id=dg.genotype_id JOIN genotype_types as gt on gt.id=genotypes.genotype_type_id WHERE gt.name='driver' AND oe.workflow_state = 'passed' AND projects.code='VisualBehaviorTask1B' AND genotypes.name='Slc17a7-IRES2-Cre' AND idep.depth BETWEEN 100 and 200"],
          "sub_selection_counts": [75, 20, 25, 10],
          "random_seed": 42,
          "comment_string": "example query for README in repo",
          "log_level": "INFO"
  }
   
running this command will make an entry into a postgres table called ``experiment_selection``. The results can be retrieved by:

::

  import slapp.utils.query_utils as qu                                       
  dbconn = qu.DbConnection(**qu.get_db_credentials(**qu.label_defaults))     
  results = dbconn.query("SELECT * FROM experiment_selection WHERE id=5")[0]

Note that the query strings have been converted to base64 for ease of writing them to text fields in postgres. They can be decoded by:

::

  In [4]: import base64
  In [5]: print(results['base64_query_strings'][0])

  U0VMRUNUIG9lLmlkIGFzIGV4cF9pZCwgaWRlcC5kZXB0aCwgZ2Vub3R5cGVzLm5hbWUgRlJPTSBvcGh5c19leHBlcmltZW50cyBhcyBvZSBKT0lOIG9waHlzX3Nlc3Npb25zIGFzIG9zIG9uIG9lLm9waHlzX3Nlc3Npb25faWQ9b3MuaWQgSk9JTiBwcm9qZWN0cyBvbiBwcm9qZWN0cy5pZD1vcy5wcm9qZWN0X2lkIEpPSU4gaW1hZ2luZ19kZXB0aHMgYXMgaWRlcCBvbiBpZGVwLmlkID0gb2UuaW1hZ2luZ19kZXB0aF9pZCBKT0lOIHNwZWNpbWVucyBvbiBvcy5zcGVjaW1lbl9pZD1zcGVjaW1lbnMuaWQgSk9JTiBkb25vcnMgb24gc3BlY2ltZW5zLmRvbm9yX2lkPWRvbm9ycy5pZCBKT0lOIGRvbm9yc19nZW5vdHlwZXMgZGcgT04gZGcuZG9ub3JfaWQ9ZG9ub3JzLmlkIEpPSU4gZ2Vub3R5cGVzIE9OIGdlbm90eXBlcy5pZD1kZy5nZW5vdHlwZV9pZCBKT0lOIGdlbm90eXBlX3R5cGVzIGFzIGd0IG9uIGd0LmlkPWdlbm90eXBlcy5nZW5vdHlwZV90eXBlX2lkIFdIRVJFIGd0Lm5hbWU9J2RyaXZlcicgQU5EIG9lLndvcmtmbG93X3N0YXRlID0gJ3Bhc3NlZCcgQU5EIHByb2plY3RzLmNvZGU9J1Zpc3VhbEJlaGF2aW9yTXVsdGlzY29wZScgQU5EIGdlbm90eXBlcy5uYW1lPSdTc3QtSVJFUy1DcmUnIEFORCBpZGVwLmRlcHRoIEJFVFdFRU4gMTAwIGFuZCAzMDA=

  In [6]: print(base64.b64decode(results['base64_query_strings'][0]))

  b"SELECT oe.id as exp_id, idep.depth, genotypes.name FROM ophys_experiments as oe JOIN ophys_sessions as os on oe.ophys_session_id=os.id JOIN projects on projects.id=os.project_id JOIN imaging_depths as idep on idep.id = oe.imaging_depth_id JOIN specimens on os.specimen_id=specimens.id JOIN donors on specimens.donor_id=donors.id JOIN donors_genotypes dg ON dg.donor_id=donors.id JOIN genotypes ON genotypes.id=dg.genotype_id JOIN genotype_types as gt on gt.id=genotypes.genotype_type_id WHERE gt.name='driver' AND oe.workflow_state = 'passed' AND projects.code='VisualBehaviorMultiscope' AND genotypes.name='Sst-IRES-Cre' AND idep.depth BETWEEN 100 and 300"
  
The field ``sub_selected_ids`` is intended to be a list of LIMS ``ophys_experiments.id`` that will be sent through segmentation.

::

  In [7]: print(repr(results['sub_selected_ids']))                                   
  [851093291, 977247468, 866518324, 853363749, 865798237, 951980481, 982903847, 986518870, 977978331, 986518863, 982903843, 957759564, 978296102, 988707128, 960995086, 867410514, 988707124, 867410520, 978296114, 953659743, 977247476, 867410518, 960995077, 982903853, 977978329, 853363739, 987317107, 856123119, 856123117, 953659752, 982344777, 853988446, 871196369, 976300303, 866518318, 871196375, 957759568, 866518326, 960995084, 868870094, 871196377, 850517344, 986518852, 866518314, 957759562, 956941841, 959388794, 989212489, 959388790, 976300297, 987317101, 864967106, 866518316, 977978327, 951980479, 854759894, 866518293, 871196365, 850517352, 875786885, 853988430, 864430668, 956941848, 868870092, 857698006, 868870085, 959388798, 977247474, 865798247, 958527479, 864967102, 988707126, 977247472, 873963899, 864430666, 994053903, 1002314807, 989191384, 1003771249, 1010812025, 986402309, 979668410, 960960480, 993344860, 957652800, 1001535125, 978827848, 993862620, 994061182, 1003456269, 1012112426, 995439942, 994790561, 984551228, 993593393, 919419001, 974433390, 889806727, 905955238, 889806719, 908381674, 932381896, 908381700, 914107592, 990681006, 886585126, 974433399, 909184300, 914580660, 989610989, 915243090, 972233193, 916220450, 905955228, 929603796, 887386949, 904363934, 886003523, 989610985, 929603805, 901559828, 935440149, 971761068, 934456506, 1011751579, 906877227, 932333410, 972683314, 994278291, 915141818]

Create Segmentation Manifest
----------------------------
We have a relatively stable, but still development pipeline for Suite2P segmentation in the repo ophys_segmentation_. This repo is not part of any automated workflow. To aid in running these segmentations manually, we have created an example job array script for launching many jobs on cluster at once, and a manifest creation method that can feed that job array script. We create a segmentation manifest from the same ``experiment_selection`` table we wrote to above.

.. _ophys_segmentation: https://github.com/AllenInstitute/ophys_segmentation

::

  $ python -m slapp.data_selection.segmentation_manifest --experiment_selection_id 5 --output_json ./example_output.json --log_level INFO
  INFO:SegmentationManifest:selected 130 experiments from table experiment_selection
  INFO:SegmentationManifest:wrote ./example_output.json


where the output looks like

::

  {
    "manifest": [
      {
        "log_level": "ERROR",
        "experiment_id": 850517344,
        "nbinned": 420,
        "input_video": "/allen/programs/braintv/production/neuralcoding/prod0/specimen_813702151/ophys_session_849304162/ophys_experiment_850517344/processed/motion_corrected_video.h5"
      },
      {
        "log_level": "ERROR",
        "experiment_id": 850517352,
        "nbinned": 420,
        "input_video": "/allen/programs/braintv/production/neuralcoding/prod0/specimen_813702151/ophys_session_849304162/ophys_experiment_850517352/processed/motion_corrected_video.h5"
      },
      ...

The ``log_level`` keys are a result of using arschema to validate this output with a schema.

Segment experiments into ROIs from segmentation manifest
--------------------------------------------------------
We have included an example_ script to document how to use the segmentation manifest to run Suite2P segmentation.

This script is an example only. The user will need to modify it. Some key points for using this script:

- ``#PBS -o`` should be redirected to somewhere where the user has write permissions.
- ``#PBS -t`` controls the array job. If your manifest contains 100 experiments ``#PBS -t 0-99`` will run them all. If you want to dribble them out 10 at a time ``#PBS -t 0-99%10``. If you want to rerun one or a few specific jobs ``#PBS -t 12,34,84`` comma-separated specification must be in order.
- Know your conda env. There is no guarantee that the one in this example exists or is up to date. The user needs a conda environment that has ophys_segmentation_ and Suite2P installed.
- The line ``source .. sourceme.sh`` is how I set the ENV variables ``MLFLOW_DB_USER``, ``MLFLOW_DB_PASSWORD``, ``LABELING_USER``, ``LABELING_PASSWORD``, ``LIMS_USER`` and ``LIMS_PASSWORD``. You need these. Set them somehow.
- Change the line ``manifest=`` to the path of the manifest you created in the step above.

The jobs from running this script will segment videos with Suite2P and write entries to the postgres tables ``segmentation_runs`` and ``rois``. They are then ready for running the transform pipeline.

.. _example: scripts/suite2p_array_job_example.pbs
