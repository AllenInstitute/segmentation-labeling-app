import pytest
from unittest.mock import ANY, call, MagicMock
from pathlib import Path
from functools import partial
import os
import numpy as np
import json
import h5py

from slapp.transforms import transform_pipeline


@pytest.fixture
def mock_db_conn_fixture(request):
    query_return_values = request.param

    def mock_query(query_string):
        return query_return_values[query_string]

    def mock_insert(statements):
        return

    mock_db_conn = MagicMock()
    mock_db_conn.query.side_effect = mock_query
    mock_db_conn.bulk_insert.side_effect = mock_insert
    return mock_db_conn


@pytest.fixture
def mock_roi():

    def mock_roi_from_query(roi_id, db_conn) -> MagicMock:
        mock_ROI = MagicMock()
        mock_ROI.roi_id = roi_id

        roi = db_conn.query(f"SELECT * FROM rois WHERE id={roi_id}")[0]
        segmentation_run = db_conn.query(
            ("SELECT * FROM segmentation_runs WHERE "
             f"id={roi['segmentation_run_id']}"))[0]

        mock_ROI.experiment_id = segmentation_run['ophys_experiment_id']
        return mock_ROI

    mock_ROI = MagicMock()
    mock_ROI.roi_from_query.side_effect = mock_roi_from_query
    return mock_ROI


@pytest.fixture
def production_roi_manifest(tmp_path, request):
    binarized = tmp_path / "mock_binary_rois.json"
    with open(binarized, "w") as fp:
        json.dump(request.param.get("binarized_content"), fp)
    movie = tmp_path / "mock_movie.h5"
    with h5py.File(movie, "w") as f:
        f.create_dataset(
                "data",
                data=np.zeros(request.param.get("movie_shape")))
    trace = tmp_path / "traces.h5"
    if 'trace_content' in request.param:
        with h5py.File(trace, "w") as f:
            f.create_dataset(
                    'data',
                    data=request.param.get("trace_content")['trace'])
            f.create_dataset(
                    'roi_names',
                    data=np.array(
                        request.param.get(
                            "trace_content")['names']).astype(np.string_),
                    dtype=h5py.special_dtype(vlen=str))
    manifest = {
        'experiment_id': request.param.get("experiment_id"),
        'binarized_rois_path': str(binarized),
        'movie_path': str(movie),
        'traces_h5_path': str(trace),
        'local_to_global_roi_id_map': request.param.get("id_map")
        }
    yield manifest, request.param


@pytest.mark.parametrize(
    "production_roi_manifest",
    [({
        'binarized_content': {'some': 'thing'},
        'movie_shape': (100, 10, 10),
        'experiment_id': 1234,
        'id_map': {1: 20002}
        })],
    indirect=["production_roi_manifest"])
def test_ProdSegSchema(production_roi_manifest):
    """tests that the schema retrieves movie shape and
    the content of the binarized json file
    """
    manifest, params = production_roi_manifest
    pschema = transform_pipeline.ProdSegmentationRunManifestSchema()
    result = pschema.load(manifest)
    print(json.dumps(result, indent=2))
    assert result['experiment_id'] == params['experiment_id']
    assert result['movie_frame_shape'] == params['movie_shape'][1:]
    assert result['binarized_rois'] == params['binarized_content']


@pytest.mark.parametrize(
        "production_roi_manifest",
        [(
            {
                'binarized_content': [
                    {
                        'id': 12,
                        'mask_matrix': [[0, 0, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 0, 0]],
                        'x': 100,
                        'y': 130},
                    {
                        'id': 13,
                        'mask_matrix': [[0, 0, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 0, 0]],
                        'x': 70,
                        'y': 30},
                    {
                        # this one will not make it through
                        # to the returned rois list
                        # because the id is not in the id_map
                        'id': 14,
                        'mask_matrix': [[0, 0, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 0, 0]],
                        'x': 20,
                        'y': 40}],
                'trace_content': {
                    'trace': [[10.2, 11.3, 12.4, 13.3, 14.2, 11.1],
                              [1.0, 2.0, 1.3, 1.4, 3.4, 2.3]],
                    'names': ['12', '13']},
                'movie_shape': (100, 200, 200),
                'experiment_id': 1234,
                'id_map': {12: 200023, 13: 200024}}
            )],
        indirect=['production_roi_manifest'])
def test_xform_from_prod_manifest(tmp_path, production_roi_manifest):
    manifest, params = production_roi_manifest
    man_path = tmp_path / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f)

    rois, movie_path = transform_pipeline.xform_from_prod_manifest(man_path)
    # one has been left out, because it is not in the id map
    assert len(rois) == 2
    # check that the ids have been translated
    ids = set([i.roi_id for i in rois])
    assert ids == set(list(params['id_map'].values()))

    # check that the traces went to the correct ROI
    reverse_map = {v: k for k, v in params['id_map'].items()}
    for roi in rois:
        local_id = reverse_map[roi.roi_id]
        ind = params['trace_content']['names'].index(str(local_id))
        np.testing.assert_array_equal(
                roi.trace, params['trace_content']['trace'][ind])


def create_expected_manifest(experiment_id, roi_id, segmentation_run_id,
                             save_path, dtime):
    out_dirname = f"seg_run_id_{segmentation_run_id}/{dtime}"
    expected_manifest = {
        "experiment-id": experiment_id,
        "roi-id": roi_id,
        "source-ref": f"{save_path}/{out_dirname}/outline_{roi_id}.png",
        "roi-mask-source-ref": f"{save_path}/{out_dirname}/mask_{roi_id}.png",
        "full-video-source-ref": f"{save_path}/{out_dirname}/full_video.webm",
        "video-source-ref": f"{save_path}/{out_dirname}/video_{roi_id}.webm",
        "max-source-ref": f"{save_path}/{out_dirname}/max_{roi_id}.png",
        "avg-source-ref": f"{save_path}/{out_dirname}/avg_{roi_id}.png",
        "trace-source-ref": f"{save_path}/{out_dirname}/trace_{roi_id}.json",
        "full-outline-source-ref": f"{save_path}/{out_dirname}/"
                                   f"full_outline_{roi_id}.png"
    }
    return expected_manifest


@pytest.mark.parametrize(("input_data, mock_db_conn_fixture, "
                          "source_video, "
                          "expected_manifest_metadata"), [
    ({"segmentation_run_id": 42,
      "ophys_experiment_id": 123,
      "ophys_segmentation_commit_hash": "Marvin",
      "artifact_basedir": "replace_me_with_a_tmp_path",
      "cropped_shape": [20, 20],
      "quantile": 0.2,
      "input_fps": 31,
      "output_fps": 4,
      "downsampling_strategy": 'average',
      "random_seed": 0},

     {"SELECT id FROM rois WHERE segmentation_run_id=42": [
         {"id": 0}, {"id": 777}],
      "SELECT * FROM rois WHERE id=0": [
          {"segmentation_run_id": 42}],
      "SELECT * FROM rois WHERE id=777": [
          {"segmentation_run_id": 42}],
      "SELECT * FROM segmentation_runs WHERE id=42": [
         {"source_video_path": "/mock/path", "ophys_experiment_id": 123}
      ]},

     "/mock/path",

     [{"experiment_id": 123, "roi_id": 0, "segmentation_run_id": 42},
      {"experiment_id": 123, "roi_id": 777, "segmentation_run_id": 42}]),

], indirect=["mock_db_conn_fixture"])
def test_transform_pipeline(tmp_path, monkeypatch, mock_db_conn_fixture,
                            mock_roi, input_data, source_video,
                            expected_manifest_metadata):

    input_data["artifact_basedir"] = str(tmp_path)

    def normalize_side_effect(value, dummya, dummyb):
        return value

    mock_downsample_h5_video = MagicMock(return_value=np.zeros((3, 5, 5)))
    mock_normalize = MagicMock(side_effect=normalize_side_effect)
    mock_imageio = MagicMock()
    mock_content_extents = MagicMock(return_value=([0, 0, 2, 2], [1, 1, 1, 1]))
    mock_transform_to_webm = MagicMock()
    mock_add_scale = MagicMock()
    mock_np = MagicMock()
    mock_np.quantile = MagicMock(return_value=(0, 1))

    mpatcher = partial(monkeypatch.setattr, target=transform_pipeline)
    mpatcher(name="ROI", value=mock_roi)
    mpatcher(name="downsample_h5_video", value=mock_downsample_h5_video)
    mpatcher(name="imageio", value=mock_imageio)
    mpatcher(name="content_extents", value=mock_content_extents)
    mpatcher(name="normalize_array", value=mock_normalize)
    mpatcher(name="add_scale", value=mock_add_scale)
    mpatcher(name="transform_to_webm", value=mock_transform_to_webm)
    mpatcher(name="np", value=mock_np)

    os.environ['TRANSFORM_HASH'] = 'dummy_hash'
    pipeline = transform_pipeline.TransformPipeline(input_data=input_data,
                                                    args=[])
    pipeline.run(mock_db_conn_fixture)

    expected_manifests = []
    for meta in expected_manifest_metadata:
        manifest = create_expected_manifest(
                **meta, save_path=str(tmp_path), dtime=pipeline.timestamp)
        expected_manifests.append(manifest)

    # Assert downsample called with correct video path
    mock_downsample_h5_video.assert_called_once_with(
            Path(source_video),
            input_data['input_fps'],
            input_data['output_fps'],
            input_data['downsampling_strategy'],
            input_data['random_seed'])

    # Assert imsaves are called with correct save paths (and in correct order)
    for manifest in expected_manifests:
        mask_save = call(Path(manifest['roi-mask-source-ref']), ANY,
                         transparency=0)
        max_save = call(Path(manifest['max-source-ref']), ANY)
        avg_save = call(Path(manifest['avg-source-ref']), ANY)
        outline_save = call(Path(manifest['source-ref']),
                            ANY, transparency=255)
        full_outline_save = call(Path(manifest['full-outline-source-ref']),
                                 ANY, transparency=255)

        calls = [mask_save, outline_save, full_outline_save,
                 max_save, avg_save]
        mock_imageio.imsave.assert_has_calls(calls, any_order=False)

    assert mock_add_scale.call_count == 4

    # Assert that created manifests are correct
    expected_insert_statements = [
            transform_pipeline.insert_str_template.format(
                json.dumps(manifest),
                os.environ['TRANSFORM_HASH'],
                manifest['roi-id'])
            for manifest in expected_manifests]

    mock_db_conn_fixture.bulk_insert.assert_has_calls(
            [call(expected_insert_statements)])
