import pytest
from unittest.mock import ANY, call, MagicMock
from pathlib import Path
from functools import partial
import os
import numpy as np

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
        return mock_ROI

    mock_ROI = MagicMock()
    mock_ROI.roi_from_query.side_effect = mock_roi_from_query
    return mock_ROI


def create_expected_manifest(experiment_id, roi_id, segmentation_run_id,
                             save_path):
    out_dirname = f"segmentation_run_id_{segmentation_run_id}"
    expected_manifest = {
        "experiment-id": experiment_id,
        "roi-id": roi_id,
        "source-ref": f"{save_path}/{out_dirname}/outline_{roi_id}.png",
        "roi-mask-source-ref": f"{save_path}/{out_dirname}/mask_{roi_id}.png",
        "video-source-ref": f"{save_path}/{out_dirname}/video_{roi_id}.mp4",
        "max-source-ref": f"{save_path}/{out_dirname}/max_{roi_id}.png",
        "avg-source-ref": f"{save_path}/{out_dirname}/avg_{roi_id}.png",
        "trace-source-ref": f"{save_path}/{out_dirname}/trace_{roi_id}.json"
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

    expected_manifests = []
    for meta in expected_manifest_metadata:
        manifest = create_expected_manifest(**meta, save_path=str(tmp_path))
        expected_manifests.append(manifest)

    def normalize_side_effect(value, dummya, dummyb):
        return value

    mock_downsample_h5_video = MagicMock(return_value=np.zeros((3, 5, 5)))
    mock_normalize = MagicMock(side_effect=normalize_side_effect)
    mock_imageio = MagicMock()
    mock_content_extents = MagicMock(return_value=([0, 0, 2, 2], [1, 1, 1, 1]))
    mock_transform_to_mp4 = MagicMock()
    mock_np = MagicMock()
    mock_np.quantile = MagicMock(return_value=(0, 1))

    mpatcher = partial(monkeypatch.setattr, target=transform_pipeline)
    mpatcher(name="ROI", value=mock_roi)
    mpatcher(name="downsample_h5_video", value=mock_downsample_h5_video)
    mpatcher(name="imageio", value=mock_imageio)
    mpatcher(name="content_extents", value=mock_content_extents)
    mpatcher(name="normalize_array", value=mock_normalize)
    mpatcher(name="transform_to_mp4", value=mock_transform_to_mp4)
    mpatcher(name="np", value=mock_np)

    os.environ['TRANSFORM_HASH'] = 'dummy_hash'
    pipeline = transform_pipeline.TransformPipeline(input_data=input_data,
                                                    args=[])
    pipeline.run(mock_db_conn_fixture)

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
        outline_save = call(Path(manifest['source-ref']), ANY,
                            transparency=0)
        max_save = call(Path(manifest['max-source-ref']), ANY)
        avg_save = call(Path(manifest['avg-source-ref']), ANY)

        calls = [mask_save, outline_save, max_save, avg_save]
        mock_imageio.imsave.assert_has_calls(calls, any_order=False)

    # Assert that created manifests are correct
    outdir_name = f"segmentation_run_id_{input_data['segmentation_run_id']}"
    expected_outdir = Path(input_data["artifact_basedir"]) / outdir_name
    expected_calls = [call(m, expected_outdir) for m in expected_manifests]
