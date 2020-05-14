import pytest
from unittest.mock import MagicMock
from slapp.data_selection import segmentation_manifest as sm
from functools import partial
import numpy as np


@pytest.fixture
def mock_label_db_conn_fixture():

    def mock_query(query_string):
        return [{'sub_selected_ids': [1, 2, 3, 4]}]

    mock_db_conn = MagicMock()
    mock_db_conn.query.side_effect = mock_query
    return mock_db_conn


@pytest.fixture
def mock_lims_db_conn_fixture():
    def mock_query(query_string):
        return_dict = []
        for i in range(4):
            return_dict.append(
                {
                    'id': i,
                    'nframes': 10,
                    'storage_directory': 'fake'
                    }
                )
        return return_dict

    mock_db_conn = MagicMock()
    mock_db_conn.query.side_effect = mock_query
    return mock_db_conn


def test_segmentation_manifest(
        mock_lims_db_conn_fixture, mock_label_db_conn_fixture, monkeypatch,
        tmp_path):

    mock_find_file = MagicMock()
    mock_h5 = MagicMock()
    mock_h5.File.return_value.__enter__.return_value = {'data': np.arange(10)}
    mock_output = MagicMock()
    mpatcher0 = partial(monkeypatch.setattr, target=sm.SegmentationManifest)
    mpatcher0(name="output", value=mock_output)
    mpatcher = partial(monkeypatch.setattr, target=sm)
    mpatcher(name="find_full_movie", value=mock_find_file)
    mpatcher(name="h5py", value=mock_h5)

    outjson = tmp_path / "output.json"
    args = {
            'experiment_selection_id': 12,
            'output_json': str(outjson)
            }

    sman = sm.SegmentationManifest(input_data=args, args=[])
    sman.run(mock_lims_db_conn_fixture, mock_label_db_conn_fixture)

    mock_label_db_conn_fixture.query.assert_called_once()
    mock_lims_db_conn_fixture.query.assert_called_once()
    mock_output.assert_called_once()
    assert mock_h5.File.call_count == 4
