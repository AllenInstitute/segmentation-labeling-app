import pytest
from unittest.mock import call, MagicMock
from slapp.data_selection import select_data as sd
import marshmallow as mm
import os
from functools import partial


@pytest.fixture
def mock_db_conn_fixture():

    def mock_query(query_string):
        return [{'exp_id': i} for i in range(100)]

    def mock_insert(statement):
        return

    mock_db_conn = MagicMock()
    mock_db_conn.query.side_effect = mock_query
    mock_db_conn.insert.side_effect = mock_insert
    return mock_db_conn


@pytest.mark.parametrize(
        "query_strings, counts",
        [
            (
                ["SELECT some stuff", "SELECT some other stuff"],
                [25, 30]
                )])
def test_select_data(mock_db_conn_fixture, query_strings, counts, monkeypatch):
    args = {
        "query_strings": query_strings,
        "sub_selection_counts": counts
        }
    os.environ['TRANSFORM_HASH'] = 'example_hash'
    mock_base64 = MagicMock()
    mock_base64.b64encode = MagicMock()

    mpatcher = partial(monkeypatch.setattr, target=sd)
    mpatcher(name="base64", value=mock_base64)

    try:
        selector = sd.DataSelector(input_data=args, args=[])
        selector.run(mock_db_conn_fixture, mock_db_conn_fixture)

        mock_db_conn_fixture.query.assert_has_calls(
                [call(q) for q in query_strings])

        mock_db_conn_fixture.insert.assert_called_once()

        assert mock_base64.b64encode.call_count == len(query_strings)

    finally:
        os.environ.pop('TRANSFORM_HASH')


@pytest.mark.parametrize(
        "query_strings, counts",
        [
            (
                ["SELECT some stuff", "SELECT some other stuff"],
                [30]
                )])
def test_select_data_validation_error(
        mock_db_conn_fixture, query_strings, counts):
    args = {
        "query_strings": query_strings,
        "sub_selection_counts": counts
        }
    with pytest.raises(mm.ValidationError):
        sd.DataSelector(input_data=args, args=[])
