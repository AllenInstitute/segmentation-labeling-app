import pytest
from slapp.data_selection.utils import (
        FindFileException, find_full_movie, candidate_movie_names)


@pytest.mark.parametrize("movie_name", candidate_movie_names)
@pytest.mark.parametrize("top_dir, write_dir",
                         [
                             ('the/same', 'the/same'),
                             ('not/the/same', 'not/the/same/this/one/deeper')
                                 ])
def test_find_full_movie(movie_name, top_dir, write_dir, tmp_path):
    sdir = tmp_path / top_dir
    sdir.mkdir(parents=True)
    wdir = tmp_path / write_dir
    wdir.mkdir(parents=True, exist_ok=True)

    filename = wdir / movie_name
    with open(filename, "w") as fp:
        fp.write("content")

    x = find_full_movie(sdir)
    assert x.name == movie_name


def test_no_full_movie_file(tmp_path):
    sdir = tmp_path / 'nothing/in/here'
    sdir.mkdir(parents=True)

    with pytest.raises(FindFileException):
        find_full_movie(sdir)


@pytest.mark.parametrize("movie_name", candidate_movie_names)
def test_multiple_movie_files(movie_name, tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    wdir1 = root / 'put/one/here'
    wdir1.mkdir(parents=True)
    wdir2 = root / 'put/another/one/here'
    wdir2.mkdir(parents=True)

    for wdir in [wdir1, wdir2]:
        filename = wdir / movie_name
        with open(filename, "w") as fp:
            fp.write("content")

    with pytest.raises(FindFileException):
        find_full_movie(root)
