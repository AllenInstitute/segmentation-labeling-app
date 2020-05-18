import pathlib

candidate_movie_names = ['concat_31Hz_0.h5', 'motion_corrected_video.h5']


class FindFileException(Exception):
    pass


def find_full_movie(search_dir: pathlib.Path) -> pathlib.Path:
    """find a full movie given a top directory

    Parameters
    ----------
    search_dir: pathlib.Path
        upper most directory for conducting the search

    Returns
    -------
    result: pathlib.Path
        resolved path to the first matched candidate file name

    Raises
    ------
    FindFileException
        if more than one match is found for one candidate or
        if no matches are found

    """
    for candidate in candidate_movie_names:
        result = list(search_dir.rglob(candidate))
        if len(result) > 1:
            raise FindFileException(
                    "multiple files matching {} in {}: {}".format(
                        candidate, search_dir, result))
        elif len(result) == 1:
            result = result[0].resolve()
            break

    if result == []:
        raise FindFileException(f"could not find a movie in {search_dir}")

    return result
