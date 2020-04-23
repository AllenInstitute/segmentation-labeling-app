import os
from collections import namedtuple

import pg8000


class LabelingEnvException(Exception):
    pass


def get_labeling_env_vars():
    """ returns a namedtuple with fields necessary for connection to
    a postgres table. Fields are populated from environment variables
    and non-credential fields have defaults.
    The order of the tuple contents are such that
    query(query_string, *label_vars) is correct
    """
    try:
        user = os.environ["LABELING_USER"]
        password = os.environ["LABELING_PASSWORD"]
    except KeyError:
        raise LabelingEnvException(
                "both env variables LABELING_USER and "
                "LABELING_PASSWORD must be set")
    host = os.environ.get('LABELING_HOST', "aibsdc-dev-db1")
    database = os.environ.get(
            'LABELING_DATABASE', "ophys_segmentation_labeling")
    port = os.environ.get('LABELING_PORT', 5432)
    LabelVars = namedtuple(
            'LabelVars',
            ['user', 'host', 'database', 'password', 'port'])
    label_vars = LabelVars(user, host, database, password, port)
    return label_vars


def _connect(user, host, database, password, port):
    conn = pg8000.connect(user=user, host=host, database=database,
                          password=password, port=port)
    return conn, conn.cursor()


def _select(cursor, query):
    cursor.execute(query)
    columns = [d[0].decode("utf-8") for d in cursor.description]
    return [dict(zip(columns, c)) for c in cursor.fetchall()]


def query(query, user, host, database, password, port):
    conn, cursor = _connect(user, host, database, password, port)

    # Guard against non-ascii characters in query
    query = ''.join([i if ord(i) < 128 else ' ' for i in query])

    try:
        results = _select(cursor, query)
    finally:
        cursor.close()
        conn.close()
    return results
