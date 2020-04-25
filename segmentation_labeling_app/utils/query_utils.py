import os
from collections import namedtuple

import pg8000


class LabelingEnvException(Exception):
    pass


def get_labeling_db_credentials() -> dict:
    """Get labeling DB credentials from environment variables.

    Returns
    -------
    dict
        A dictionary of DB credentials. Contains the following fields:
        'user', 'host', 'database', 'password', 'port'.

    Raises
    ------
    LabelingEnvException
        Raised if LABELING_USER/LABELING_PASSWORD are not set.
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
    db_credentials = {'user': user, 'host': host, 'database': database,
                      'password': password, 'port': port}

    return db_credentials


class DbConnection():

    def __init__(self, user, host, database, password, port):
        self.user = user
        self.host = host
        self.database = database
        self.password = password
        self.port = port

    @staticmethod
    def _connect(user, host, database, password, port):
        conn = pg8000.connect(user=user, host=host, database=database,
                              password=password, port=port)
        return conn, conn.cursor()

    @staticmethod
    def _select(cursor, query):
        cursor.execute(query)
        columns = [d[0].decode("utf-8") for d in cursor.description]
        return [dict(zip(columns, c)) for c in cursor.fetchall()]

    def query(self, query):
        conn, cursor = DbConnection._connect(self.user, self.host,
                                             self.database,
                                             self.password, self.port)

        # Guard against non-ascii characters in query
        query = ''.join([i if ord(i) < 128 else ' ' for i in query])

        try:
            results = DbConnection._select(cursor, query)
        finally:
            cursor.close()
            conn.close()
        return results
