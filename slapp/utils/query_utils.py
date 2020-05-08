import os
import pg8000


class CredentialsException(Exception):
    pass


label_defaults = {
        "host": "aibsdc-dev-bd1",
        "database": "ophys_segmentation_labeling",
        "port": 5432}


lims_defaults = {
        "host": "limsdb2",
        "database": "lims2",
        "port": 5432}


def get_db_credentials(
        env_prefix="LABELING_", host=None, database=None, port=None) -> dict:
    """Get labeling DB credentials from environment variables.
    keys are ['user', 'password', 'host', 'port', 'database']

    Parameters
    ----------
    env_prefix : str
        expected environment variable prefix for credential keys.
        expected ENV variables are <prefix><KEY> where <KEY> is key.upper()
    host : str
        default host value, if not found in os.environ
    database : str
        default database value, if not found in os.environ
    port : int
        default port value, if not found in os.environ

    Returns
    -------
    dict
        A dictionary of DB credentials. Contains the following fields:
        'user', 'host', 'database', 'password', 'port'.

    Raises
    ------
    CredentialsException
        Raised if user and password are not set as ENV variables
    """

    db_credentials = {}
    for key in ['user', 'password']:
        env_key = env_prefix + key.upper()
        try:
            db_credentials[key] = os.environ[env_key]
        except KeyError:
            raise CredentialsException(f"ENV variable {env_key} must be set")

    defaults = {'host': host, 'port': port, 'database': database}

    for key in ['host', 'port', 'database']:
        env_key = env_prefix + key.upper()
        db_credentials[key] = os.environ.get(env_key, defaults[key])
        if db_credentials[key] is None:
            raise CredentialsException(
                    "no ENV variable found and no default provided "
                    f"for {env_key}")

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

    def insert(self, statement):
        self.bulk_insert(self, [statement])

    def bulk_insert(self, statements):
        """insert multiple statements with a single commit

        Parameters
        ----------
        statements: list
            each element of statements should be a valid INSERT statement
        """
        conn, cursor = DbConnection._connect(self.user, self.host,
                                             self.database,
                                             self.password, self.port)
        try:
            for statement in statements:
                cursor.execute(statement)
        finally:
            conn.commit()
            cursor.close()
            conn.close()

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
