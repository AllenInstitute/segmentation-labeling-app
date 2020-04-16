import os
from typing import Tuple
from collections import namedtuple

import pg8000
import sqlite3
from sqlite3 import Error


class LabelingEnvException(Exception):
    pass


def get_labeling_env_vars():
    """ returns a namedtuple with fields necessary for connection to
    a postgres table. Fields are populated from environment variables
    and non-credential fields have defaults.
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
            ['user', 'password', 'host', 'database', 'port'])
    label_vars = LabelVars(user, password, host, database, port)
    return label_vars


def create_connection_sqlite(db_file):
    """
    Creates a database connection to the SQLite database
    specified by db_file
    Args:
        db_file: database file

    Returns:

    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as err:
        print(err)
    return conn


def create_table_sqlite(conn: sqlite3.Connection, create_table_sql: str):
    """
    Creates a table from the create_table_sql statement
    Args:
        conn: Connection object
        create_table_sql: a CREATE TABLE statement

    Returns:

    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as err:
        print(err)


def insert_into_sqlite_table(conn: sqlite3.Connection, sqlite_insert_command: str,
                             task: Tuple):
    """
    Creates a new entry in a table specified by table_string
    Args:
        conn: Connection object
        sqlite_insert_command: SQL table insert string
        task: values to be inserted

    Returns:
        unique id of object created
    """
    cur = conn.cursor()
    cur.execute(sqlite_insert_command, task)
    conn.commit()
    return cur.lastrowid


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