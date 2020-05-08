import slapp.utils.query_utils as qu
import pytest
import os


@pytest.mark.parametrize(
        "env_prefix, user, password, defaults, expected",
        [
            (
                "PYTEST_DB_CREDS",
                "secret_user",
                "secret_password",
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234
                    },
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    'user': "secret_user",
                    'password': "secret_password"
                    })])
def test_get_db_credentials(
        env_prefix, user, password, defaults, expected):
    os.environ[env_prefix + 'USER'] = user
    os.environ[env_prefix + 'PASSWORD'] = password
    try:
        creds = qu.get_db_credentials(env_prefix=env_prefix, **defaults)
        assert creds == expected
    finally:
        os.environ.pop(env_prefix+'USER')
        os.environ.pop(env_prefix+'PASSWORD')


@pytest.mark.parametrize(
        "env_prefix, skipos, user, password, defaults, expected",
        [
            (
                "PYTEST_DB_CREDS",
                False,
                "secret_user",
                "secret_password",
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    },
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    'user': "secret_user",
                    'password': "secret_password"
                    }),
            (
                "PYTEST_DB_CREDS",
                True,
                "secret_user",
                "secret_password",
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    },
                {
                    'host': 'myserver',
                    'database': 'mydatabase',
                    'port': 1234,
                    'user': "secret_user",
                    'password': "secret_password"
                    }),

                ])
def test_get_db_credentials_exceptions(
        env_prefix, skipos, user, password, defaults, expected):
    if not skipos:
        os.environ[env_prefix + 'USER'] = user
        os.environ[env_prefix + 'PASSWORD'] = password
    try:
        with pytest.raises(qu.CredentialsException):
            qu.get_db_credentials(env_prefix=env_prefix, **defaults)
    finally:
        if not skipos:
            os.environ.pop(env_prefix+'USER')
            os.environ.pop(env_prefix+'PASSWORD')
