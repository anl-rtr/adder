import pytest


@pytest.fixture(scope='module', autouse=True)
def setup_integration_test(request):
    # Change to test directory
    olddir = request.fspath.dirpath().chdir()
    try:
        yield
    finally:
        olddir.chdir()
