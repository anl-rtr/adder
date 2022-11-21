import pytest

from tests import default_config
import os


def pytest_addoption(parser):
    parser.addoption('--update', action='store_true')
    parser.addoption('--plot', action='store_true')


def pytest_configure(config):
    opts = ['update', 'plot']
    for opt in opts:
        if config.getoption(opt) is not None:
            default_config[opt] = config.getoption(opt)


@pytest.fixture
def run_in_tmpdir(tmpdir):
    orig = tmpdir.chdir()
    try:
        yield
    finally:
        orig.chdir()

@pytest.fixture(autouse=True, scope='function')
def cleanup(request):
    def do_it():
        files = ["adder.log", "test.h5", "test_lib.h5", "test_rx.h5",
                 "test.lib", "test_in.lib", "test_out.lib", "test_1g.lib",
                 "test_2g.lib"]
        for file in files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
    request.addfinalizer(do_it)
