import adder
from adder.origen22 import Origen22Depletion
import pytest
import numpy as np


def test_depletion_init():
    # Simple test of initialization of the Origen22Depletion class

    # We will provide values, apply them with the wrong types and see
    # if it fails.

    exec_cmd = "origen2.EXE"
    num_threads = 1
    num_procs = 2
    chunksize = 3

    # Check the type and value checks of each of the input parameters
    # Check exec_cmd
    with pytest.raises(TypeError):
        test_d = Origen22Depletion(1, num_threads, num_procs, chunksize)
    # Check num_threads
    with pytest.raises(TypeError):
        test_d = Origen22Depletion(exec_cmd, str(num_threads), num_procs,
            chunksize)
    with pytest.raises(ValueError):
        test_d = Origen22Depletion(exec_cmd, 0, num_procs, chunksize)
    # Check num_procs
    with pytest.raises(TypeError):
        test_d = Origen22Depletion(exec_cmd, num_threads, str(num_procs),
            chunksize)
    with pytest.raises(ValueError):
        test_d = Origen22Depletion(exec_cmd, num_threads, 0, chunksize)
    # Check chunksize
    with pytest.raises(TypeError):
        test_d = Origen22Depletion(exec_cmd, num_threads, num_procs,
                                   str(chunksize))
    with pytest.raises(ValueError):
        test_d = Origen22Depletion(exec_cmd, num_threads, num_procs, 0)

    # Check that the attributes exist and their values are set correctly
    test_d = Origen22Depletion(exec_cmd, num_threads, num_procs, chunksize)
    assert test_d.solver == "origen2.2"
    assert test_d.exec_cmd == exec_cmd
    assert test_d.num_threads == num_threads
    assert test_d.num_procs == num_procs
    assert test_d.chunksize == chunksize
