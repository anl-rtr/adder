import adder
from adder.cram import CRAMDepletion
import pytest
import numpy as np


def test_depletion_init():
    # Simple test of initialization of the CRAMDepletion class

    # We will provide values, apply them with the wrong types and see
    # if it fails.

    exec_cmd = ""
    num_threads = 20
    num_procs = 200
    order = 16
    chunksize = 100

    # Check the type and value checks of each of the input parameters
    # Check exec_cmd
    with pytest.raises(TypeError):
        test_d = CRAMDepletion(1, num_threads, num_procs, chunksize, order)
    # Check num_threads
    with pytest.raises(TypeError):
        test_d = CRAMDepletion(exec_cmd, str(num_threads), num_procs,
                               chunksize, order)
    with pytest.raises(ValueError):
        test_d = CRAMDepletion(exec_cmd, 0, num_procs, chunksize, order)
    # Check num_procs
    with pytest.raises(TypeError):
        test_d = CRAMDepletion(exec_cmd, num_threads,
                               str(num_procs), chunksize, order)
    with pytest.raises(ValueError):
        test_d = CRAMDepletion(exec_cmd, num_threads, 0, chunksize, order)
    # Check chunksize
    with pytest.raises(TypeError):
        test_d = CRAMDepletion(exec_cmd, num_threads,
                               num_procs, str(chunksize), order)
    with pytest.raises(ValueError):
        test_d = CRAMDepletion(exec_cmd, num_threads, num_procs, 0, order)
    # Check order
    with pytest.raises(ValueError):
        test_d = CRAMDepletion(exec_cmd, num_threads, num_procs, chunksize, "16")
    with pytest.raises(ValueError):
        test_d = CRAMDepletion(exec_cmd, num_threads, num_procs, chunksize, 17)

    # Check that the attributes exist and their values are set correctly
    test_d = CRAMDepletion(exec_cmd, num_threads, num_procs, chunksize, order)
    assert test_d.solver == "cram16"
    assert test_d.exec_cmd == exec_cmd
    assert test_d.num_threads == num_threads
    assert test_d.num_procs == num_procs
    assert test_d.chunksize == chunksize
    # We verify alpha, theta and alpha0 by simply checking the first and
    # last values
    assert test_d.alpha0 == 2.124853710495224e-16
    assert len(test_d.alpha) == 8
    assert test_d.alpha[0] == np.complex128(5.464930576870210e+3 -
                                            3.797983575308356e+4j)
    assert test_d.alpha[-1] == np.complex128(2.394538338734709e+1 -
                                             5.650522971778156e+0j)
    assert len(test_d.theta) == 8
    assert test_d.theta[0] == np.complex128(3.509103608414918 +
                                            8.436198985884374j)
    assert test_d.theta[-1] == np.complex128(-10.84391707869699 +
                                             19.27744616718165j)

    # Repeat for CRAM48
    test_d = CRAMDepletion(exec_cmd, num_threads, num_procs, chunksize, 48)
    assert test_d.solver == "cram48"
    # We verify alpha, theta and alpha0 by simply checking the first and
    # last values
    assert test_d.alpha0 == 2.258038182743983e-47
    assert len(test_d.alpha) == 24
    assert test_d.alpha[0] == np.complex128(6.387380733878774e+2 -
                                            6.743912502859256e+2j)
    assert test_d.alpha[-1] == np.complex128(1.041366366475571e+2 -
                                             2.777743732451969e+2j)
    assert len(test_d.theta) == 24
    assert test_d.theta[0] == np.complex128(-4.465731934165702e+1 +
                                            6.233225190695437e+1j)
    assert test_d.theta[-1] == np.complex128(1.316284237125190e+1 +
                                             2.042951874827759e+1j)
