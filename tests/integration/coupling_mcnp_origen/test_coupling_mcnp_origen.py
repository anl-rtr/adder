import pytest
from tests.testing_harness import CouplingHarness

def test_coupling_mcnp_origen():
    test = CouplingHarness("ORIGEN2.2", "predictor", 8, test_lib_name="test.h5",
                           input_fname="test.add")

    # Run two predictors, increasing the num time steps, and then repeat
    # for two cecms
    for method in ["predictor", "cecm"]:
        test.depl_meth = method
        for num in [8, 16]:
            test.num_adder_t = num
            # Create a new library each time because cleanup is cleaning
            # them all
            test._create_test_lib()
            # and execute
            test.main()
