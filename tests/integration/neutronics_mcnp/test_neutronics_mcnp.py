import pytest
import os
import numpy as np
from tests import mcnp_2x2x2
from tests import default_config as config
from tests.testing_harness import TestHarness
from adder.depletionlibrary import DepletionLibrary

ADDER_EXEC = config["exe"]

class MCNPHarness(TestHarness):
    def execute_test(self):
        """Run ADDER as in TestHarness, get results, run again and make sure
        the results are the same and that MCNP was skipped. Then cleanup"""
        try:
            self._create_test_lib()
            # Run 1
            self._run_adder()
            results, exec_skip_lines_1 = self._get_results()
            self._write_results(results)
            self._compare_results()

            # Run 2 (w/ ff)
            self._run_adder_ff()
            results, exec_skip_lines_2 = self._get_results()
            self._write_results(results)
            self._compare_results()

            # Now we know that we got the same results, lets just verify that
            # the first pass actually executed mcnp while the second skipped it
            for line in exec_skip_lines_1:
                assert "Executing MCNP" in line
            for line in exec_skip_lines_2:
                assert "Skipping MCNP" in line
        finally:
            self._cleanup()

    def update_results(self):
        """Update the results_true using the current version of OpenMC."""
        try:
            self._create_test_lib()
            self._run_adder()
            results, _ = self._get_results()
            self._write_results(results)
            self._overwrite_results()
        finally:
            self._cleanup()

    def _get_outputs(self):
        # Perform upstream output 'getting', and also look at the log for
        # the MCNP execution statements
        outstr = super()._get_outputs()

        # Now get from the log whether or not we skipped MCNP
        exec_skip_lines = []
        with open("adder.log", "r") as fin:
            for line in fin.readline():
                if "Executing MCNP" in line or "Skipping MCNP" in line:
                    exec_skip_lines.append(line[30:].ltrim())
        return outstr, exec_skip_lines

    def _run_adder_ff(self):
        os.system("{} ./{} -f".format(ADDER_EXEC, self.input_fname))

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2)

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""
        self._compare_library("output_depletion.h5", ["hi"], ["1", "None"])
        self._compare_library("output_all_modified.h5", [], ["11", "12"])

        # And perform the rest
        super()._compare_results()

    def _compare_library(self, output_lib_name, matching, unmatching):
        # Create the reference A-matrix
        flux = np.array([1.E13])
        ref_A = np.zeros((7, 7))
        # Test matrix created by manually inspecting ref lib A matrix
        # and adding in rows/cols for the new isotopes added from the
        # library during ADDER execution
        ref_A[3, 5] = 1.2E-11
        ref_A[4, 5] = 8.E-12
        ref_A[5, 5] = -1.E-11

        for name in matching:
            # Load the testing library
            test_lib = DepletionLibrary.from_hdf5(output_lib_name, name)

            # Now make sure we have the correct data; most direct and
            # concise way is to just compare the depletion matrices
            test_A = test_lib.build_depletion_matrix(flux).todense()
            np.testing.assert_allclose(ref_A, test_A, rtol=1.e-15)

        for name in unmatching:
            # Load the testing library
            test_lib = DepletionLibrary.from_hdf5(output_lib_name, name)

            # Now make sure we have the correct data; most direct and
            # concise way is to just compare the depletion matrices
            test_A = test_lib.build_depletion_matrix(flux).todense()
            # This material doesn't have H1 so the lib doesnt have it
            # either
            assert list(test_lib.isotopes.keys()) == \
                ["H1", "He4", "I134", "I135", "O16", "U235", "U238"]
            nz = np.nonzero(test_A)
            ref_nz = np.nonzero(ref_A)
            assert len(nz[0]) == len(np.nonzero(ref_A)[0])
            # Check the non zero entries, but take off 1 since we lost
            # a row/col by not having H1
            assert np.array_equal(nz[0], (ref_nz[0]))
            assert np.array_equal(nz[1], (ref_nz[1]))
            assert np.all(((test_A[nz] - ref_A[ref_nz]) / ref_A[ref_nz]) >
                          1.e-13)


def test_neutronics_mcnp():
    # This tests the execution of MCNP

    output_text_files = ["state{}.inp".format(i) for i in range(4)]
    test = MCNPHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()
