import pytest
from tests.testing_harness import TestHarness
import os


class DepleteHarness(TestHarness):
    def _build_inputs(self):
        # Write a blank file since adder is going to look that the
        # file exists, even though our test neutronics solver doesnt
        # need it.
        with open("test.inp", mode="w") as input_file:
            input_file.write("")

    def _get_results(self):
        """Digest info in the results.h5 and return as a string."""

        # Process results.h5 using adder_extract_materials (testing this too)
        os.system("../../../scripts/adder_extract_materials.py results.h5 mat_1.csv 1")
        os.system("../../../scripts/adder_extract_materials.py results.h5 mat_2.csv 2")

        outstr = "mat 1\n"
        with open("mat_1.csv", "r") as fin:
            outstr += "".join(fin.readlines())
        outstr += "\nmat 2\n"
        with open("mat_2.csv", "r") as fin:
            outstr += "".join(fin.readlines())

        return outstr

    def _cleanup(self):
        """Delete statepoints, tally, and test files."""

        # Do whatever the harness wants
        super()._cleanup()

        # And add our own
        for f in ["out.txt", "mat_1.csv", "mat_2.csv"]:
            if os.path.exists(f):
                os.remove(f)


def test_deplete_msr():
    test = DepleteHarness([""], "test.h5", "test.add")
    test._build_inputs()
    test.main()
