import filecmp
import pytest
import os
import numpy as np
import h5py
from tests.testing_harness import TestHarness

inp_file = """Cylinder rod test case
1 1 4e-3 -1  10 -11 imp:n=1    $ homogenized fuel
2 0      #1         imp:n=0    $ problem exterior

c Define surfaces
1 cz 4
10 pz 0.    $ rx bottom
11 pz 5.    $ A range of 5 to 25 is suitable. as the crit pos is ~14cm

c define material
m1 92235.70c 1.0 nlib=70c
kcode 30000 1.0 20 100
ksrc 0. 0. 1. 1. 1. 1. 1. -1. 1. -1. 1. 1. -1. -1. 1.
"""


class GeomSweepHarness(TestHarness):
    REF_RANGES = [(8.58, 8.84), (2.14, 2.22)]

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def update_results(self):
        pass

    def _get_results(self):
        """Digest info in the output and return as array of the 2 positions."""
        output = self._get_outputs()
        self.displacements = output

    def _write_results(self, results_string):
        pass

    def _overwrite_results(self):
        pass

    def _compare_results(self):
        """Make sure the current results agree with the _true standard."""

        assert len(self.REF_RANGES) == len(self.displacements)
        for i in range(len(self.REF_RANGES)):
            assert self.REF_RANGES[i][0] <= self.displacements[i] <= self.REF_RANGES[i][1]

    def _get_outputs(self):
        # Get the HDF5 file and obtain the control group displacement in time
        h5_group_names = ["case_1/operation_1/step_1/control_groups/bank_1/",
                          "case_2/operation_1/step_1/control_groups/bank_1/"]

        displacements = []
        with h5py.File("results.h5", "r") as h5:
            for group_name in h5_group_names:
                grp = h5[group_name]
                displacements.append(float(grp.attrs["displacement"]))

        return displacements


def test_mcnp_surf_search():
    output_text_files = []
    test = GeomSweepHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()


