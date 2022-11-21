import pytest
import os
import h5py
import numpy as np
from tests.testing_harness import TestHarness

inp_file = """Cylinder rod
c Control rod in the middle of a cell of finite height
1 1 9.8064637E-02    -1  10 -11 imp:n=1    $ control rod bottom (non-absorber)
2 2 -2.37   -1  11 -12 imp:n=1    $ control rod itself
3 1 9.8064637E-02  1 -2  10 -12 imp:n=1    $ homogenized fuel
4 0       #1 #2 #3   imp:n=0    $ problem exterior

c Define surfaces
1 cz 2.     $ control rod cylinder
2 cz 20.    $ reactor cylinder
10 pz 0.    $ rx bottom
11 pz 0.  $ CR position to be transformed
12 pz 20.   $ rx top

c define materials
m1 92235.70c 9.8467E-4 92238.70c 7.7667E-5 1001.70c 5.81E-2
     8016.70c 3.6927E-2 nlib=70c
m2 53135.70c 0.2 54135.70c 0.8 nlib=70c
kcode 100 1.0 20 50
ksrc 4. 4. 5. 4. -4. 5. -4. 4. 5. -4. -4. 5.
"""


class GeomSweepHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(inp_file)

    def _get_outputs(self):
        # This should be the same as upstream, except, we also want to get
        # the group displacement information from the HDF5 file
        outstr = super()._get_outputs()

        # Get the HDF5 file and obtain the control group displacement in time
        h5_group_names = ["case_1/operation_1/step_1/control_groups/bank_1/",
                          "case_1/operation_1/step_2/control_groups/bank_1/",
                          "case_1/operation_1/step_3/control_groups/bank_1/",
                          "case_1/operation_1/step_4/control_groups/bank_1/",
                          "case_1/operation_1/step_5/control_groups/bank_1/",
                          "case_1/operation_2/step_1/control_groups/bank_1/",
                          "case_1/operation_2/step_2/control_groups/bank_1/",
                          "case_1/operation_2/step_3/control_groups/bank_1/",
                          "case_2/operation_2/step_1/control_groups/bank_1/",
                          "case_2/operation_3/step_1/control_groups/bank_1/",
                          "case_2/operation_3/step_2/control_groups/bank_1/"]

        test_displacements = []
        with h5py.File("results.h5", "r") as h5:
            for group_name in h5_group_names:
                grp = h5[group_name]
                test_displacements.append(str(float(grp.attrs["displacement"])))

        outstr += "\n Control Group Displacements\n"
        outstr += "\n".join(test_displacements)

        return outstr


def test_mcnp_surf_sweep():
    # This tests MCNPs ability to shuffle fuel

    output_text_files = [
        "case_1_op_1_step_{}.inp".format(i + 1) for i in range(5)]
    output_text_files += [
        "case_1_op_2_step_{}.inp".format(i + 1) for i in range(3)]
    output_text_files += ["post_transform.inp"]
    output_text_files += [
        "case_2_op_3_step_{}.inp".format(i + 1) for i in range(2)]
    test = GeomSweepHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()


