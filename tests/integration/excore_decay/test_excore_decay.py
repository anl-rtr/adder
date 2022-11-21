import pytest
import filecmp
import glob
import os
import h5py
import numpy as np
from tests import mcnp_2x2x2
from tests.testing_harness import TestHarness
from adder.depletionlibrary import DepletionLibrary, DecayData, ReactionData, \
    YieldData

test_input = """Simple IHM problem to test depletion
1 1 1.0 1 -2 11 -12 21 -22 u=0 imp:n=1
99 0 #1        imp:n=0

*1  px  0.
*2  px 10.
*11 py  0.
*12 py 10.
*21 pz  0.
*22 pz 10.

m1  92235  1.0 nlib=70c
m2  92235  1.0 nlib=70c
kcode 10000 1.0 5 35
sdef  x=d1 y=d2 z=d3 erg=2
si1    0 10
sp1    0  1
si2    0 10
sp2    0  1
si3    0 10
sp3    0  1

"""


class ExCoreHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as input_file:
            input_file.write(test_input)

    def _get_results(self):
        """Digest info in the results.h5 and return as a string."""

        # Process results.h5 using adder_extract_materials and store in outstr
        outstr = ""
        mat_names = ["1", "2", "2[2]", "3"]
        cmd = "../../../scripts/adder_extract_materials.py " + \
            "results.h5 mat_{}.csv {}"
        for mat in mat_names:
            os.system(cmd.format(mat, mat))
            outstr += "mat {}\n".format(mat)
            with open("mat_{}.csv".format(mat), "r") as fin:
                outstr += "".join(fin.readlines())
            outstr += "\n"
        return outstr

    def _cleanup(self):
        """Delete statepoints, tally, and test files."""
        # Do whatever the harness wants
        super()._cleanup()

        # And add our own
        output = glob.glob('*csv*')
        for f in output:
            if os.path.exists(f):
                os.remove(f)

    def _create_test_lib(self):
        # This will be a simple depletion library
        depllib = DepletionLibrary("test", np.array([0., 20.]))

        # He4
        he4dk = DecayData(None, "s", 0.)
        depllib.add_isotope("He4", decay=he4dk)

        # U235
        u235xs = ReactionData("b", 1)
        u235xs.add_type("fission", "b", [1.0])
        u235dk = DecayData(1.0, "d", 5.)
        u235dk.add_type("alpha", 1., ["Th231"])
        u235yd = YieldData()
        u235yd.add_isotope("I135", 2. * 0.4)
        u235yd.add_isotope("I134", 2. * 0.6)
        depllib.add_isotope("U235", xs=u235xs, decay=u235dk, nfy=u235yd)

        # Th231, stable
        th231dk = DecayData(None, "s", 0.)
        depllib.add_isotope("Th231", decay=th231dk)

        # I135, stable
        i135dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I135", decay=i135dk)

        # I134, stable
        i134dk = DecayData(None, "s", 0.)
        depllib.add_isotope("I134", decay=i134dk)

        depllib.to_hdf5(self._test_lib_name)

        self._create_ce_data()

def test_excore_decay():
    # This tests the ex-core decay handling
    test = ExCoreHarness([""], "test.h5", "test.add")
    test._build_inputs()
    test.main()
