import pytest
from tests import mcnp_lattice
from tests.testing_harness import TestHarness


class ShuffleHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_lattice)


def test_mcnp_shuffle_byuniv():
    # This tests MCNPs ability to shuffle fuel using universes

    output_text_files = ["state{}.inp".format(i + 1) for i in range(13)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()
