import pytest
import re
from tests import mcnp_2x2x2_trcl
from tests.testing_harness import TestHarness


class ShuffleHarness(TestHarness):
    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2_trcl)

    def _build_inputs_no_volumes(self):
        mcnp_2x2x2_trcl_no_volumes = re.sub("vol 1000.0 7R 0.0", "c",
                                            mcnp_2x2x2_trcl)
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_2x2x2_trcl_no_volumes)


def test_mcnp_shuffle_bymat():
    # This tests MCNPs ability to shuffle fuel. The second test is the same as
    # the first, but the volumes are removed from the materials in the MCNP
    # input to check if ADDER raises a warning during the shuffling of materials
    # with no volumes.

    output_text_files = ["state{}.inp".format(i + 1) for i in range(8)]
    test = ShuffleHarness(output_text_files, "test.h5")
    test._build_inputs()
    test.main()

    # Warning message given when materials with no volumes are shuffled.
    warning_message = "At least one of the in-core materials being shuffled " \
                      "does not have a volume set!"

    output_text_files = ["state0.inp"]
    test = ShuffleHarness(output_text_files, "test.h5")
    test.log_messages = [('warning', warning_message, 9)]
    test._build_inputs_no_volumes()
    test._create_test_lib()
    test._run_adder()
    test._check_log_messages()
    test._cleanup()
