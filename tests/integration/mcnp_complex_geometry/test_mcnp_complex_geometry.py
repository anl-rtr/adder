import pytest
from tests import mcnp_plate_lattice
from tests.testing_harness import TestHarness


class ComplexGeometryHarness(TestHarness):
    def __init__(self):

        super().__init__([])

        # Warning 1 and 2 should be given for the complement operator being
        # used and warning 3 should be given for the lattice being split.
        warning_1 = "ADDER does not modify complement operators on cells " \
                    "during shuffling. Cells [201, 202, 203, 204, 205] are " \
                    "members of a shuffled universe (universe 1000) and " \
                    "are attached to a complement operator in the definition " \
                    "of another cell. The user must confirm that any " \
                    "shuffling of this universe still results in a correct " \
                    "region definition with the original complement."

        warning_2 = "ADDER does not modify complement operators on cells " \
                    "during cloning. Cells [201, 202, 203, 204, 205] are " \
                    "members of a universe (universe 1000) being cloned and " \
                    "are attached to a complement operator in the definition " \
                    "of another cell. The user must confirm that cloning of " \
                    "this universe still results in a correct region " \
                    "definition with the original complement."

        warning_3 = "There are more cells in universe 1016 than the universe " \
                    "it was cloned from due to a lattice structure being " \
                    "split up. The volumes for cells in universe 1016 should " \
                    "be checked."

        self.log_messages = [('warning', warning_1, 1),
                             ('warning', warning_2, 1),
                             ('warning', warning_3, 1)]

    def execute_test(self):
        """Run ADDER as in TestHarness, get results, then cleanup."""
        try:
            self._create_test_lib()
            self._run_adder()
            self._get_results()

        finally:
            self._cleanup()

    def _build_inputs(self):
        with open("test.inp", mode="w") as mcnp_input_file:
            mcnp_input_file.write(mcnp_plate_lattice)


def test_mcnp_complex_geometry():
    # This tests the execution of MCNP with a model that has a lattice that will
    # be broken up and a complement operator used with universe shuffling and
    # cloning.

    test = ComplexGeometryHarness()
    test._build_inputs()
    test.main()
