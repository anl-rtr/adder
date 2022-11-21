import adder
import pytest


def test_isotope_init():
    # Tests the initialization of an Isotope object

    # Set the test parameters
    name = "Am241_m1"
    xs_library = "80c"
    is_depleting = False

    # We need to check the type-checks of name, xs_library, is_depleting
    with pytest.raises(TypeError):
        test_iso = adder.Isotope(241, xs_library, is_depleting)

    with pytest.raises(TypeError):
        test_iso = adder.Isotope(name, 80, is_depleting)

    with pytest.raises(TypeError):
        test_iso = adder.Isotope(name, xs_library, str(is_depleting))

    # Check that the attributes exist and their values are set correctly
    test_iso = adder.Isotope(name, xs_library, is_depleting)
    assert test_iso.name == name
    assert test_iso.Z == 95
    assert test_iso.A == 241
    assert test_iso.M == 1
    assert test_iso.xs_library == xs_library
    assert test_iso.is_depleting == is_depleting

    # Now check the default value of is_depleting is applied
    test_iso = adder.Isotope(name, xs_library)
    assert test_iso.Z == 95
    assert test_iso.A == 241
    assert test_iso.M == 1
    assert test_iso.xs_library == xs_library
    assert test_iso.is_depleting


def test_isotope_atomic_mass():
    # Test a value with the known reference from the
    # Atomic Mass Evaluation 2016
    # <https://www-nds.iaea.org/amdc/ame2016/AME2016-a.pdf>
    name = "O16"
    xs_library = "80c"
    test_iso = adder.Isotope(name, xs_library)
    assert test_iso.atomic_mass == 15.99491461960

    # Test an elemental value
    name = "C0"
    xs_library = "80c"
    test_iso = adder.Isotope(name, xs_library)
    assert abs(test_iso.atomic_mass - 12.0111151648645) < 1.e-13
