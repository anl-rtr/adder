import pytest
import numpy as np

from adder.depletionlibrary import DepletionLibrary, ReactionData, DecayData, \
    YieldData


@pytest.fixture(scope='module')
def depletion_lib():
    depllib = DepletionLibrary("test", np.array([0., 20.]))

    # He4
    he4dk = DecayData(None, "s", 0.)
    depllib.add_isotope("He4", decay=he4dk)

    # U235
    u235xs = ReactionData("b", 1)
    u235xs.add_type("fission", "b", [100.0])
    u235dk = DecayData(np.log(2.) / 0.5, "s", 200.)
    u235dk.add_type("alpha", 1., "Th231")
    u235yd = YieldData()
    u235yd.add_isotope("H1", 0.5)
    u235yd.add_isotope("O16", 0.5)
    depllib.add_isotope("U235", xs=u235xs, nfy=u235yd, decay=u235dk)

    # U235 alpha decays to the stable and non-absorbing Th231
    th231dk = DecayData(None, "s", 201.)
    depllib.add_isotope("Th231", decay=th231dk)

    # U238
    u238xs = ReactionData("b", 1)
    u238xs.add_type("fission", "b", [10.0], "fission")
    u238yd = YieldData()
    u238yd.add_isotope("H1", 1.)
    u238yd.add_isotope("N16", 1.)
    u238dk = DecayData(None, "s", 202.)
    depllib.add_isotope("U238", xs=u238xs, nfy=u238yd, decay=u238dk)

    # H1, stable so all H1 made as a fission product stays as H1
    h1dk = DecayData(None, "s", 0.)
    depllib.add_isotope("H1", decay=h1dk)

    # H1 absorbs a neutron and becomes H2 so add H2 without any
    # decay or xs absorption
    depllib.add_isotope("H2")

    # O16
    o16dk = DecayData(np.log(2.) / 2., "s", 0.)
    o16dk.add_type("ec/beta+", 1., ["N16"])
    depllib.add_isotope("O16", decay=o16dk)

    # O16 decays to N16 per the beta+ec channel above, so lets make
    # a stable N16 without any xs
    n16dk = DecayData(None, "s", 0.)
    depllib.add_isotope("N16", decay=n16dk)

    depllib.finalize_library()

    return depllib


@pytest.fixture(scope='module')
def simple_lib():
    lib = DepletionLibrary("test", np.array([0., 20.]))

    u235xs = ReactionData("b", 1)
    u235xs.add_type("fission", "b", np.array([10.]), yields_=[1.])
    lib.add_isotope("U235", xs=u235xs)

    u238xs = ReactionData("b", 1)
    u238xs.add_type("fission", "b", np.array([1.]))
    lib.add_isotope("U238", xs=u238xs)
    lib.finalize_library()

    # Test the pseudo library class
    assert lib.get_1g_micro_xs("U235", "fission", np.array([1.0]), True) == 10.
    assert lib.get_1g_micro_xs("U235", "fission", np.array([0.0]), True) == 0.
    assert lib.get_1g_micro_xs("U238", "fission", np.array([1.0]), True) == 1.
    assert lib.get_1g_micro_xs("Np235", "fission", np.array([1.0]), True) == 0.
    assert (lib._get_Qrec("U235", 1.) - 201.70114397139) < 1.e-15
    assert (lib._get_Qrec("U235", 1.) - 202.77378137195) < 1.e-15
    assert lib._get_Qrec("Np235", 1.) == 0.
    Q, FR = lib.get_composition_Q_fiss_rate(["U235", "U238", "Np235"],
                                            [1., 1., 1.],
                                            [1.], "correlation")
    refQ = 10. * 201.70114397139 + 1. * 202.77378137195 + 1. * 0.
    refFR = 10. + 1.
    assert (Q - refQ) < 1.e-15
    assert (FR - refFR) < 1.e-15

    return lib


@pytest.fixture(scope='module')
def simple_lib_h1u234():
    lib = DepletionLibrary("test", np.array([0., 20.]))

    u235xs = ReactionData("b", 1)
    u235xs.add_type("fission", "b", np.array([10.]), yields_=[1.])
    lib.add_isotope("U235", xs=u235xs)

    u238xs = ReactionData("b", 1)
    u238xs.add_type("fission", "b", np.array([1.]))
    lib.add_isotope("U238", xs=u238xs)

    h1xs = ReactionData("b", 1)
    h1xs.add_type("(n,gamma)", "b", np.array([1.]), "U235")
    lib.add_isotope("H1", xs=h1xs)

    u234xs = ReactionData("b", 1)
    u234xs.add_type("(n,gamma)", "b", np.array([1.]), "U235")
    lib.add_isotope("U234", xs=u234xs)

    lib.finalize_library()

    return lib


@pytest.fixture(scope='module')
def depletion_lib_2g():
    # Creates a 2G library
    depllib = DepletionLibrary("test", np.array([0., 1.e-6, 20.]))

    # He4
    he4dk = DecayData(None, "s", 0.)
    depllib.add_isotope("He4", decay=he4dk)

    # U235
    u235xs = ReactionData("b", 2)
    u235xs.add_type("fission", "b", [50.0, 125.0])
    u235dk = DecayData(np.log(2.) / 0.5, "s", 200.)
    u235dk.add_type("alpha", 1., "Th231")
    u235yd = YieldData()
    u235yd.add_isotope("H1", 0.5)
    u235yd.add_isotope("O16", 0.5)
    depllib.add_isotope("U235", xs=u235xs, nfy=u235yd, decay=u235dk)

    # U235 alpha decays to the stable and non-absorbing Th231
    th231dk = DecayData(None, "s", 201.)
    depllib.add_isotope("Th231", decay=th231dk)

    # U238
    u238xs = ReactionData("b", 2)
    u238xs.add_type("fission", "b", [5.0, 12.5])
    u238yd = YieldData()
    u238yd.add_isotope("H1", 1.)
    u238yd.add_isotope("N16", 1.)
    u238dk = DecayData(None, "s", 202.)
    depllib.add_isotope("U238", xs=u238xs, nfy=u238yd, decay=u238dk)

    # H1, stable so all H1 made as a fission product stays as H1
    h1dk = DecayData(None, "s", 0.)
    depllib.add_isotope("H1", decay=h1dk)

    # Add H2 without any source or losses due to decay or xs absorption
    depllib.add_isotope("H2")

    # O16
    o16dk = DecayData(np.log(2.) / 2., "s", 0.)
    o16dk.add_type("ec/beta+", 1., ["N16"])
    depllib.add_isotope("O16", decay=o16dk)

    # O16 decays to N16 per the beta+ec channel above, so lets make
    # a stable N16 without any xs
    n16dk = DecayData(None, "s", 0.)
    depllib.add_isotope("N16", decay=n16dk)
    depllib.finalize_library()

    return depllib
