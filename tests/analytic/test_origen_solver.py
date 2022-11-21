import adder
from adder.origen22 import Origen22Depletion
import pytest
from tests import default_config as config
import numpy as np

EXEC_CMD = config["origen_exe"]


def test_single_decay():
    # Initialize our depletion library for a single 2-isotope system
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    # The MAGIC nuclide will actually be Xe135
    magicdk = adder.DecayData(6., "hr", 0.)
    magicdk.add_type("beta-", 1., "Cs135")
    depllib.add_isotope("Xe135", decay=magicdk)

    # The dump nuclide for MAGIC is Cs135
    cs135dk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("Cs135", decay=cs135dk)
    depllib.set_isotope_indices()

    ref_soln = np.zeros((2, 6))
    ref_soln[0, :] = \
        np.array([6.250000000000000E+22, 3.906250000000000E+21,
                  2.441406250000000E+20, 1.525878906250000E+19,
                  9.536743164062520E+17, 5.960464477539070E+16])
    ref_soln[1, :] = \
        np.array([9.375000000000000E+23, 9.960937500000000E+23,
                  9.997558593750000E+23, 9.999847412109370E+23,
                  9.999990463256840E+23, 9.999999403953550E+23])

    def deplete_substeps(num_substeps, lib):
        num_threads = 1
        num_procs = 1
        chunksize = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, chunksize)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material
        mat = adder.Material("test", 1, 1.,
                             [("Xe135", "70c", True)], [1.], True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = 1.E13 * np.ones(3)
        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete
        for i, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.atom_fractions,
                                       ref_soln[:, i] / np.sum(ref_soln[:, i]),
                                       rtol=1E-4)
            np.testing.assert_allclose(mat.number_densities, ref_soln[:, i],
                                       rtol=1E-4)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib)
    deplete_substeps(4, depllib)


def test_branch_decay():
    # Initialize our depletion library
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))
    tot_decay = np.log(2.) / (6. * 3600.)
    brs = np.array([0.1, 0.35, 0.25, 0.3])
    # Modify the verification case into valid ORIGEN decay channels
    decay_types = ["beta-", "ec/beta+", "alpha", "it"]
    targets = [["Cs135", "Cs135_m1"], ["I135", "I135_m1"], ["Te131"],
               ["Xe135"]]
    yields = [[0.025 / 0.1, 0.075 / 0.1], [0.15 / 0.35, 0.20 / 0.35],
              [1.], [1.]]
    stable = adder.DecayData(None, "s", 0.)

    magicdk = adder.DecayData(6., "hr", 0.)

    for i, type_ in enumerate(decay_types):
        magicdk.add_type(type_, brs[i], targets[i], yields[i])
        # Now add in the isotope we just made, as stable
        for iso_name in targets[i]:
            depllib.add_isotope(iso_name, decay=stable)

    depllib.add_isotope("Xe135_m1", decay=magicdk)
    # Also add in an alpha
    depllib.add_isotope("He4", decay=stable)
    depllib.set_isotope_indices()

    ref_soln = np.array([
        [6.250000000000000E+22, 3.906250000000000E+21, 2.441406250000000E+20,
         1.525878906250000E+19, 9.536743164062520E+17, 5.960464477539070E+16],
        [2.343750000000000E+22, 2.490234375000000E+22, 2.499389648437500E+22,
         2.499961853027340E+22, 2.499997615814210E+22, 2.499999850988390E+22],
        [7.031250000000000E+22, 7.470703125000000E+22, 7.498168945312500E+22,
         7.499885559082030E+22, 7.499992847442630E+22, 7.499999552965160E+22],
        [1.406250000000000E+23, 1.494140625000000E+23, 1.499633789062500E+23,
         1.499977111816410E+23, 1.499998569488530E+23, 1.499999910593030E+23],
        [1.875000000000000E+23, 1.992187500000000E+23, 1.999511718750000E+23,
         1.999969482421880E+23, 1.999998092651370E+23, 1.999999880790710E+23],
        [2.343750000000000E+23, 2.490234375000000E+23, 2.499389648437500E+23,
         2.499961853027340E+23, 2.499997615814210E+23, 2.499999850988390E+23],
        [2.343750000000000E+23, 2.490234375000000E+23, 2.499389648437500E+23,
         2.499961853027340E+23, 2.499997615814210E+23, 2.499999850988390E+23],
        [2.812500000000000E+23, 2.988281250000000E+23, 2.999267578125000E+23,
         2.999954223632810E+23, 2.999997138977050E+23, 2.999999821186070E+23]])

    # Now we need to reorder so that our isotope ordering matches what
    # will come from the depletion solver
    # Now it is [Xe135_m1, Cs135, Cs135_m1, I135, I135_m1, Te131, He4, Xe135]
    # We want [He4, Te131, I135, I135_m1, Xe135, Xe135_m1, Cs135, Cs135_m1]
    permutations = [6, 5, 3, 4, 7, 0, 1, 2]
    ref_val = np.zeros_like(ref_soln)
    for i in range(ref_val.shape[0]):
        ref_val[i, :] = ref_soln[permutations[i], :]

    def deplete_substeps(num_substeps, lib, ref_val):
        num_threads = 1
        num_procs = 1
        chunksize = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, chunksize)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material, density chosen to start with
        # a number density of 1
        mat = adder.Material("test", 1, 1.,
                             [("Xe135_m1", "70c", True)], [1.], True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = 1.E13 * np.ones(3)
        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        ref_isotopes = []
        for iso in ["He4", "Te131", "I135", "I135_m1", "Xe135", "Xe135_m1",
                    "Cs135", "Cs135_m1"]:
            ref_isotopes.append((iso, "70c", True))

        # And deplete
        for i, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_val[:, i],
                                       rtol=1E-4)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, ref_val)
    deplete_substeps(4, depllib, ref_val)


def test_fission_yield_decay():
    # Initialize our depletion library for a single 2-isotope system
    # consisting of U235 and Xe135
    depllib = adder.DepletionLibrary("test", np.array([0., 20.]))

    u235dk = adder.DecayData(np.log(2) / 1.407412E-05, "s", 0.)
    u235dk.add_type("sf", 1., "fission")
    u235nfy = adder.YieldData()
    u235nfy.add_isotope("Xe135", 2.)
    depllib.add_isotope("U235", decay=u235dk, nfy=u235nfy)

    xe135dk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("Xe135", decay=xe135dk)

    # Since ORIGEN will not use fission yields for spontaneous fission,
    # and it will instead dump to the pseudo-nuclide 162500.
    # We will add that isotope in to the library just so we can get its
    # output
    pseudodk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("S250", decay=pseudodk)

    depllib.set_isotope_indices()

    ref_vals = np.array(
        [[1.203587723580930E+24, 1.412139762388060E+24, 1.473957146962730E+24,
          1.492280578646770E+24, 1.497711868744050E+24, 1.499321769805680E+24],
         [1.482061382095360E+23, 4.393011880596840E+22, 1.302142651863740E+22,
          3.859710676613000E+21, 1.144065627973860E+21,
          3.391150971605470E+20]])

    def deplete_substeps(num_substeps, lib, ref_vals):
        num_threads = 1
        num_procs = 1
        chunksize = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, chunksize)
        test_d.isotope_types = {"actinide": set(["U235", "S250"]),
                                "fp": set(["Xe135"]),
                                "activation": set()}

        # Build the starting material
        mat = adder.Material("test", 1, 1.,
                             [("Xe135", "70c", True), ("U235", "70c", True)],
                             [0.5, 0.5], True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([1.E19])
        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete/compare
        for i, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            # We know that ORIGEN actually puts the spontaneous fission
            # into the pseudo-nuclide, 162500. So, we have to actually
            # compare a modified version of mat.number_densities to
            # the reference values
            mat_N = mat.number_densities
            test_mat_N = np.array([2. * mat_N[0] + mat_N[1], mat_N[2]])
            np.testing.assert_allclose(test_mat_N, ref_vals[:, i], rtol=5e-5)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, ref_vals)
    deplete_substeps(4, depllib, ref_vals)


def test_fission_yield_xs():
    # Initialize our depletion library for a single 2-isotope system
    # consisting of U235 and Xe135
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    u235xs = adder.ReactionData("b", 3)
    u235xs.add_type("fission", "b", 0.1 * np.ones(3))
    u235nfy = adder.YieldData()
    u235nfy.add_isotope("Xe135", 2.)
    depllib.add_isotope("U235", xs=u235xs, nfy=u235nfy)

    xe135dk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("Xe135", decay=xe135dk)
    depllib.set_isotope_indices()

    ref_vals = np.array(
        [[1.203587723580930E+24, 1.412139762388060E+24, 1.473957146962730E+24,
          1.492280578646770E+24, 1.497711868744050E+24, 1.499321769805680E+24],
         [1.482061382095360E+23, 4.393011880596840E+22, 1.302142651863740E+22,
          3.859710676613000E+21, 1.144065627973860E+21,
          3.391150971605470E+20]])

    def deplete_substeps(num_substeps, lib, ref_vals):
        num_threads = 1
        num_procs = 1
        chunksize = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, chunksize)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material
        mat = adder.Material("test", 1, 1.,
                             [("Xe135", "70c", True), ("U235", "70c", True)],
                             [0.5, 0.5], True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([1.83423, 6.54654, 5.69335]) * 1.E19
        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete/compare
        for i, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_vals[:, i],
                                       rtol=5e-5)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, ref_vals)
    deplete_substeps(4, depllib, ref_vals)


def test_multi_fission_yield():
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    fissile = ["Th232", "U233", "U235", "U238", "Pu239", "Pu241", "Cm245",
               "Cf252"]
    yields = [2., 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
    xss = 1.E-24 * np.array([0.1, 1.9048, 0.35419, 2.7751, 0.039382, .7,
                             .8, .9])
    yields = np.array(yields)
    for i in range(len(fissile)):
        xs = adder.ReactionData("b", 3)
        xs.add_type("fission", "cm2", xss[i] * np.ones(3))
        nfy = adder.YieldData()
        nfy.add_isotope("Xe135", yields[i])
        depllib.add_isotope(fissile[i], xs=xs, nfy=nfy)
    xe135dk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("Xe135", decay=xe135dk)
    depllib.set_isotope_indices()

    ref_soln = np.array(
        [[8.759274405669060E+23, 9.838912747822430E+22, 1.096037332455900E+22,
          7.222846759593330E+22, 3.803790577937050E+21, 1.059155452244680E+23,
          4.743346176486480E+22, 4.200243224685080E+22, 3.719324394657800E+22],
         [1.086411202837560E+24, 8.712378365333640E+22, 1.081168050723340E+21,
          4.695256378131110E+22, 1.302194048472240E+20, 1.009629244817670E+23,
          2.024939965499000E+22, 1.587783883186170E+22, 1.245003655742690E+22],
         [1.187557150245490E+24, 7.714829750628010E+22, 1.066500491626210E+20,
          3.052180558461980E+22, 4.457946107001950E+18, 9.624188874548770E+22,
          8.644492118667930E+21, 6.002170647855180E+21, 4.167515221417740E+21],
         [1.248101800348670E+24, 6.831498310266030E+22, 1.052031918514410E+19,
          1.984088921074280E+22, 1.526138405888870E+17, 9.174160907919810E+22,
          3.690343677487590E+21, 2.268951893735170E+21, 1.395030692531400E+21],
         [1.289480020521770E+24, 6.049306423044340E+22, 1.037759631864290E+18,
          1.289769321089370E+22, 5.224599800053130E+15, 8.745176290853930E+22,
          1.575411981527860E+21, 8.577134836917770E+20, 4.669714517425560E+20],
         [1.320506530099060E+24, 5.356673827305760E+22, 1.023680968775060E+17,
          8.384225545307930E+21, 1.788595514364050E+14, 8.336251034368940E+22,
          6.725451959074070E+20, 3.242344723737680E+20,
          1.563136480867370E+20]])
    isos = [("Xe135", "70c", True)] + [(iso, "70c", True) for iso in fissile]

    def deplete_substeps(num_substeps, lib, isos, ref_soln):
        num_threads = 1
        num_procs = 1
        chunksize = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, chunksize)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material, density chosen to start with
        # a number density of 1

        mat = adder.Material("test", 1, 1., isos, np.ones(len(isos)),
                             True, "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([1.83423, 6.54654, 5.69335]) * 1.E18
        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete
        for i, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_soln[i, :],
                                       rtol=1E-4)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, isos, ref_soln)
    deplete_substeps(4, depllib, isos, ref_soln)


def test_burn_actinide():
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    src = "Pu238"
    targets = [["Pu239", "Pu239_m1"], ["Pu237", "Pu237_m1"], ["Pu236"]]
    yields = [[0.5, 0.5], [0.5, 0.5], [1.]]
    types = ["(n,gamma)", "(n,2n)", "(n,3n)"]
    xss = 0.1 * np.ones(3)
    xs = adder.ReactionData("b", 3)
    stable_decay = adder.DecayData(None, "s", 0.)
    for i in range(len(targets)):
        if len(targets[i]) > 1:
            xs_vals = 2. * xss
        else:
            xs_vals = xss
        xs.add_type(types[i], "b", xs_vals, targets=targets[i],
                    yields_=yields[i])
        for t in targets[i]:
            depllib.add_isotope(t, decay=stable_decay)
    depllib.add_isotope(src, xs=xs)
    # And add in the secondary products
    # (This case has none, but the test_burn_nonactinide version does)
    secondaries = []
    for secondary in secondaries:
        depllib.add_isotope(secondary, decay=stable_decay)
    depllib.set_isotope_indices()

    ref_soln = np.array(
        [[2.727781759794730E+23, 2.727781759794730E+23, 2.277817597947300E+22,
          1.361091201026350E+23, 2.727781759794730E+23, 2.277817597947300E+22],
         [2.851794459399090E+23, 2.851794459399090E+23, 3.517944593990910E+22,
          7.410277030045430E+22, 2.851794459399090E+23, 3.517944593990910E+22],
         [2.919311497097900E+23, 2.919311497097900E+23, 4.193114970979000E+22,
          4.034425145105010E+22, 2.919311497097900E+23, 4.193114970979000E+22],
         [2.956070235470380E+23, 2.956070235470380E+23, 4.560702354703850E+22,
          2.196488226480760E+22, 2.956070235470380E+23, 4.560702354703850E+22],
         [2.976083033614230E+23, 2.976083033614230E+23, 4.760830336142330E+22,
          1.195848319288370E+22, 2.976083033614230E+23, 4.760830336142330E+22],
         [2.986978730998840E+23, 2.986978730998840E+23, 4.869787309988360E+22,
          6.510634500582230E+21, 2.986978730998840E+23,
          4.869787309988360E+22]])

    def deplete_substeps(num_substeps, lib, targets, ref_soln):
        num_threads = 1
        num_procs = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, 1)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material, density chosen to start with
        # a number density of 1
        isos = [src]
        for target in targets:
            isos += target
        num_frac = []
        for iso in isos:
            if not iso.endswith("_m1"):
                num_frac.append(1.)
            else:
                num_frac.append(0.)
        isos = [(iso, "70c", True) for iso in isos]
        mat = adder.Material("test", 1, 1., isos, num_frac, True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete
        for idx, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, i,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_soln[idx, :],
                                       rtol=3E-5)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, targets, ref_soln)
    deplete_substeps(4, depllib, targets, ref_soln)


def test_burn_notactinide():
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    src = "Xe135"
    # Ive already checked non-ground state in test_burn_actinide, so
    # we will skip that here
    targets = [["Xe136"], ["Xe134"], ["Te132"], ["I135"]]
    types = ["(n,gamma)", "(n,2n)", "(n,a)", "(n,p)"]
    xss = 0.1 * np.ones(3)
    xs = adder.ReactionData("b", 3)
    stable_decay = adder.DecayData(None, "s", 0.)
    for i in range(len(targets)):
        xs.add_type(types[i], "b", xss, targets=targets[i])
        for t in targets[i]:
            depllib.add_isotope(t, decay=stable_decay)
    depllib.add_isotope(src, xs=xs)
    # And add in the secondary products
    secondaries = ["H1", "He4"]
    for secondary in secondaries:
        depllib.add_isotope(secondary, decay=stable_decay)
    depllib.set_isotope_indices()

    ref_soln = np.array(
        [[2.407288398448310E+22, 2.407288398448310E+22, 2.407288398448310E+22,
          2.740728839844830E+23, 2.740728839844830E+23, 1.537084640620680E+23,
          2.740728839844830E+23],
         [3.887370807568010E+22, 3.887370807568010E+22, 3.887370807568010E+22,
          2.888737080756800E+23, 2.888737080756800E+23, 9.450516769727970E+22,
          2.888737080756800E+23],
         [4.797375582732300E+22, 4.797375582732300E+22, 4.797375582732300E+22,
          2.979737558273230E+23, 2.979737558273230E+23, 5.810497669070800E+22,
          2.979737558273230E+23],
         [5.356877327850900E+22, 5.356877327850900E+22, 5.356877327850900E+22,
          3.035687732785090E+23, 3.035687732785090E+23, 3.572490688596390E+22,
          3.035687732785090E+23],
         [5.700877943379810E+22, 5.700877943379810E+22, 5.700877943379810E+22,
          3.070087794337980E+23, 3.070087794337980E+23, 2.196488226480760E+22,
          3.070087794337980E+23],
         [5.912381168377230E+22, 5.912381168377230E+22, 5.912381168377230E+22,
          3.091238116837720E+23, 3.091238116837720E+23, 1.350475326491090E+22,
          3.091238116837720E+23]])

    def deplete_substeps(num_substeps, lib, targets, ref_soln):
        num_threads = 1
        num_procs = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, 1)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material, density chosen to start with
        # a number density of 1
        isos = [src]
        for target in targets:
            isos += target
        num_frac = []
        for iso in isos:
            if iso != "Te132":
                num_frac.append(1.)
            else:
                num_frac.append(0.)
        isos = [(iso, "70c", True) for iso in isos]
        mat = adder.Material("test", 1, 1., isos, num_frac, True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete
        for idx, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, idx,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_soln[idx, :],
                                       rtol=2E-5)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, targets, ref_soln)
    deplete_substeps(4, depllib, targets, ref_soln)


def test_simple_burn_and_decay():
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    src = "Xe135"
    # Incorporate the reactions
    targets = [["Xe136"], ["Xe134"], ["Te132"], ["I135"]]
    types = ["(n,gamma)", "(n,2n)", "(n,a)", "(n,p)"]
    xss = 0.1 * np.ones(3)
    xs = adder.ReactionData("b", 3)
    stable_decay = adder.DecayData(None, "s", 0.)
    for i in range(len(targets)):
        xs.add_type(types[i], "b", xss, targets=targets[i])
        for t in targets[i]:
            if t != "Xe134":
                depllib.add_isotope(t, decay=stable_decay)
            else:
                decay = adder.DecayData(np.log(2.) / 1.E-8, "s", 0.)
                decay.add_type("ec/beta+", 1., "I134")
                depllib.add_isotope(t, decay=decay)
    depllib.add_isotope(src, xs=xs)
    # And add in the secondary products
    secondaries = ["H1", "He4", "Te130", "I134"]
    for secondary in secondaries:
        depllib.add_isotope(secondary, decay=stable_decay)
    depllib.set_isotope_indices()

    ref_soln = np.array(
        [[2.407288398448300E+22, 2.407288398448300E+22, 2.407288398448300E+22,
          2.271425893260720E+20, 2.740728839844830E+23, 2.738457413951560E+23,
          1.537084640620670E+23, 2.740728839844830E+23],
         [3.887370807568000E+22, 3.887370807568000E+22, 3.887370807568000E+22,
          4.705513229683310E+20, 2.888737080756800E+23, 2.884031567527110E+23,
          9.450516769727970E+22, 2.888737080756800E+23],
         [4.797375582732300E+22, 4.797375582732300E+22, 4.797375582732300E+22,
          7.238714282960880E+20, 2.979737558273230E+23, 2.972498843990260E+23,
          5.810497669070800E+22, 2.979737558273230E+23],
         [5.356877327850900E+22, 5.356877327850900E+22, 5.356877327850900E+22,
          9.831958533320410E+20, 3.035687732785090E+23, 3.025855774251770E+23,
          3.572490688596380E+22, 3.035687732785090E+23],
         [5.700877943379810E+22, 5.700877943379810E+22, 5.700877943379810E+22,
          1.246122487560440E+21, 3.070087794337980E+23, 3.057626569462370E+23,
          2.196488226480750E+22, 3.070087794337980E+23],
         [5.912381168377220E+22, 5.912381168377220E+22, 5.912381168377220E+22,
          1.511174509361240E+21, 3.091238116837720E+23, 3.076126371744110E+23,
          1.350475326491080E+22, 3.091238116837720E+23]])

    def deplete_substeps(num_substeps, lib, targets, ref_soln):
        num_threads = 1
        num_procs = 1
        # Initialize our depletion solver
        test_d = Origen22Depletion(EXEC_CMD, num_threads, num_procs, 1)
        test_d.isotope_types = Origen22Depletion.assign_isotope_types(lib)

        # Build the starting material, density chosen to start with
        # a number density of 1
        isos = [src]
        for target in targets:
            isos += target
        num_frac = []
        for iso in isos:
            if not iso.startswith("Te"):
                num_frac.append(1.)
            else:
                num_frac.append(0.)
        isos = [(iso, "70c", True) for iso in isos]
        mat = adder.Material("test", 1, 1., isos, num_frac, True,
                             "70c", 3, [], adder.constants.IN_CORE,
                             check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        # And deplete
        for idx, dt in enumerate(delta_ts):
            mtx = test_d.compute_library(lib, mat.flux)
            new_isos, new_fracs, new_density = \
                test_d.execute(mat, mtx, lib.isotope_indices,
                               lib.inverse_isotope_indices, dt, idx,
                               num_substeps)
            mat.apply_new_composition(new_isos, new_fracs, new_density)
            np.testing.assert_allclose(mat.number_densities, ref_soln[idx, :],
                                       rtol=3e-5)

    # Deplete with 1 substep, and with 4 substeps
    deplete_substeps(1, depllib, targets, ref_soln)
    deplete_substeps(4, depllib, targets, ref_soln)
