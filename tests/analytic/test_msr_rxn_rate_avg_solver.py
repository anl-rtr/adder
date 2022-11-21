import adder
from adder.msr import MSRDepletion
import pytest
import numpy as np


# For simplicity, we will mock the mass_density attribute of Material
# to return the number density; this provides the chance for nice and clean
# answers with easily controlled times and number density initial conditions.
class MockMaterial(adder.Material):
    @property
    def mass_density(self):
        return self._density

# These tests are variants of the equivalently named test_msr_solver.py.
# Only functionality unique to the rxn_rate_avg method will be tested here.


def test_simple_feed_addition():
    # This test adds a feed to a reactor with no depletion to ensure
    # proper build-up.

    # Create a fully stable system
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))
    stable_dk = adder.DecayData(None, "s", 0.)

    # Now add stable, no xs versions of U235, U238, and Na23
    isos = ["Na23", "U235", "U238"]
    for iso in isos:
        depllib.add_isotope(iso, decay=stable_dk)
    depllib.set_isotope_indices()
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, lib, isos):
        solve_method = "rxn_rate_avg"
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that half the time is spent
        # in core (one hour in-core, one hour ex-core)
        # Set the feed to be 20% U235, 20% U238, 60% Na23

        feed = {"feed_vector": [{"U235": 0.2, "U238": 0.2, "Na23": 0.6}],
                "vector_units": "ao", "feed_rate": [1.E21],
                "feed_rate_units": "atoms/sec", "density": [1.]}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": feed,
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 3.6, "density": 1.,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Build the starting material
        num_frac = [0.5, 0.5, 0.]
        initial_isos = [(iso, "70c", True) for iso in isos]
        mat = MockMaterial("test", 1, 1., initial_isos, num_frac, True,
                           "70c", 3, [], adder.constants.IN_CORE,
                           check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19
        mat.volume = 1.

        # Now initialize the msr data
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        feed_vector = np.array([6.E20, 2.E20, 2.E20])

        ref_soln = mat.number_densities.copy()

        # Lets make sure the feed vectors match
        np.testing.assert_array_equal(test_d.systems[0].feed_vector,
                                      [feed_vector])

        # Get our volumes and volume ratios for computing the new feed
        v_sys = sys_data["component_1"]["volume"] + \
            sys_data["component_2"]["volume"]
        v_ratio = sys_data["component_1"]["volume"] / v_sys
        # Get the volumetric feed rate
        mass = np.dot(feed_vector, lib.atomic_mass_vector) / \
            adder.constants.AVOGADRO
        # feed["density"] is in units of g/cc and mass in units of g
        feed_volume_rate = mass / feed["density"]
        # The time fed to the control volume is simply the time added
        # to this volume, hence we need to include how much time this
        # volume was being fed to
        feed_duration = 86400. * v_ratio / num_substeps
        # The total feed volume added is thus feed_volume * feed_duration
        added_feed_volume = feed_volume_rate * feed_duration

        # Get the component and total volume for normalization
        component_volume = sys_data["component_1"]["volume"] * 1.e6
        tot_volume = component_volume + added_feed_volume

        # And deplete
        for t, time in enumerate(delta_ts):
            # Get the new number densities by computing the total
            # number of atoms (atoms/cc * cc) existing in the fluid and
            # the total that was added.
            for _ in range(num_substeps):
                ref_soln = (ref_soln * component_volume +
                            feed_vector * feed_duration) / tot_volume

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, time, t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_soln,
                                       rtol=2.5e-13)

    deplete_substeps(12, 16, depllib, isos)
    deplete_substeps(12, 48, depllib, isos)


def test_simple_feed_addition_w_tank():
    # This test adds a feed to a reactor with no depletion to ensure
    # proper build-up. This test has a tank, unlike simple_feed_addition

    # Create a fully stable system
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))
    stable_dk = adder.DecayData(None, "s", 0.)

    # Now add stable, no xs versions of U235, U238, and Na23
    isos = ["Na23", "U235", "U238"]
    for iso in isos:
        depllib.add_isotope(iso, decay=stable_dk)
    depllib.set_isotope_indices()
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, lib, isos):
        solve_method = "rxn_rate_avg"
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that half the time is spent
        # in core (one hour in-core, one hour ex-core)
        # Set the feed to be 20% U235, 20% U238, 60% Na23

        feed = {"feed_vector": [{"U235": 0.2, "U238": 0.2, "Na23": 0.6}],
                "vector_units": "ao", "feed_rate": [1.E21],
                "feed_rate_units": "atoms/sec", "density": [1.]}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": feed,
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "tank", "name": "c2",
                                    "volume": 3.6, "density": 1.,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Build the starting material
        num_frac = [0.5, 0.5, 0.]
        initial_isos = [(iso, "70c", True) for iso in isos]
        mat = MockMaterial("test", 1, 1., initial_isos, num_frac, True,
                           "70c", 3, [], adder.constants.IN_CORE,
                           check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19
        mat.volume = 1.

        # Now initialize the msr data
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        feed_vector = np.array([6.E20, 2.E20, 2.E20])

        ref_soln = mat.number_densities.copy()

        # Lets make sure the feed vectors match
        np.testing.assert_array_equal(test_d.systems[0].feed_vector,
                                      [feed_vector])

        # We will use this to test:
        # 1) tank delta_t
        # 2) tank volume
        # 3) feed vector in time

        # To determine all we need to compute the feed volume added to
        # the system
        # This is simply the mass feed rate * the feed density * time
        feed_mass_rate = np.dot(feed_vector, lib.atomic_mass_vector) / \
            adder.constants.AVOGADRO
        feed_volume_rate = feed_mass_rate / feed["density"]
        # feed_volume_rate is in units of [m^3/s]

        # Set the initial tank volume [m^3]
        V_tank = sys_data["component_2"]["volume"] * 1.e6
        # Set the initial duration in the tank
        delta_t_tank = 3600.

        # Set the rx volume and other constants we need
        V_rx = sys_data["component_1"]["volume"] * 1.e6
        m_dot_kg = sys_data["flowrate"]
        rho_tank_kg = sys_data["component_2"]["density"] * 1000.
        sys = test_d.systems[0]
        tank = sys.component_network[sys.starting_component][0]

        # And deplete
        for t, time in enumerate(delta_ts):
            # Get the new number densities by computing the total
            # number of atoms (atoms/cc * cc) existing in the fluid and
            # the total that was added.
            for ns in range(num_substeps):
                # The volume of the tank changes and so the fraction of the
                # the spent in the tank vs in core changes, so lets update
                # the parameters used to compute the feed duration and the
                # feed volume
                feed_duration = 86400. * \
                    (V_rx / (V_tank + V_rx)) / num_substeps
                feed_volume = feed_volume_rate * feed_duration
                V_t = V_rx + feed_volume

                ref_soln = \
                    (ref_soln * V_rx + feed_vector * feed_duration) / V_t

                # Now we can compute the new values for the next time through
                # The volume added to the tank is the feed volume added to
                # the whole system
                V_tank += feed_volume_rate * 86400. / num_substeps
                delta_t_tank = V_tank / 1.e6 * rho_tank_kg / m_dot_kg

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, time, t, num_substeps, 0.)

            # Now compare the tested values
            np.testing.assert_allclose(mat.number_densities, ref_soln,
                                       rtol=2.5e-13)
            refs = np.array([delta_t_tank, V_tank / 1.e6])
            tests = np.array([tank.delta_t, tank.volume]).reshape((2, 1))
            np.testing.assert_allclose(tests, refs, rtol=1.e-15)

    deplete_substeps(2, 16, depllib, isos)
    deplete_substeps(2, 48, depllib, isos)


def test_stationary():
    # Simple test of the MSR solver with no fluid flow
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    src = "Pu238"
    targets = ["Pu239", "Np238", "U235", "Pu237", "Np237", "Np236"]
    types = ["(n,gamma)", "(n,p)", "(n,a)", "(n,2n)", "(n,d)", "(n,t)"]
    # The starting nuclide, Pu238, doesnt have decay, and neither does
    # the final, Np236. Indices of half_lives correspond to targets
    half_lives = [np.log(2) / (1.E-8), np.log(2) / (1.E-7),
                  np.log(2) / (3.E-7), np.log(2) / (6.E-7),
                  np.log(2) / (9.E-6), None]
    xss = 0.1 * np.ones(3)
    xs = adder.ReactionData("b", 3)
    for i in range(len(targets)):
        xs.add_type(types[i], "b", xss, targets=targets[i])
        target_dk = adder.DecayData(half_lives[i], "s", 0.)
        target_dk.add_type("beta-", 1., ["B10"])
        depllib.add_isotope(targets[i], decay=target_dk)
    depllib.add_isotope(src, xs=xs)
    # Add in B10 as stable
    depllib.add_isotope("B10", decay=adder.DecayData(None, "s", 0.))
    # And add in the secondary products
    secondaries = ["H1", "H2", "H3", "He4"]
    for secondary in secondaries:
        target_dk = adder.DecayData(None, "s", 0.)
        depllib.add_isotope(secondary, decay=target_dk)
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss):
        solve_method = "rxn_rate_avg"
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information where there is no time spent ex-core
        # and there is no feed
        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Build the starting material, density chosen to start with
        # a number density of 1
        isos = ["Pu238"] + targets + ["B10"]
        num_frac = []
        for iso in isos:
            if iso not in ["B10", "Np236"]:
                num_frac.append(1.)
            else:
                num_frac.append(0.)
        isos = [(iso, "70c", True) for iso in isos]
        mat = MockMaterial("test", 1, 1., isos, num_frac, True,
                           "70c", 3, [], adder.constants.IN_CORE, check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19
        mat.volume = 1.

        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = [3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8]
        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu238
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)
        for t, dt in enumerate(delta_ts):
            # Compute the reference number densities
            exp_flux = \
                np.exp(-6. * phi_xs * dt * 86400.)
            for i, lam in zip(i_targets, lambdas):
                exp_dk = np.exp(-lam * dt * 86400.)

                # Accrue this component into B-10
                ref_Ns[iB10] += (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                                 (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                    phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                    (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                # Now modify isotope i
                ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                    (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                if i in secondary_map:
                    ref_Ns[secondary_map[i]] += (1. / 6.) * ref_Ns[i_src] * \
                        (1. - exp_flux)
            # And end with the starting nuclide
            ref_Ns[i_src] *= exp_flux

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, dt, t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=2.4e-13)

    # Deplete with 16th order and 48th order
    deplete_substeps(2, 16, depllib, targets, xss)
    deplete_substeps(2, 48, depllib, targets, xss)


def test_flowing():
    # Test the solver with a simple three-component flow.
    # 2 of the components will have unique xs and fluxes. The third will have
    # a removal constant.

    # The base depletion library
    depllib = adder.DepletionLibrary("base", np.array([0., 0.01, 1., 20.]))
    xss = [0.1, 0.2, 0.3]
    xs = adder.ReactionData("b", 3)
    xs.add_type
    xs.add_type("(n,gamma)", "b", xss, targets="Pu239")
    am239_dk = adder.DecayData(360. * 12., "s", 0.)  # 1.2 hr half life
    am239_dk.add_type("ec/beta+", 1., ["Pu239"])
    depllib.add_isotope("Am239", decay=am239_dk)
    depllib.add_isotope("Pu238", xs=xs)
    depllib.add_isotope("Pu239", decay=adder.DecayData(None, "s", 0.))
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, base_lib, xs_1):
        solve_method = "rxn_rate_avg"
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: base_lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that half the time is spent
        # in core (one hour in-core, one hour ex-core)
        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 1.2, "mat_name": "test_1",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "in-core", "name": "c2",
                                    "volume": 2.4, "mat_name": "test_2",
                                    "removal_vector": {},
                                    "downstream_components": ["c3"],
                                    "downstream_mass_fractions": [1.]},
                    "component_3": {"type": "generic", "name": "c3",
                                    "volume": 3.6, "density": 1.,
                                    "removal_vector":
                                        {"Am239": np.log(2) / (10. * 86400.)},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Build the starting material, even mix of Pu238 and Am239
        isos = [("Pu238", "70c", True), ("Am239", "70c", True)]
        num_frac = [0.5, 0.5]
        mat1 = MockMaterial("test_1", 1, 1., isos, num_frac, True,
                            "70c", 3, [], adder.constants.IN_CORE, check=False)
        mat1.is_default_depletion_library = True
        mat1.depl_lib_name = 0
        mat1.flux = np.array([0.2, 0.3, 0.5]) * 1.E+19
        mat1.volume = 1.

        mat2 = MockMaterial("test_2", 1, 1., isos, num_frac, True,
                            "70c", 3, [], adder.constants.IN_CORE, check=False)
        mat2_lib = depl_libs[0].clone(new_name="mat2")
        n_xs = mat2_lib.isotopes["Pu238"].neutron_xs
        _, t, y, q = n_xs._products["(n,gamma)"]
        xs_2 = np.array([0.2, 0.2, 0.1])
        n_xs._products["(n,gamma)"] = (xs_2, t, y, q)
        mat2.is_default_depletion_library = False
        mat2.depl_lib_name = mat2_lib.name
        mat2.flux = np.ones(3) * 1.E19
        mat1.volume = 1.
        depl_libs[mat2_lib.name] = mat2_lib

        test_d.set_msr_params("histogram", solve_method, [sys_data],
                              [mat1, mat2], depl_libs)

        # Set up our solution space
        i238 = 0
        i239 = 1
        iAm = 2
        ref_Ns = [5E23, 0., 5E23]
        # The time-averaged rxn rates are based on 20 min in component 1,
        # 40 min in component 2, 60 min in component 3 (units dont matter as
        # they are normalized out) (60 * 0 is for the 0 flux ex-core)
        avg_rr = (20. * np.dot(mat1.flux, xs_1) +
                  40. * np.dot(mat2.flux, xs_2) + 60. * 0.) / 120. * 1E-24

        avg_removal_lambda = 60. / 120. * \
            sys_data["component_3"]["removal_vector"]["Am239"]
        lam = np.log(2.) / (12. * 360.) + avg_removal_lambda
        lambda_fraction = (lam - avg_removal_lambda) / lam

        # Now we have a simple three-isotope system:
        # Pu238 -> (n,gamma) -> Pu239 -> beta- + removal -> Am239
        # where the xs and decay constants are all averaged as above

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)
        # And deplete
        for t, time in enumerate(delta_ts):
            # Compute the reference number densities
            dt = time * 86400. / num_substeps
            for _ in range(num_substeps):
                exp_flux = np.exp(-avg_rr * dt)
                exp_dk = np.exp(-lam * dt)

                ref_Ns[i239] += ref_Ns[i238] * (1. - exp_flux) + \
                    lambda_fraction * ref_Ns[iAm] * (1. - exp_dk)
                ref_Ns[i238] *= exp_flux
                ref_Ns[iAm] *= exp_dk

            # Deplete in our solver to be tested
            test_d.execute([mat1, mat2], depl_libs, time, t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            if cram_order == 16:
                tol = 7e-10
            elif cram_order == 48:
                tol = 2.2e-14
            np.testing.assert_allclose(mat1.number_densities, ref_Ns, rtol=tol)
            np.testing.assert_allclose(mat2.number_densities, ref_Ns, rtol=tol)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, xss)
    deplete_substeps(2, 48, depllib, xss)


def test_multipath():
    # Test the solver with a four-component flow.
    # 2 of the components will have unique xs and fluxes. The third and fourth
    # will have removal constants but be in parallel to each other

    # The base depletion library
    depllib = adder.DepletionLibrary("base", np.array([0., 0.01, 1., 20.]))
    xss = [0.1, 0.2, 0.3]
    xs = adder.ReactionData("b", 3)
    xs.add_type
    xs.add_type("(n,gamma)", "b", xss, targets="Pu239")
    am239_dk = adder.DecayData(360. * 12., "s", 0.)  # 1.2 hr half life
    am239_dk.add_type("ec/beta+", 1., ["Pu239"])
    depllib.add_isotope("Am239", decay=am239_dk)
    depllib.add_isotope("Pu238", xs=xs)
    depllib.add_isotope("Pu239", decay=adder.DecayData(None, "s", 0.))
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, base_lib, xs_1):
        solve_method = "rxn_rate_avg"
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: base_lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that in the first path,
        # half the time is spent in core (one hour in-core, one hour ex-core)
        # and in the second path, 1 hr in core, 2 hrs ex-core
        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 1.2, "mat_name": "test_1",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "in-core", "name": "c2",
                                    "volume": 2.4, "mat_name": "test_2",
                                    "removal_vector": {},
                                    "downstream_components": ["c3", "c4"],
                                    "downstream_mass_fractions": [0.2, 0.8]},
                    "component_3": {"type": "generic", "name": "c3",
                                    "volume": 0.2 * 3.6, "density": 1.,
                                    "removal_vector":
                                        {"Am239": np.log(2) / (10. * 86400.)},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]},
                    "component_4": {"type": "generic", "name": "c4",
                                    "volume": 1.6 * 3.6, "density": 1.,
                                    "removal_vector":
                                        {"Am239": np.log(2) / (5. * 86400.)},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Build the starting material, even mix of Pu238 and Am239
        isos = [("Pu238", "70c", True), ("Am239", "70c", True)]
        num_frac = [0.5, 0.5]
        mat1 = MockMaterial("test_1", 1, 1., isos, num_frac, True,
                            "70c", 3, [], adder.constants.IN_CORE, check=False)
        mat1.is_default_depletion_library = True
        mat1.depl_lib_name = 0
        mat1.flux = np.array([0.2, 0.3, 0.5]) * 1.E+19
        mat1.volume = 1.

        mat2 = MockMaterial("test_2", 1, 1., isos, num_frac, True,
                            "70c", 3, [], adder.constants.IN_CORE, check=False)
        mat2_lib = base_lib.clone(new_name="mat2")
        n_xs = mat2_lib.isotopes["Pu238"].neutron_xs
        _, t, y, q = n_xs._products["(n,gamma)"]
        xs_2 = np.array([0.2, 0.2, 0.1])
        n_xs._products["(n,gamma)"] = (xs_2, t, y, q)
        mat2.is_default_depletion_library = False
        mat2.flux = np.ones(3) * 1.E19
        mat2.depl_lib_name = mat2_lib.name
        mat2.volume = 1.
        depl_libs[mat2_lib.name] = mat2_lib

        test_d.set_msr_params("histogram", solve_method, [sys_data],
                              [mat1, mat2], depl_libs)

        # Set up our solution space
        i238 = 0
        i239 = 1
        iAm = 2
        ref_Ns = [5E23, 0., 5E23]
        # The time-averaged rxn rates are based on 20 min in component 1,
        # 40 min in component 2, x min in ex-core components
        # (units dont matter as they are normalized out)
        # (60 * 0 is for the 0 flux ex-core)
        avg_rr_1 = (20. * np.dot(mat1.flux, xs_1) +
                    40. * np.dot(mat2.flux, xs_2) + 60. * 0.) / 120. * 1E-24
        avg_rr_2 = (20. * np.dot(mat1.flux, xs_1) +
                    40. * np.dot(mat2.flux, xs_2) + 120. * 0.) / 180. * 1E-24

        # Path 1 lambda values
        avg_removal_lambda_1 = 60. / 120. * \
            sys_data["component_3"]["removal_vector"]["Am239"]
        lam_1 = np.log(2.) / (12. * 360.) + avg_removal_lambda_1
        lambda_fraction_1 = (lam_1 - avg_removal_lambda_1) / lam_1
        # flow_1 = 0.2
        flow_1 = 0.14285714
        # Apply component-vol*mass flow weighting to the path weights
        flow_1 = (0.2 / 1.0 * 1.2) + (0.2 / 1.0 * 2.4) + (0.2 / 0.2 * 0.2 * 3.6)
        flow_2 = (0.8 / 1.0 * 1.2) + (0.8 / 1.0 * 2.4) + (0.8 / 0.8 * 1.6 * 3.6)
        tot = flow_1 + flow_2
        flow_1 /= tot
        flow_2 /= tot

        # Path 2 lambda values
        avg_removal_lambda_2 = 120. / 180. * \
            sys_data["component_4"]["removal_vector"]["Am239"]
        lam_2 = np.log(2.) / (12. * 360.) + avg_removal_lambda_2
        lambda_fraction_2 = (lam_2 - avg_removal_lambda_2) / lam_2

        # Now we have a simple three-isotope system:
        # Pu238 -> (n,gamma) -> Pu239 -> beta- + removal -> Am239
        # where the xs and decay constants are all averaged as above

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)
        # And deplete
        for t, time in enumerate(delta_ts):
            # Compute the reference number densities
            dt = time * 86400. / num_substeps
            exp_flux_1 = np.exp(-avg_rr_1 * dt)
            exp_flux_2 = np.exp(-avg_rr_2 * dt)
            exp_dk_1 = np.exp(-lam_1 * dt)
            exp_dk_2 = np.exp(-lam_2 * dt)

            for _ in range(num_substeps):
                # Account for paths 1 and 2
                ref_Ns[i239] += \
                    flow_1 * (ref_Ns[i238] * (1. - exp_flux_1) +
                              lambda_fraction_1 * ref_Ns[iAm] * (1. - exp_dk_1))
                ref_Ns[i239] += \
                    flow_2 * (ref_Ns[i238] * (1. - exp_flux_2) +
                              lambda_fraction_2 * ref_Ns[iAm] * (1. - exp_dk_2))
                ref_Ns[i238] *= (flow_1 * exp_flux_1 + flow_2 * exp_flux_2)
                ref_Ns[iAm] *= (flow_1 * exp_dk_1 + flow_2 * exp_dk_2)

            # Deplete in our solver to be tested
            test_d.execute([mat1, mat2], depl_libs, time, t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            if cram_order == 16:
                tol = 7e-10
            elif cram_order == 48:
                tol = 2e-14
            np.testing.assert_allclose(mat1.number_densities, ref_Ns, rtol=tol)
            np.testing.assert_allclose(mat2.number_densities, ref_Ns, rtol=tol)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, xss)
    deplete_substeps(2, 48, depllib, xss)
