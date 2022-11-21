import adder
from adder.msr import MSRDepletion
import pytest
import numpy as np
import adder.constants as constants
from adder.data import atomic_mass

# For simplicity, we will mock the mass_density attribute of Material
# to return the number density; this provides the chance for nice and clean
# answers with easily controlled times and number density initial conditions.


class MockMaterial(adder.Material):
    @property
    def mass_density(self):
        return self._density

# These tests are variants of the simple_burn_and_decay problem
# from the CRAM verification. The differences will relate to
# in-core vs ex-core time, purification parameters, removal, and feed


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

    def deplete_substeps(num_substeps, cram_order, lib, isos, solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Build the starting material
        num_frac = [0.5, 0.5, 0.]
        initial_isos = [(iso, "70c", True) for iso in isos]
        mat = MockMaterial("test", 1, 1., initial_isos, num_frac, True, "70c",
                           3, [], adder.constants.IN_CORE, check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.2, 0.3, 0.5]) * 1.407416667E+19
        mat.volume = 1.

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
        feed_volume_rate = mass / feed["density"][0]
        # The time fed to the control volume is simply the time added
        # to this volume, hence we need to include how much time this
        # volume was being fed to
        feed_duration = 86400. * v_ratio / 12.
        # The total feed volume added is thus
        # feed_volume * feed_duration
        added_feed_volume = feed_volume_rate * feed_duration

        # Get the component and total volume for normalization
        component_volume = sys_data["component_1"]["volume"] * 1.e6
        tot_volume = component_volume + added_feed_volume

        # And deplete
        for t in range(len(delta_ts)):
            # Get the new number densities by computing the total
            # number of atoms (atoms/cc * cc) existing in the fluid and
            # the total that was added.
            for i in range(12):
                ref_soln = (ref_soln * component_volume +
                            feed_vector * feed_duration) / tot_volume

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[t], t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_soln,
                                       rtol=2.5e-13)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, isos, "brute")
    deplete_substeps(2, 16, depllib, isos, "tmatrix")
    deplete_substeps(2, 16, depllib, isos, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, isos, "brute")
    deplete_substeps(2, 48, depllib, isos, "tmatrix")


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

    def deplete_substeps(num_substeps, cram_order, lib, isos, solve_method):
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
                           "70c", 3, [], adder.constants.IN_CORE, check=False)
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
        for t in range(len(delta_ts)):
            # Get the new number densities by computing the total
            # number of atoms (atoms/cc * cc) existing in the fluid and
            # the total that was added.
            # The 0.5 is present since half of the time was spent
            # injecting this feed.
            feed_duration = 86400. * 0.5 / 12.
            feed_volume = feed_volume_rate * feed_duration
            V_t = feed_volume + V_rx
            for i in range(12):
                ref_soln = (ref_soln * V_rx +
                            feed_vector * feed_duration) / V_t

            # Now we can compute the new values we want to compare
            # The volume added to the tank is the feed volume added to
            # the whole system
            V_tank += feed_volume_rate * 86400.
            delta_t_tank = V_tank / 1.e6 * rho_tank_kg / m_dot_kg

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[t], t, num_substeps, 0.)

            # Now compare the tested values
            np.testing.assert_allclose(mat.number_densities, ref_soln,
                                       rtol=2.5e-13)
            refs = np.array([delta_t_tank, V_tank / 1.e6])
            tests = np.array([tank.delta_t, tank.volume]).reshape((2, 1))
            np.testing.assert_allclose(tests, refs, rtol=1.e-15)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, isos, "brute")
    deplete_substeps(2, 16, depllib, isos, "tmatrix")
    deplete_substeps(2, 16, depllib, isos, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, isos, "brute")
    deplete_substeps(2, 48, depllib, isos, "tmatrix")


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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
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
        for t in range(len(delta_ts)):
            # Compute the reference number densities
            exp_flux = \
                np.exp(-6. * phi_xs * delta_ts[t] * 86400.)
            for i, lam in zip(i_targets, lambdas):
                exp_dk = np.exp(-lam * delta_ts[t] * 86400.)

                # Accrue this component into B-10
                ref_Ns[iB10] += \
                    (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
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
            test_d.execute([mat], depl_libs, delta_ts[t], t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=2.4e-13)

    # Deplete with 16th order and 48th order brute
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_flowing():
    # Test the MSR solver with the fluid flowing through the primary
    # (i.e., no CP system)
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that half the time is spent
        # in core (one hour in-core, one hour ex-core)
        # Init our MSR information where there is no time spent ex-core
        # and there is no feed
        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
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
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)
        for time in range(len(delta_ts)):
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            dt = 1. / 24.
            exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) *
                              dt * 86400.)
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now ex-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # Accrue this component into B-10
                    ref_Ns[iB10] += ref_Ns[i] * (1. - exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] *= exp_dk

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=2.5e-13)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_flowing_with_feed():
    # Same as test_flowing, except here we use the feed vector as well
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Init our MSR information so that half the time is spent
        # in core (one hour in-core, one hour ex-core)
        # Set the feed to be U235 and elemental Pu238 (which should just
        # add the Pu isotopes in the library: Pu237-239), and
        # elemental H (H1-H3, though only H1 and H2 added since H3 is
        # not natural)
        # Note we are adding enough to see it in the answer

        feed = {"feed_vector": [{"U235": 1.E20, "Pu": 2.E20, "H": 3.E20}],
                "vector_units": "ao", "feed_rate": [6.E20],
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

        # Initialize with this feed; we expect a ValueError exception
        # because of the presence of Pu
        with pytest.raises(ValueError):
            test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
                depl_libs)

        # So now we re-do it with a corrected feed
        feed = {"feed_vector": [{"U235": 1.E20, "Pu239": 2.E20, "H": 3.E20}],
                "vector_units": "ao", "feed_rate": [6.E20],
                "feed_rate_units": "atoms/sec", "density": [1.]}
        sys_data["feed"] = feed

        # Now re-initialize with the corrected feed
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        feed_vector = np.zeros(len(iso_list))
        feed_vector[:2] = \
            np.array([0.99984426, 0.00015574]) * feed["feed_vector"][0]["H"]
        feed_vector[11] = feed["feed_vector"][0]["Pu239"]
        feed_vector[5] = feed["feed_vector"][0]["U235"]

        # Get our volumes and volume ratios for computing the new feed
        v_sys = sys_data["component_1"]["volume"] + \
            sys_data["component_2"]["volume"]
        v_ratio = sys_data["component_1"]["volume"] / v_sys
        # Get the volumetric feed rate
        mass = np.dot(feed_vector, lib.atomic_mass_vector) / \
            adder.constants.AVOGADRO
        # Feed["density"] is in units of g/cc, mass in g
        feed_volume = mass / feed["density"]
        # The time fed to the control volume is simply the time added
        # to this volume, hence we need to include how much time this
        # volume was being fed to
        feed_duration = 3600. * 2. * v_ratio
        # The total feed volume added is thus
        # feed_volume * feed_duration
        added_feed_volume = feed_volume * feed_duration

        # Get the component volume for normalization
        component_volume = sys_data["component_1"]["volume"] * 1.e6

        # Lets make sure the feed vectors match
        np.testing.assert_array_equal(test_d.systems[0].feed_vector,
                                      [feed_vector])

        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)
        for time in range(len(delta_ts)):
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            dt = 1. / 24.
            exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) *
                              dt * 86400.)
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now ex-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # Accrue this component into B-10
                    ref_Ns[iB10] += ref_Ns[i] * (1. - exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] *= exp_dk

                # And add in the feed
                ref_Ns = (ref_Ns * component_volume + feed_vector *
                          feed_duration) / \
                    (component_volume + added_feed_volume)

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=2.5e-13)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_fully_cp():
    # This tests the CP-path by directing all flow through the CP and
    # removing some isotopes with the purification system.
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Set the removal rates
        # We will elementally define Pu as having a removal rate
        # and also set U235 to have one
        cp_removal = {"Pu": 1.E-7, "U235": 1.E-8}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 3.6, "density": 1.,
                                    "removal_vector": cp_removal,
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

        # Init our MSR information
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        # Set the ex-core lambdas
        ex_core_lambdas = lambdas + np.array([cp_removal["U235"], 0., 0., 0.,
                                              cp_removal["Pu"],
                                              cp_removal["Pu"]])

        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)
        for time in range(len(delta_ts)):
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            dt = 1. / 24.
            exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) *
                              dt * 86400.)
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now remove isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now ex-core
                for i, lam, b10_lam in zip(i_targets, ex_core_lambdas,
                                           lambdas):
                    # exp_dk takes away the cp + decay part
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # exp_b10_dk takes away only the decay
                    exp_b10_dk = np.exp(-b10_lam * dt * 86400.)
                    # Accrue this component into B-10
                    ref_Ns[iB10] += ref_Ns[i] * (1. - exp_b10_dk)

                    # Now remove isotope i, including the CP portion
                    ref_Ns[i] *= exp_dk
                # And remove the Pu238 from CP
                ref_Ns[i_src] *= np.exp(-cp_removal["Pu"] * dt * 86400.)

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=1.8e-5)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_mix_no_delay():
    # This tests the main and CP-paths but w/o the need to store the
    # CP output for later usage
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Set the removal rates
        # We will elementally define Pu as having a removal rate
        # and also set U235 to have one
        cp_removal = {"Pu": 1.E-7, "U235": 1.E-8}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2", "c3"],
                                    "downstream_mass_fractions": [0.5, 0.5]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 1.8, "density": 1.,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]},
                    "component_3": {"type": "generic", "name": "c3",
                                    "volume": 1.8, "density": 1.,
                                    "removal_vector": cp_removal,
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

        # Init our MSR information
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        # Set the ex-core lambdas
        ex_core_lambdas = lambdas + np.array([cp_removal["U235"], 0., 0., 0.,
                                              cp_removal["Pu"],
                                              cp_removal["Pu"]])
        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)
        for time in range(len(delta_ts)):
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            dt = 1. / 24.
            exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) *
                              dt * 86400.)
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now to do the ex-core, we have main and CP
                # Main (same math as above, but no flux)
                Nmain = np.copy(ref_Ns)
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # Accrue this component into B-10
                    Nmain[iB10] += Nmain[i] * (1. - exp_dk)

                    # Now modify isotope i
                    Nmain[i] *= exp_dk

                # CP
                Ncp = np.copy(ref_Ns)
                for i, lam, b10_lam in zip(i_targets, ex_core_lambdas,
                                           lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    exp_b10_dk = np.exp(-b10_lam * dt * 86400.)
                    # Accrue this component into B-10
                    Ncp[iB10] += Ncp[i] * (1. - exp_b10_dk)

                    # Now modify isotope i
                    Ncp[i] *= exp_dk
                # And remove the Pu238 from CP
                Ncp[i_src] *= np.exp(-cp_removal["Pu"] * dt * 86400.)

                ref_Ns = 0.5 * (Nmain + Ncp)

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=5E-5)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_mix_delay():
    # This tests the main and CP-paths where the CP path holds on to
    # the fluid for 2 loop transports.
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Set the removal rates
        # We will elementally define Pu as having a removal rate
        # and also set U235 to have one
        cp_removal = {"Pu": 1.E-7, "U235": 1.E-8}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": {},
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2", "c3"],
                                    "downstream_mass_fractions": [0.5, 0.5]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 1.8, "density": 1.,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]},
                    "component_3": {"type": "generic", "name": "c3",
                                    "volume": 5.4, "density": 1.,
                                    "removal_vector": cp_removal,
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

        # Init our MSR information
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        # Set the ex-core lambdas
        ex_core_lambdas = lambdas + np.array([cp_removal["U235"], 0., 0., 0.,
                                              cp_removal["Pu"],
                                              cp_removal["Pu"]])
        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)

        Ncp_store = np.zeros((2 + 12, ref_Ns.shape[0]))
        for t in range(Ncp_store.shape[0]):
            Ncp_store[t, :] = ref_Ns[:]

        dt = 1. / 24.
        cp_dt = 3. / 24.
        exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) * dt * 86400.)
        for time in range(len(delta_ts)):
            # Adjust the Ncp_store data to capture the history from the
            # last time
            Ncp_store[:2] = Ncp_store[12:]
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now to do the ex-core, we have main and CP
                # Main (same math as above, but no flux)
                Nmain = np.copy(ref_Ns)
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # Accrue this component into B-10
                    Nmain[iB10] += Nmain[i] * (1. - exp_dk)

                    # Now modify isotope i
                    Nmain[i] *= exp_dk

                # CP
                Ncp = np.copy(ref_Ns)
                for i, lam, b10_lam in zip(i_targets, ex_core_lambdas,
                                           lambdas):
                    exp_dk = np.exp(-lam * cp_dt * 86400.)
                    exp_b10_dk = np.exp(-b10_lam * cp_dt * 86400.)
                    # Accrue this component into B-10
                    Ncp[iB10] += Ncp[i] * (1. - exp_b10_dk)

                    # Now modify isotope i
                    Ncp[i] = Ncp[i] * exp_dk
                # And remove the Pu238 from CP
                Ncp[i_src] = Ncp[i_src] * np.exp(-cp_removal["Pu"] * cp_dt *
                                                 86400.)
                Ncp_store[t + 1] = Ncp[:]

                ref_Ns = 0.5 * (Nmain + Ncp_store[t])

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=6e-5)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_mix_delay_with_feed():
    # This tests the main and CP-paths where the CP path holds on to
    # the fluid for 2 loop transports and with an additional feed
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

    def deplete_substeps(num_substeps, cram_order, lib, targets, xss,
                         solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Set the removal rates
        # We will elementally define Pu as having a removal rate
        # and also set U235 to have one
        cp_removal = {"Pu": 1.E-7, "U235": 1.E-8}

        # Set the feed; the same feed as used in test_flowing_with_feed
        feed = {"feed_vector": [{"U235": 1.E20, "Pu239": 2.E20, "H": 3.E20}],
                "vector_units": "ao", "feed_rate": [6.E20],
                "feed_rate_units": "atoms/sec", "density": [1.]}

        sys_data = {"name": "sys1", "flowrate": 1., "flow_start": "c1",
                    "feed": feed,
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2", "c3"],
                                    "downstream_mass_fractions": [0.5, 0.5]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 1.8, "density": 1.,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]},
                    "component_3": {"type": "generic", "name": "c3",
                                    "volume": 5.4, "density": 1.,
                                    "removal_vector": cp_removal,
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

        # Init our MSR information
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our decay times; will look at t = [0, 1, 2, 3, 4, 5] days
        delta_ts = np.ones(6)

        iso_list = ["H1", "H2", "H3", "He4", "B10", "U235", "Np236", "Np237",
                    "Np238", "Pu237", "Pu238", "Pu239"]
        i_src = 10
        i_targets = [5, 6, 7, 8, 9, 11]
        secondary_map = {9: 0, 5: 3, 7: 1, 8: 2}
        lambdas = np.array([3E-7, 0., 9E-6, 1E-7, 6E-7, 1E-8])
        # Set the ex-core lambdas
        ex_core_lambdas = lambdas + np.array([cp_removal["U235"], 0., 0., 0.,
                                              cp_removal["Pu"],
                                              cp_removal["Pu"]])
        # Set the initial conditions
        ref_Ns = np.zeros(len(iso_list))
        # all targets start at 1.
        ref_Ns[i_targets] = 1.
        iB10 = 4
        iNp236 = 6
        # except for Np236 and B10
        ref_Ns[[iNp236, iB10]] = 0.
        # And add in the starting nuclide, Pu239
        ref_Ns[i_src] = 1.
        ref_Ns *= 1.E24 / 6.
        # And deplete
        phi_xs = np.dot(xss * 1.E-24, mat.flux)

        Ncp_store = np.zeros((2 + 12, ref_Ns.shape[0]))
        for t in range(Ncp_store.shape[0]):
            Ncp_store[t, :] = ref_Ns[:]

        feed_vector = np.zeros(len(iso_list))
        feed_vector[:2] = \
            np.array([0.99984426, 0.00015574]) * feed["feed_vector"][0]["H"]
        feed_vector[11] = feed["feed_vector"][0]["Pu239"]
        feed_vector[5] = feed["feed_vector"][0]["U235"]

        # Get the volumetric feed rate
        mass = np.dot(feed_vector, lib.atomic_mass_vector) / \
            adder.constants.AVOGADRO
        # feed["density"] is in units of g/cc, mass in g
        feed_volume = mass / feed["density"]
        # The time fed to the control volume is simply the time added
        # to this volume, hence we need to include how much time this
        # volume was being fed to
        feed_duration = 3600.
        # The total feed volume added is the feed_volume * feed_duration
        added_feed_volume = feed_volume * feed_duration

        # Get the component volume for normalization
        component_volume = sys_data["component_1"]["volume"] * 1.e6

        # Lets make sure the feed vectors match
        np.testing.assert_array_equal(test_d.systems[0].feed_vector[0][:12],
                                      feed_vector)

        dt = 1. / 24.
        cp_dt = 3. / 24.
        exp_flux = np.exp(-6. * np.dot(xss * 1.E-24, mat.flux) * dt * 86400.)
        for time in range(len(delta_ts)):
            # Adjust the Ncp_store data to capture the history from the
            # last time
            Ncp_store[:2] = Ncp_store[12:]
            # Compute the reference number densities
            # Since each time step is 1 d, there 12 in-core/ex-core
            # durations
            for t in range(12):
                # First in-core
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)

                    # Accrue this component into B-10
                    ref_Ns[iB10] += \
                        (ref_Ns[i] + phi_xs * ref_Ns[i_src] /
                         (lam - 6. * phi_xs) * (lam / (6. * phi_xs) - 1.))
                    ref_Ns[iB10] += -ref_Ns[i] * exp_dk + \
                        phi_xs * ref_Ns[i_src] / (lam - 6. * phi_xs) * \
                        (-lam / (6. * phi_xs) * exp_flux + exp_dk)

                    # Now modify isotope i
                    ref_Ns[i] = ref_Ns[i] * exp_dk + phi_xs * ref_Ns[i_src] / \
                        (lam - 6. * phi_xs) * (exp_flux - exp_dk)
                    if i in secondary_map:
                        ref_Ns[secondary_map[i]] += \
                            (1. / 6.) * ref_Ns[i_src] * (1. - exp_flux)
                # And end with the starting nuclide
                ref_Ns[i_src] *= exp_flux

                # Now to do the ex-core, we have main and CP
                # Main (same math as above, but no flux)
                Nmain = np.copy(ref_Ns)
                for i, lam in zip(i_targets, lambdas):
                    exp_dk = np.exp(-lam * dt * 86400.)
                    # Accrue this component into B-10
                    Nmain[iB10] += Nmain[i] * (1. - exp_dk)

                    # Now modify isotope i
                    Nmain[i] *= exp_dk

                # CP
                Ncp = np.copy(ref_Ns)
                for i, lam, lam_b10 in zip(i_targets, ex_core_lambdas,
                                           lambdas):
                    exp_dk_b10 = np.exp(-lam_b10 * cp_dt * 86400.)
                    exp_dk = np.exp(-lam * cp_dt * 86400.)
                    # Accrue this component into B-10
                    Ncp[iB10] += Ncp[i] * (1. - exp_dk_b10)

                    # Now modify isotope i
                    Ncp[i] = Ncp[i] * exp_dk
                # And remove the Pu238 from CP
                Ncp[i_src] = Ncp[i_src] * np.exp(-cp_removal["Pu"] * cp_dt *
                                                 86400.)
                Ncp_store[t + 1] = Ncp[:]

                ref_Ns = 0.5 * (Nmain + Ncp_store[t])

                # And add in the feed
                ref_Ns = (ref_Ns * component_volume +
                          feed_vector * feed_duration) / \
                    (component_volume + added_feed_volume)

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[time], time,
                num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_Ns,
                                       rtol=6e-5)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(2, 16, depllib, targets, xss, "brute")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix")
    deplete_substeps(2, 16, depllib, targets, xss, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(2, 48, depllib, targets, xss, "brute")
    deplete_substeps(2, 48, depllib, targets, xss, "tmatrix")


def test_simple_variable_feed_addition():
    # This test adds a variable feed to a reactor with no depletion
    # to ensure proper build-up.

    # Create a fully stable system
    depllib = adder.DepletionLibrary("test", np.array([0., 20.]))
    stable_dk = adder.DecayData(None, "s", 0.)

    # Now add stable, no xs versions of U235, and U238
    isos = ["U235", "U238"]
    for iso in isos:
        depllib.add_isotope(iso, decay=stable_dk)
    depllib.set_isotope_indices()
    depllib.set_atomic_mass_vector()

    def deplete_substeps(num_substeps, cram_order, lib, isos, solve_method):
        exec_cmd = ""
        num_threads = 200
        num_procs = 200
        depl_libs = {0: lib}
        # Initialize our depletion solver
        test_d = MSRDepletion(exec_cmd, num_threads, num_procs, 1, cram_order)

        # Build the starting material
        dens = 1200.  # g/cc
        # convert density from g/cc to atom/b-cm
        input_mass_density = dens * constants.AVOGADRO / atomic_mass('U235')
        num_frac = [1., 0.]
        initial_isos = [(iso, "70c", True) for iso in isos]
        mat = adder.Material("test", 1, input_mass_density * 1e-24,
                             initial_isos, num_frac, True, "70c",
                             1, [], adder.constants.IN_CORE, check=False)
        mat.is_default_depletion_library = True
        mat.flux = np.array([0.])
        mat.volume = 1.

        # Init our MSR information so that half the time is spent
        # in core (120 hour in-core, 120 hour ex-core)
        # Set the feed to be dep step 1: 50% U235, 50% U238
        # dep step 2: 100% U238
        # dep step 3: 25% U235, 75% U238
        # the corresponding adder input looks like:
        #     feed_material = f2, f1, (f1 f2)
        #     feed_mixture = 1, 1, (1 1)
        #     [[[[material_f1]]]]
        #         names = U238
        #         vector = 100
        #     [[[[material_f2]]]]
        #         names = U238, U235
        #         vector = 0.5, 0.5
        feed = {"feed_vector": [{'U238': 0.5, 'U235': 0.5}, {'U238': 1.0},
                                {'U238': 1.5, 'U235': 0.5}],
                "vector_units": "ao", "feed_rate": [1e+22, 1e+22, 1e+22],
                "feed_rate_units": "atoms/sec", "density": [10., 10., 10.]}

        sys_data = {"name": "sys1", "flowrate": 10., "flow_start": "c1",
                    "feed": feed,
                    "component_1": {"type": "in-core", "name": "c1",
                                    "volume": 3.6, "mat_name": "test",
                                    "removal_vector": {},
                                    "downstream_components": ["c2"],
                                    "downstream_mass_fractions": [1.]},
                    "component_2": {"type": "generic", "name": "c2",
                                    "volume": 3.6, "density": dens,
                                    "removal_vector": {},
                                    "downstream_components": ["c1"],
                                    "downstream_mass_fractions": [1.]}}

        # Now initialize the msr data
        test_d.set_msr_params("histogram", solve_method, [sys_data], [mat],
            depl_libs)

        # Set our times; will look at t = [50, 100, 150] days
        one_step = sys_data["component_1"]["volume"] * dens * 1000. / 86400.
        delta_ts = np.ones(3) * one_step
        feed_vector = np.array(
            [[5.E21, 5.E21], [0.E22, 1.E22], [2.5e+21, 7.5e+21]])
        feed_density = np.array([10, 10, 10])

        ref_soln = mat.number_densities.copy()

        # Lets make sure the feed vectors match
        np.testing.assert_array_equal(test_d.systems[0].feed_vector,
                                      feed_vector)

        # Get our volumes and volume ratios for computing the new feed
        v_sys = sys_data["component_1"]["volume"] + \
            sys_data["component_2"]["volume"]
        # The time fed to the control volume is simply the time added
        # to this volume, hence we need to include how much time this
        # volume was being fed to
        feed_duration = 3.6 * dens / 10 * 1e3
        # number of times salt loops through the system for each time step
        loops = 50 / feed_duration / 2 * 86400

        # Deplete!
        # loop through each time step (each step has diff feed)
        for t in range(3):
            for ns in range(num_substeps):
                # Get the volumetric feed rate
                mass = np.dot(feed_vector[t], depllib.atomic_mass_vector) / \
                    adder.constants.AVOGADRO
                feed_volume_rate = mass / feed_density[t]
                # The total feed volume added is thus
                # feed_volume * feed_duration
                added_feed_volume = feed_volume_rate * feed_duration
                # Get the component and total volume for normalization
                component_volume = sys_data["component_1"]["volume"] * 1.e6
                tot_volume = component_volume + added_feed_volume
                for i in range(int(loops / num_substeps)):
                    ref_soln = \
                        (ref_soln * component_volume +
                         feed_vector[t] * feed_duration) / tot_volume

            # Deplete in our solver to be tested
            test_d.execute([mat], depl_libs, delta_ts[t], t, num_substeps, 0.)

            # Now compare the isotopes and their number densities
            np.testing.assert_allclose(mat.number_densities, ref_soln,
                                       rtol=2.5e-13)

    # Deplete with 16th order and each of the solver types
    deplete_substeps(5, 16, depllib, isos, "brute")
    deplete_substeps(5, 16, depllib, isos, "tmatrix")
    deplete_substeps(5, 16, depllib, isos, "tmatrix_expm")

    # And repeat with CRAM48
    deplete_substeps(5, 48, depllib, isos, "brute")
    deplete_substeps(5, 48, depllib, isos, "tmatrix")
