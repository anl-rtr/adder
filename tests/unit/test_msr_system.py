import numpy as np
import scipy.sparse as ss
import pytest

from adder import Material
from adder.data import atomic_mass
from adder.constants import AVOGADRO
from adder.msr import MSRSystem
from adder.constants import IN_CORE


def test_system_init(depletion_lib):
    method = "brute"
    lib = depletion_lib
    # For these we will apply a number density that yields a mass density
    # of 1, 2, and 3 g/cc
    num_dens = 1. / (235.043928190 * 1.e24) * AVOGADRO
    mat1 = Material("mat1", 1, 1. * num_dens, [["U235", "70c"]], [1.],
                    True, "71c", 1, [], IN_CORE)
    mat2 = Material("mat2", 2, 2. * num_dens, [["U235", "70c"]], [1.],
                    True, "71c", 1, [], IN_CORE)
    mat3 = Material("mat3", 3, 3. * num_dens, [["U235", "70c"]], [1.],
                    True, "71c", 1, [], IN_CORE)
    mat1.is_default_depletion_library = True
    mat2.is_default_depletion_library = True
    mat3.is_default_depletion_library = False
    lib.set_atomic_mass_vector()
    new_lib = lib.clone()
    mat1.depl_lib_name = 0
    mat2.depl_lib_name = 0
    mat3.depl_lib_name = new_lib.name
    depl_libs = {0: lib, new_lib.name: new_lib}
    materials = [mat1, mat2, mat3]
    # Set the top-level data
    sys_data = {"name": "sys1", "flowrate": 1000., "flow_start": "incore1"}

    # Now add a feed vector
    feed_dict = {"feed_vector": [{"H1": 1.}], "vector_units": "ao",
                 "feed_rate_units": "atoms/sec", "feed_rate": [2.E20],
                 "density": [1.]}
    sys_data["feed"] = feed_dict

    # And make a system with 2 in-core flowpaths, and an ex-core
    # loop with a parallel flowpath
    # in-core paths:
    comp1 = {"type": "in-core", "name": "incore1", "volume": 5.,
             "mat_name": "mat1", "removal_vector": {},
             "downstream_components": ["incore2", "incore3"],
             "downstream_mass_fractions": [0.4, 0.6]}
    comp2 = {"type": "in-core", "name": "incore2", "volume": 10.,
             "mat_name": "mat2", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp3 = {"type": "in-core", "name": "incore3", "volume": 10.,
             "mat_name": "mat3", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp4 = {"type": "generic", "name": "outlet", "volume": 20.,
             "density": 1., "removal_vector": {},
             "downstream_components": ["path1", "path2"],
             "downstream_mass_fractions": [0.8, 0.2]}
    comp5 = {"type": "generic", "name": "path1", "volume": 30.,
             "density": 1., "removal_vector": {"U235": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}
    comp6 = {"type": "generic", "name": "path2", "volume": 40.,
             "density": 1., "removal_vector": {"H1": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}

    sys_data["component_1"] = comp1
    sys_data["component_2"] = comp2
    sys_data["component_3"] = comp3
    sys_data["component_4"] = comp4
    sys_data["component_5"] = comp5
    sys_data["component_6"] = comp6

    # Begin with some mal-formed input, making sure we get the right
    # errors
    with pytest.raises(ValueError):
        test_s = MSRSystem(1, sys_data, depl_libs, materials)
    with pytest.raises(TypeError):
        test_s = MSRSystem(method, 1, depl_libs, materials)
    with pytest.raises(TypeError):
        test_s = MSRSystem(method, sys_data, 1, materials)
    with pytest.raises(TypeError):
        test_s = MSRSystem(method, sys_data, depl_libs, 1)
    # Invalid start component
    sys_data["flow_start"] = 1
    with pytest.raises(ValueError):
        test_s = MSRSystem(method, sys_data, depl_libs, materials)
    sys_data["flow_start"] = "incore1"

    # Now we check for a successful start
    test_s = MSRSystem(method, sys_data, depl_libs, materials)
    assert test_s.name == sys_data["name"]
    assert test_s.method == method
    assert test_s.mass_flowrate == sys_data["flowrate"]
    assert test_s.library.name == sys_data["name"] + "_" + depletion_lib.name
    assert test_s.path_T_matrices == []
    assert test_s.num_original_isotopes == 7
    assert test_s._concentration_history is None
    assert len(test_s.component_network) == 6
    assert len(test_s.component_network_weights) == 6
    # As we already test Components elsewhere, we only need to make
    # sure we handle differences between in-core and ex-core inits
    # To do that we will check volume, density, m_dots, and in-core
    ref_flow_fracs = np.array([1., 0.4, 0.6, 1., 0.8, 0.2]) * \
        sys_data["flowrate"]
    for dkey, name, m_dot in zip(["component_1", "component_2", "component_3",
                                  "component_4", "component_5", "component_6"],
                                 ["incore1", "incore2", "incore3", "outlet",
                                  "path1", "path2"], ref_flow_fracs):
        # Get the right component
        for comp in test_s.component_network:
            if comp.name == name:
                break
        # Get the weights
        wgts = test_s.component_network_weights[comp]

        # Now we can compare
        assert comp.volume == sys_data[dkey]["volume"]
        if name.startswith("incore"):
            # get the material name
            mat_name = comp.mat_name
            # the density is the material name # as a float
            density = float(mat_name[-1])
            assert abs(comp.density - density) < 1e-15
            assert comp.in_core is True
        else:
            assert comp.density == sys_data[dkey]["density"]
            assert comp.in_core is False
        assert comp.mass_flowrate == m_dot

        assert wgts == sys_data[dkey]["downstream_mass_fractions"]

        # Check that we find the right start
        if name == "incore1":
            assert test_s.starting_component == comp

    # Check out paths are correct by comparing the names
    ref_pathnames = \
        [["incore1", "incore2", "outlet", "path1"],
         ["incore1", "incore3", "outlet", "path1"],
         ["incore1", "incore2", "outlet", "path2"],
         ["incore1", "incore3", "outlet", "path2"]]
    for p_idx, path in enumerate(test_s.paths):
        for c_idx, comp in enumerate(path):
            assert comp.name == ref_pathnames[p_idx][c_idx]

    # Check that the path weights are correct
    np.testing.assert_allclose(test_s.path_weights, [0.32, 0.48, 0.08, 0.12])

    # Check feed
    feed = np.zeros(test_s.library.num_isotopes)
    feed[0] = 2.E20
    np.testing.assert_allclose(test_s.feed_vector, [feed])
    mass_feed_rate = feed[0] * atomic_mass("H1") / AVOGADRO / 1000.
    np.testing.assert_allclose(test_s.mass_feed_rate, mass_feed_rate)
    np.testing.assert_allclose(test_s.feed_density, 1.)
    np.testing.assert_allclose(test_s.feed_v_rate, mass_feed_rate * 1e3 / 1.)
    # MSRSystem.add_feed() is sufficiently tested by the analytic test

    # Now lets test some other paths of init_feed
    # First, empty
    test_s._init_feed({})
    feed = np.zeros(test_s.library.num_isotopes)
    np.testing.assert_allclose(test_s.feed_vector, [feed])
    np.testing.assert_allclose(test_s.mass_feed_rate, 0.)

    # Next, a feed vector defined in terms of w/o
    feed_dict["vector_units"] = "wo"
    # to really test, will need more than one isotope
    feed_dict["feed_vector"] = [{"H1": 0.1, "U235": 0.9}]
    # With one isotope this should yield the same as wo
    test_s._init_feed(feed_dict)
    feed[0] = (0.1 / atomic_mass("H1"))
    feed[5] = (0.9 / atomic_mass("U235"))
    feed *= 2.E20 / np.sum(feed)
    np.testing.assert_allclose(test_s.feed_vector, [feed])
    mass_feed_rate = (feed[0] * atomic_mass("H1") +
                      feed[5] * atomic_mass("U235")) / AVOGADRO / 1000.
    np.testing.assert_allclose(test_s.mass_feed_rate, mass_feed_rate)

    # Now a case with kg/sec as the feed rate (1 g/sec)
    feed_dict["feed_rate_units"] = "kg/sec"
    feed_dict["feed_rate"] = [1.e-3]
    test_s._init_feed(feed_dict)
    feed /= 2.E20  # (take out the old magnitude so we can calc and
    # apply the new)
    g_per_mole = feed[0] * atomic_mass("H1") + feed[5] * atomic_mass("U235")
    # Since we are injecting 1g/mole, dont need to scale from kg to g
    feed *= AVOGADRO / g_per_mole
    np.testing.assert_allclose(test_s.feed_vector, [feed])
    mass_feed_rate = 1.e-3
    np.testing.assert_allclose(test_s.mass_feed_rate, mass_feed_rate)

    # Finally, test the properties
    # Start with path times
    delta_ts = {"incore1": 5. * 1000. / 1000.,
                "incore2": 10. * 2000. / 400.,
                "incore3": 10. * 3000. / 600.,
                "outlet": 20. * 1000. / 1000.,
                "path1": 30. * 1000. / 800.,
                "path2": 40. * 1000. / 200.}
    ref_path_times = [[delta_ts[c] for c in path] for path in ref_pathnames]

    sum_path_times = [np.sum(dt) for dt in ref_path_times]
    test_s.update_time_properties()
    for i in range(len(ref_path_times)):
        np.testing.assert_allclose(test_s.path_times[i], sum_path_times[i])
    np.testing.assert_allclose(test_s.min_transport_time,
                               np.min(sum_path_times))

    np.testing.assert_allclose(test_s.path_offsets, [0, 0, 1, 1])


def test_system_calc(depletion_lib):
    method = "brute"
    lib = depletion_lib
    num_dens = 1. / (235.043928190 * 1.e24) * AVOGADRO
    mat1 = Material("mat1", 1, 1. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)
    mat2 = Material("mat2", 2, 2. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)

    mat3 = Material("mat3", 3, 3. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)
    mat1.is_default_depletion_library = True
    mat2.is_default_depletion_library = True
    mat3.is_default_depletion_library = False
    lib.set_atomic_mass_vector()
    new_lib = lib.clone()
    mat1.depl_lib_name = 0
    mat2.depl_lib_name = 0
    mat3.depl_lib_name = new_lib.name
    depl_libs = {0: lib, new_lib.name: new_lib}
    materials = [mat1, mat2, mat3]
    # Set the top-level data
    sys_data = {"name": "sys1", "flowrate": 1000., "flow_start": "incore1"}

    # Now add a feed vector
    feed_dict = {"feed_vector": [{"H1": 1.}], "vector_units": "ao",
                 "feed_rate_units": "atoms/sec", "feed_rate": [2.E20],
                 "density": [1.]}
    sys_data["feed"] = feed_dict

    # And make a system with 2 in-core flowpaths, and an ex-core
    # loop with a parallel flowpath
    # in-core paths:
    comp1 = {"type": "in-core", "name": "incore1", "volume": 5.,
             "mat_name": "mat1", "removal_vector": {},
             "downstream_components": ["incore2", "incore3"],
             "downstream_mass_fractions": [0.4, 0.6]}
    comp2 = {"type": "in-core", "name": "incore2", "volume": 10.,
             "mat_name": "mat2", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp3 = {"type": "in-core", "name": "incore3", "volume": 10.,
             "mat_name": "mat3", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp4 = {"type": "generic", "name": "outlet", "volume": 20.,
             "density": 1., "removal_vector": {},
             "downstream_components": ["path1", "path2"],
             "downstream_mass_fractions": [0.8, 0.2]}
    comp5 = {"type": "generic", "name": "path1", "volume": 30.,
             "density": 1., "removal_vector": {"U235": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}
    comp6 = {"type": "generic", "name": "path2", "volume": 40.,
             "density": 1., "removal_vector": {"H1": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}

    sys_data["component_1"] = comp1
    sys_data["component_2"] = comp2
    sys_data["component_3"] = comp3
    sys_data["component_4"] = comp4
    sys_data["component_5"] = comp5
    sys_data["component_6"] = comp6
    test_s = MSRSystem(method, sys_data, depl_libs, materials)
    # Now lets compute the matrices
    mat1.flux = np.array([1.])
    mat2.flux = np.array([2.])
    mat3.flux = np.array([3.])
    mat_by_name = {"mat1": mat1, "mat2": mat2, "mat3": mat3}
    test_s.compute_matrices(mat_by_name, 337.5 / 86400., None)
    # Make sure everybody's matrix has been
    # We dont need to check the A matrices since we did that in the
    # component test.
    for comp in test_s.component_network.keys():
        assert isinstance(comp._A_matrix, ss.csr_matrix)
        assert comp._T_matrix is None

    # Change the component to be variable density
    for comp in test_s.component_network.keys():
        comp.variable_density = True

    # Now, we can move ahead to solve. Since the verification suite will
    # test analytical depletion solutions, we will just pass a solver
    # that acts like an identity matrix operator
    def operator(mat, n_in, dt, units):
        # Again, we simply want to return the values of n_in
        return n_in.copy()

    n_in = np.zeros(7)
    n_in[5] = 1.E20
    amu = np.array([1.00782503, 4.00260325, 16.00610192, 15.99491462,
                    231.03630285, 235.04392819, 238.050787])

    # Now we need a reference solution.
    ref_feed = np.zeros_like(n_in)
    const = 1.6735328119542856e-03
    denom_const = 5.e6 + const
    feed_dt = 5.
    ref_feed[0] = 2.E20 * feed_dt
    path_outs = test_s.path_weights[0] * n_in.copy() + \
        test_s.path_weights[1] * n_in.copy() + \
        test_s.path_weights[2] * n_in.copy() + \
        test_s.path_weights[3] * n_in.copy()
    n_end_loop_1 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 2
    path_outs = test_s.path_weights[0] * n_end_loop_1.copy() + \
        test_s.path_weights[1] * n_end_loop_1.copy() + \
        test_s.path_weights[2] * n_in.copy() + \
        test_s.path_weights[3] * n_in.copy()
    n_end_loop_2 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 3
    path_outs = test_s.path_weights[0] * n_end_loop_2.copy() + \
        test_s.path_weights[1] * n_end_loop_2.copy() + \
        test_s.path_weights[2] * n_end_loop_1.copy() + \
        test_s.path_weights[3] * n_end_loop_1.copy()
    n_end_loop_3 = (5.e6 * path_outs + ref_feed) / denom_const
    ref_n_out = n_end_loop_3

    # Solve and test
    test_s.update_time_properties()
    n_out = test_s.solve(337.5 / 86400., 0, n_in, operator, amu)
    np.testing.assert_allclose(n_out, ref_n_out, rtol=1.8E-7)
    # Now we can check the densities
    density_ratio = np.dot(n_out, amu) / np.dot(n_in, amu)
    densities = [density_ratio, 2. * density_ratio, 3. * density_ratio,
                 density_ratio, density_ratio, density_ratio]
    for c, comp in enumerate(test_s.component_network.keys()):
        np.testing.assert_allclose(comp.density, densities[c])

    # Repeat the above, with a t-matrix approach
    test_s = MSRSystem("tmatrix", sys_data, depl_libs, materials)
    test_s.update_time_properties()
    test_s.compute_matrices(mat_by_name, 337.5 / 86400.,
                            lambda x, y, z, a: y)
    # Make sure everybody's matrix has been set correctly
    # We dont need to check the A matrices since we did that in the
    # component test.
    for comp in test_s.component_network.keys():
        assert isinstance(comp._A_matrix, ss.csc_matrix)
        assert isinstance(comp._T_matrix, ss.csc_matrix)

    # Now we replace the T-matrices with an identity matrix (scaled by
    # m_dot) so we can re-use the above info
    eye = ss.eye(test_s.library.num_isotopes, format="csc")
    test_s.path_T_matrices[0] = 0.4 * 0.8 * eye
    test_s.path_T_matrices[1] = 0.6 * 0.8 * eye
    test_s.path_T_matrices[2] = 0.4 * 0.2 * eye
    test_s.path_T_matrices[3] = 0.6 * 0.2 * eye

    n_out = test_s.solve(337.5 / 86400., 0, n_in, operator, amu)
    np.testing.assert_allclose(n_out, ref_n_out)
    # Now we can check the densities
    density_ratio = np.dot(n_out, amu) / np.dot(n_in, amu)
    densities = [density_ratio, 2. * density_ratio, 3. * density_ratio,
                 density_ratio, density_ratio, density_ratio]
    for c, comp in enumerate(test_s.component_network.keys()):
        np.testing.assert_allclose(comp.density, densities[c], rtol=1.8E-7)


def test_system_calc_with_tank(depletion_lib):
    method = "tmatrix"
    lib = depletion_lib
    num_dens = 1. / (235.043928190 * 1.e24) * AVOGADRO
    mat1 = Material("mat1", 1, 1. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)
    mat2 = Material("mat2", 2, 2. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)
    mat3 = Material("mat3", 3, 3. * num_dens, [["U235", "70c"]],
                    [1.], True, "71c", 1, [], IN_CORE)
    mat1.is_default_depletion_library = True
    mat2.is_default_depletion_library = True
    mat3.is_default_depletion_library = False
    lib.set_atomic_mass_vector()
    new_lib = lib.clone()
    mat1.depl_lib_name = 0
    mat2.depl_lib_name = 0
    mat3.depl_lib_name = new_lib.name
    depl_libs = {0: lib, new_lib.name: new_lib}
    materials = [mat1, mat2, mat3]
    # Set the top-level data
    sys_data = {"name": "sys1", "flowrate": 1000., "flow_start": "incore1"}

    # Now add a feed vector; this feed rate will be *very* large so that
    # the tank volume increases significantly enough that we see it in
    # one loop transport
    feed_dict = {"feed_vector": [{"H1": 1.}], "vector_units": "ao",
                 "feed_rate_units": "atoms/sec", "feed_rate": [2.E29],
                 "density": [1.]}
    sys_data["feed"] = feed_dict

    # And make a system with 2 in-core flowpaths, and an ex-core
    # loop with a parallel flowpath
    # in-core paths:
    # comp6 ("path2") is now a tank, unlike in the above
    comp1 = {"type": "in-core", "name": "incore1", "volume": 5.,
             "mat_name": "mat1", "removal_vector": {},
             "downstream_components": ["incore2", "incore3"],
             "downstream_mass_fractions": [0.4, 0.6]}
    comp2 = {"type": "in-core", "name": "incore2", "volume": 10.,
             "mat_name": "mat2", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp3 = {"type": "in-core", "name": "incore3", "volume": 10.,
             "mat_name": "mat3", "removal_vector": {},
             "downstream_components": ["outlet"],
             "downstream_mass_fractions": [1.]}
    comp4 = {"type": "generic", "name": "outlet", "volume": 20.,
             "density": 1., "removal_vector": {},
             "downstream_components": ["path1", "path2"],
             "downstream_mass_fractions": [0.8, 0.2]}
    comp5 = {"type": "generic", "name": "path1", "volume": 30.,
             "density": 1., "removal_vector": {"U235": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}
    comp6 = {"type": "tank", "name": "path2", "volume": 40.,
             "density": 1., "removal_vector": {"H1": 0.5},
             "downstream_components": ["incore1"],
             "downstream_mass_fractions": [1.]}

    sys_data["component_1"] = comp1
    sys_data["component_2"] = comp2
    sys_data["component_3"] = comp3
    sys_data["component_4"] = comp4
    sys_data["component_5"] = comp5
    sys_data["component_6"] = comp6
    test_s = MSRSystem(method, sys_data, depl_libs, materials)

    # Now lets compute the matrices so we can do the depletion
    mat1.flux = np.array([1.])
    mat2.flux = np.array([2.])
    mat3.flux = np.array([3.])
    mat_by_name = {"mat1": mat1, "mat2": mat2, "mat3": mat3}
    test_s.compute_matrices(mat_by_name, 337.5 / 86400.,
                            lambda x, y, z, a: y)

    # Now we replace the T-matrices with an identity matrix (scaled by
    # m_dot) so we can re-use the above info
    eye = ss.eye(test_s.library.num_isotopes, format="csc")
    test_s.path_T_matrices[0] = 0.4 * 0.8 * eye
    test_s.path_T_matrices[1] = 0.6 * 0.8 * eye
    test_s.path_T_matrices[2] = 0.4 * 0.2 * eye
    test_s.path_T_matrices[3] = 0.6 * 0.2 * eye

    # Let's verify the starting delta_ts and path_offsets
    ref_pathnames = \
        [["incore1", "incore2", "outlet", "path1"],
         ["incore1", "incore3", "outlet", "path1"],
         ["incore1", "incore2", "outlet", "path2"],
         ["incore1", "incore3", "outlet", "path2"]]
    delta_ts = {"incore1": 5. * 1000. / 1000.,
                "incore2": 10. * 2000. / 400.,
                "incore3": 10. * 3000. / 600.,
                "outlet": 20. * 1000. / 1000.,
                "path1": 30. * 1000. / 800.,
                "path2": 40. * 1000. / 200.}
    ref_path_times = [[delta_ts[c] for c in path] for path in ref_pathnames]

    sum_path_times = [np.sum(dt) for dt in ref_path_times]

    test_s.update_time_properties()
    for i in range(len(ref_path_times)):
        np.testing.assert_allclose(test_s.path_times[i], sum_path_times[i])
    np.testing.assert_allclose(test_s.min_transport_time,
                               np.min(sum_path_times))

    np.testing.assert_allclose(test_s.path_offsets, [0, 0, 1, 1])

    # Now, we can move ahead to solve. Since the verification suite will
    # test analytical depletion solutions, we will just pass a solver
    # that acts like an identity matrix operator
    def operator(mat, n_in, dt, units):
        # Again, we simply want to return the values of n_in
        return n_in.copy()

    n_in = np.zeros(7)
    n_in[5] = 1.E20
    amu = np.array([1.00782503, 4.00260325, 16.00610192, 15.99491462,
                    231.03630285, 235.04392819, 238.050787])

    # Now we need a reference solution.
    mass_feed_rate = 2.E29 * atomic_mass("H1") / AVOGADRO / 1000.
    const = 1.6735328119542856e+06
    denom_const = 5.e6 + const
    ref_feed = np.zeros_like(n_in)
    feed_dt = 5.
    ref_feed[0] = 2.E29 * feed_dt
    path_outs = test_s.path_weights[0] * n_in.copy() + \
        test_s.path_weights[1] * n_in.copy() + \
        test_s.path_weights[2] * n_in.copy() + \
        test_s.path_weights[3] * n_in.copy()
    n_end_loop_1 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 2
    path_outs = test_s.path_weights[0] * n_end_loop_1.copy() + \
        test_s.path_weights[1] * n_end_loop_1.copy() + \
        test_s.path_weights[2] * n_in.copy() + \
        test_s.path_weights[3] * n_in.copy()
    n_end_loop_2 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 3
    path_outs = test_s.path_weights[0] * n_end_loop_2.copy() + \
        test_s.path_weights[1] * n_end_loop_2.copy() + \
        test_s.path_weights[2] * n_end_loop_1.copy() + \
        test_s.path_weights[3] * n_end_loop_1.copy()
    n_end_loop_3 = (5.e6 * path_outs + ref_feed) / denom_const
    ref_n_out = n_end_loop_3

    n_out = test_s.solve(337.5 / 86400., 0, n_in, operator, amu)
    test_s.update_time_properties()

    np.testing.assert_allclose(n_out, ref_n_out, rtol=1e-6)
    # Verify the change in times/path offsets
    vol_ratio = (40. + 337.5 * mass_feed_rate / 1000.) / 40.
    delta_ts["path2"] *= vol_ratio
    np.testing.assert_allclose(test_s.paths[-1][-1].delta_t, delta_ts["path2"])
    ref_path_times = [[delta_ts[c] for c in path] for path in ref_pathnames]
    sum_path_times = [np.sum(dt) for dt in ref_path_times]

    for i in range(len(ref_path_times)):
        np.testing.assert_allclose(test_s.path_times[i], sum_path_times[i])
    np.testing.assert_allclose(test_s.min_transport_time,
                               np.min(sum_path_times))
    np.testing.assert_array_equal(test_s.path_offsets, [0, 0, 6, 6])
    # Check the concentration histories so we can understand the effect
    # after running solve again
    np.testing.assert_array_equal(test_s._concentration_history[:2], [1, 1])
    np.testing.assert_array_equal([len(test_s._concentration_history[i])
                                   for i in [2, 3]], [1, 1])
    # The [:-2] is because the conc_history includes 0s for the removed
    # isotopes, for which we dont have in n_end_loop_2
    np.testing.assert_allclose(test_s._concentration_history[2][0][:-2],
                               test_s.path_weights[2] * n_end_loop_2,
                               rtol=1.e-6)
    np.testing.assert_allclose(test_s._concentration_history[3][0][:-2],
                               test_s.path_weights[3] * n_end_loop_2,
                               rtol=1.e-6)

    # Now we can run solve again
    # First get the reference solution
    n_in = ref_n_out
    const = 1673532.8119542855
    denom_const = 5.e6 + const
    ref_feed = np.zeros_like(n_in)
    feed_dt = 5.
    ref_feed[0] = 2.E29 * 5.
    path_outs = test_s.path_weights[0] * n_end_loop_3.copy() + \
        test_s.path_weights[1] * n_end_loop_3.copy() + \
        test_s.path_weights[2] * n_end_loop_2.copy() + \
        test_s.path_weights[3] * n_end_loop_2.copy()
    n_end_loop_4 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 2
    path_outs = test_s.path_weights[0] * n_end_loop_4.copy() + \
        test_s.path_weights[1] * n_end_loop_4.copy() + \
        test_s.path_weights[2] * n_end_loop_3.copy() + \
        test_s.path_weights[3] * n_end_loop_3.copy()
    n_end_loop_5 = (5.e6 * path_outs + ref_feed) / denom_const
    # Loop 3
    path_outs = test_s.path_weights[0] * n_end_loop_5.copy() + \
        test_s.path_weights[1] * n_end_loop_5.copy() + \
        test_s.path_weights[2] * n_end_loop_3.copy() + \
        test_s.path_weights[3] * n_end_loop_3.copy()
    n_end_loop_6 = (5.e6 * path_outs + ref_feed) / denom_const
    ref_n_out = n_end_loop_6

    n_out = test_s.solve(337.5 / 86400., 0, n_in, operator, amu)

    np.testing.assert_allclose(n_out, ref_n_out, rtol=1.E-5)

    # Check the concentration histories to make sure they grew by the
    # correct number
    np.testing.assert_array_equal(test_s._concentration_history[:2], [1, 1])
    np.testing.assert_array_equal([len(test_s._concentration_history[i])
                                   for i in [2, 3]], [6, 6])
    # And now check their values
    for i in [2, 3]:
        test = test_s._concentration_history
        wgt = test_s.path_weights[i]
        for j in [0, 1, 2, 3]:
            np.testing.assert_allclose(test[i][j][:-2], wgt * n_end_loop_3,
                                       rtol=1.e-6)
        j = 4
        np.testing.assert_allclose(test[i][j][:-2], wgt * n_end_loop_4,
                                   rtol=1.e-6)
        j = 5
        np.testing.assert_allclose(test[i][j][:-2], wgt * n_end_loop_5,
                                   rtol=1.e-6)

