import numpy as np
import pytest

from adder import Material
from adder.constants import AVOGADRO
from adder.isotope import *
from adder.msr import MSRDepletion
from adder.constants import IN_CORE


def test_set_msr_params(depletion_lib):
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

    exec_cmd = ""
    num_threads = 1
    num_procs = 2
    cram_order = 16
    chunksize = 3
    test_d = MSRDepletion(exec_cmd, num_threads, num_procs, chunksize,
        cram_order)
    assert test_d.exec_cmd == exec_cmd
    assert test_d.num_threads == num_threads
    assert test_d.num_procs == num_procs
    assert test_d.chunksize == chunksize
    # We verify alpha, theta and alpha0 by simply checking the first and
    # last values
    assert test_d.alpha0 == 2.124853710495224e-16
    assert len(test_d.alpha) == 8
    assert test_d.alpha[0] == np.complex128(5.464930576870210e+3 -
                                            3.797983575308356e+4j)
    assert test_d.alpha[-1] == np.complex128(2.394538338734709e+1 -
                                             5.650522971778156e+0j)
    assert len(test_d.theta) == 8
    assert test_d.theta[0] == np.complex128(3.509103608414918 +
                                            8.436198985884374j)
    assert test_d.theta[-1] == np.complex128(-10.84391707869699 +
                                             19.27744616718165j)
    depl_libs[0].set_atomic_mass_vector()

    test_d.set_msr_params("average", method, [sys_data], materials, depl_libs)

    assert test_d.flux_smoothing_method == "average"
    assert len(test_d.systems) == 1
    assert test_d.fluid_mats == ["mat1", "mat2", "mat3"]
    # Do a minimal amount of checking the one system we have
    test_s = test_d.systems[0]
    assert test_s.name == sys_data["name"]
    assert test_s.method == method
    assert test_s.mass_flowrate == sys_data["flowrate"]
    assert test_s.library.name == sys_data["name"] + "_" + depletion_lib.name
    assert test_s.path_T_matrices == []
    assert test_s.num_original_isotopes == 7
    assert test_s._concentration_history is None
    assert len(test_s.component_network) == 6
    assert len(test_s.component_network_weights) == 6

    # Check the property
    assert test_d.solver == "MSR_cram16"


def test_execute(depletion_lib):
    method = "brute"
    lib = depletion_lib
    depl_libs = {0: lib}
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

    exec_cmd = ""
    num_threads = 1
    num_procs = 2
    cram_order = 16
    chunksize = 3
    test_d = MSRDepletion(exec_cmd, num_threads, num_procs, chunksize,
        cram_order)

    test_d.set_msr_params("average", method, [sys_data], materials, depl_libs)

    # Lets check the flux setting
    mat1.flux = np.array([1.])
    mat2.flux = np.array([2.])
    mat3.flux = np.array([3.])
    test_d.execute(materials, depl_libs, 337.5 / 86400., 0, 1, 0.)

    avg_flux = (5. * 5. * mat1.flux + 10. * 50. * mat2.flux +
                10. * 50. * mat3.flux) / (5. * 5. + 10. * 50. + 10. * 50.)
    np.testing.assert_allclose(mat1.flux, avg_flux)
    np.testing.assert_allclose(mat2.flux, avg_flux)
    np.testing.assert_allclose(mat3.flux, avg_flux)

    # Reset the flux and do it with a histogram smoothing
    test_d.flux_smoothing_method = "histogram"
    mat1.flux = np.array([1.])
    mat2.flux = np.array([2.])
    mat3.flux = np.array([3.])

    # First lets check the flux setting
    test_d.execute(materials, depl_libs, 337.5 / 86400., 0, 1, 0.)
    np.testing.assert_allclose(mat1.flux, np.array([1.]))
    np.testing.assert_allclose(mat2.flux, np.array([2.]))
    np.testing.assert_allclose(mat3.flux, np.array([3.]))

    # Ok, now we need to actually do the execution, but, like in
    # test_msr_system, we will use a brute solver with an actual
    # identity matrix sort of solution so there is no transmutation
    def operator(mat, n_in, dt, units):
        # Again, we simply want to return the values of n_in
        return n_in.copy()
    test_d._eval_expm = operator
    test_d.systems[0]._concentration_history = None

    # reset mat1's concentrations to match what we expect for the ref
    mat1.isotopes = [Isotope("U235", "70c")]
    mat1.atom_fractions = [1.]
    mat1.density = 1.E-4

    test_d.execute(materials, depl_libs, 337.5 / 86400., 0, 1, 0.)

    # Solution same as in test_msr_system
    calc_isos = [iso.name for iso in mat1.isotopes]
    assert calc_isos == ["H1", "U235"]
    np.testing.assert_allclose(mat1.number_densities,
                               [5.28e14, 1.0e+20], rtol=1.8E-7)
    np.testing.assert_allclose(mat1.density,
                               np.sum(mat1.number_densities) / 1.E24)

