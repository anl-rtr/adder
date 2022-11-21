import adder
import pytest
import numpy as np
import h5py
import os


def test_material_init():
    # Tests the initialization of a Material object

    # Set the parameters we will frequently use
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE

    # Check the type and value checks of each of the input parameters
    # Check name
    with pytest.raises(TypeError):
        test_mat = adder.Material(int(name), mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    # Check id_
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, str(mat_id), density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)
    with pytest.raises(ValueError):
        test_mat = adder.Material(name, 0, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)
    with pytest.raises(ValueError):
        test_mat = adder.Material(name, 100000000, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    # Check density
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, [1.0],
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, "2",
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, -1.,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    # adder.Isotope test will handle isotope_data as that test has proven
    # Isotope checks types properly and wont allow a bad type

    # Check atom_fractions
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, 1.0, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, ["1", 1, 1], is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, [1., 1.], is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, [1., 1., -1.], is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    # Check is_depleting
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, "false",
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)

    # Check default_xs_library
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  70, num_groups, thermal_xs_libraries, status)

    # Check num_groups
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, "1",
                                  thermal_xs_libraries, status)

    # Check thermal_xs_libraries
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  "lwtr.20t", status)
    with pytest.raises(TypeError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  ["lwtr.20t", 1], status)

    # Check status
    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, "in core")
    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, 3)
    with pytest.raises(ValueError):
        test_mat = adder.Material(name, mat_id, density,
                                  isotope_data, atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, -1)

    # Check that the attributes exist and their values are set correctly
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    assert test_mat.name == name
    assert test_mat.id == mat_id
    assert test_mat.density == density
    for i, test_iso in enumerate(test_mat.isotopes):
        assert test_iso.name == isotope_data[i][0]
        assert test_iso.xs_library == isotope_data[i][1]
    np.testing.assert_array_equal(test_mat.atom_fractions, [0.4, 0.5, 0.1])
    assert test_mat.is_depleting == is_depleting
    assert test_mat.default_xs_library == default_xs_library
    assert test_mat.num_groups == num_groups
    assert test_mat.thermal_xs_libraries == thermal_xs_libraries
    assert test_mat.isotopes_in_neutronics == [True, True, True]
    assert test_mat.volume is None
    np.testing.assert_array_equal(test_mat.flux, np.zeros(1))
    assert test_mat.Q == 0.
    ref_N = np.array([0.4, 0.5, 0.1]) * 2. * 1.E24
    np.testing.assert_array_equal(test_mat.number_densities, ref_N)
    assert test_mat.num_isotopes == 3
    assert isinstance(test_mat.isotopes_to_keep_in_model, set)
    assert len(test_mat.isotopes_to_keep_in_model) == 0

    # Quick test of the material name repr
    repr_str = str(test_mat)
    assert repr_str == "<Material Name: 1, Id: 1, is_depleting: True>"

    # Test the mass_density functionality
    ref_amu = np.array([1.00782503224, 235.043928190, 238.050786996])
    ref_rho = np.dot(ref_N, ref_amu) / 6.022140857e23
    assert abs(test_mat.mass_density - ref_rho) / ref_rho < 1.e-13

    # Test the initialization of a depleting material that has no default
    # cross section library provided. In this case we expect an error raised
    # to the log file

    # Now setup the situation and test. We will do this twice, once with
    # an empty string and again with a set of blanks.
    is_depleting = True
    for default_xs_library in ['', '    ']:
        test_mat = adder.Material(name, mat_id, density, isotope_data,
                                  atom_fractions, is_depleting,
                                  default_xs_library, num_groups,
                                  thermal_xs_libraries, status)
        # We expect test_mat.logs to now contain a single error message as
        # described in ref_val
        ref_val = ("error", f'Material {name} (id: {mat_id}) is depleting '
                   'without a default cross section library defined!', None)
        assert len(test_mat.logs) == 1
        assert test_mat.logs[0] == ref_val


def test_material_establish_initial_isotopes():
    # Tests the initialization of a Material object's isotopes_to_keep_in_model

    # Set the parameters we will  use
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE

    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)

    # Try both True and False arguments to Material.establish_initial_isotopes
    test_mat.establish_initial_isotopes(False)
    # Since the value is False, then we apply the setting to all isotopes present
    assert len(test_mat.isotopes_to_keep_in_model) == 3
    assert sorted(test_mat.isotopes_to_keep_in_model) == \
        sorted({'H1', 'U235', 'U238'})

    # Now reset and do the other value
    test_mat.isotopes_to_keep_in_model = set()
    test_mat.establish_initial_isotopes(True)
    # Now we apply it to none of them
    assert len(test_mat.isotopes_to_keep_in_model) == 0


def test_material_clone(simple_lib):
    # Create a material to be cloned
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    depl_libs = {0: simple_lib}
    status = adder.constants.SUPPLY
    orig_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    orig_mat.is_default_depletion_library = True
    orig_mat.isotopes_to_keep_in_model.add('H1')

    # Now call orig_mat.clone to create test_mat, the clone we want to test
    # We will have to do this with and without a new name
    # First with a name
    new_name = "test_me"
    test_mat = orig_mat.clone(depl_libs, new_name)

    # Now verify the clone
    assert test_mat.name == new_name
    assert test_mat.id == 2
    assert test_mat.density == density
    for i, test_iso in enumerate(test_mat.isotopes):
        assert test_iso.name == isotope_data[i][0]
        assert test_iso.xs_library == isotope_data[i][1]
    np.testing.assert_array_equal(test_mat.atom_fractions, [0.4, 0.5, 0.1])
    assert test_mat.is_depleting == is_depleting
    assert test_mat.default_xs_library == default_xs_library
    assert test_mat.num_groups == num_groups
    assert test_mat.thermal_xs_libraries == thermal_xs_libraries
    assert test_mat.isotopes_in_neutronics == [True, True, True]
    assert test_mat.volume is None
    np.testing.assert_array_equal(test_mat.flux, np.zeros(1))
    assert test_mat.Q == 0.
    ref_N = np.array([0.4, 0.5, 0.1]) * 2. * 1.E24
    np.testing.assert_array_equal(test_mat.number_densities, ref_N)
    assert test_mat.num_isotopes == 3
    assert test_mat.is_default_depletion_library is True
    assert test_mat.depl_lib_name == 0
    assert test_mat.isotopes_to_keep_in_model == {'H1'}

    # Now re-do, with the default name; no need to re-test the code
    # aside from the name and the id
    # We also will make sure the depletion library is updated
    orig_mat.is_default_depletion_library = False
    test_mat = orig_mat.clone(depl_libs)

    # Now verify the clone
    assert test_mat.name == name + "[2]"
    assert test_mat.id == 3
    assert test_mat.depl_lib_name == test_mat.name
    assert len(depl_libs) == 2


def test_material_update_isotope_is_depleting(simple_lib):
    # Create a material to test
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c", True), ["U235", "70c", True],
                    ["U238", "72c", False]]
    atom_fractions = [4., 5., 1.]
    is_depleting = False
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    depl_libs = {0: simple_lib}
    new_lib = depl_libs[0].clone()
    depl_libs[new_lib.name] = new_lib
    status = adder.constants.IN_CORE
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)

    # First, if we call update_isotope_is_depleting, since the mat is
    # not depleting, nothing happens
    test_mat.update_isotope_is_depleting(depl_libs[0])

    # Verify by checking the isotopes' status hasn't changed from what
    # we initialized it to
    for i, iso in enumerate(test_mat.isotopes):
        assert iso.is_depleting == isotope_data[i][-1]

    # Ok, now assign the depletion library to be a clone of simple_lib
    # (we do a clone so that simple_lib isn't modified for other uses
    # in this module)
    test_mat.is_default_depletion_library = False
    test_mat.is_depleting = True
    test_mat.depl_lib_name = new_lib.name
    test_mat.update_isotope_is_depleting(new_lib)

    # Since simple_lib does not include H1, we should see that H1 is now
    # non-depleting
    isotope_data[0] = ("H1", "70c", False)
    for i, iso in enumerate(test_mat.isotopes):
        assert iso.is_depleting == isotope_data[i][-1]
    # We should also see the info that H1 was changed
    assert len(test_mat.logs) == 1
    assert test_mat.logs[0][0] == "info_file"
    # And finally, we should see our depletion_library now includes
    # a stable, zero xs H1
    assert "H1" in new_lib.isotopes.keys()
    iso_lib = new_lib.isotopes["H1"]
    np.testing.assert_array_equal(iso_lib.get_total_removal_xs("b"), [0.])
    np.testing.assert_array_equal(iso_lib.get_total_decay_const("s"), [0.])


def test_material_compute_Q(simple_lib):
    name = "1"
    mat_id = 1
    density = 2.
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE
    depl_libs = {0: simple_lib}
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    # need flux to have a value for this to work, so set to 1.0
    test_mat.volume = 1.
    test_mat.flux = np.ones(1)
    test_mat.depl_lib_name = simple_lib.name

    # Before checking, get ref values
    v_Q = 0.5 * (1.29927E-3 * 92. * 92. * np.sqrt(235.) + 33.12) * 10.0
    v_Q += 0.1 * (1.29927E-3 * 92. * 92. * np.sqrt(238.) + 33.12) * 1.0
    # include volume and density
    v_Q *= 1.0 * 2.0
    ref_tot_Q = v_Q

    v_fr = 0.5 * 10.0
    v_fr += 0.1 * 1.0
    # include volume and density
    v_fr *= 1.0 * 2.0
    ref_tot_fr = v_fr

    test_Q, test_fr = test_mat.compute_Q(simple_lib)
    assert abs(test_Q - ref_tot_Q) < 1.e-14
    assert abs(test_fr - ref_tot_fr) < 1.e-14

    # Lets verify that a non-depleting material returns 0s
    test_mat.is_depleting = False
    test_Q, test_fr = test_mat.compute_Q(simple_lib)
    assert test_Q == 0.
    assert test_fr == 0.


def test_material_determine_important_isotopes(simple_lib, simple_lib_h1u234):
    # Create a material to test
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c", False), ["U235", "70c", True],
                    ["U238", "72c", True]]
    atom_fractions = [4., 5., 1.e-11]
    is_depleting = False
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    lib = simple_lib.clone()
    lib.set_isotope_indices()
    depl_libs = {0: simple_lib, lib.name: lib}
    test_mat.depl_lib_name = lib.name

    # Before we go ahead, prove that test_mat.isotopes_in_neutronics
    # is all true
    assert all(test_mat.isotopes_in_neutronics)

    # Since flux is 0, the method will set all isotopes to being in the
    # neutronics except for ones with concentrations less tan 1E-10
    # (so U238 should be considered not in neutronics)
    test_mat.determine_important_isotopes(lib)
    assert test_mat.isotopes_in_neutronics == [True, True, False]

    # Now we need to have a flux so we can step through the rest of
    # the function
    test_mat.flux = np.array([1.E13])
    test_mat.determine_important_isotopes(lib)
    # Now, we expect that H1 will be in the model (since it is not depleting)
    # U235 will be because its important, and U238 is at such a low
    # concentration that it will not be. Therefore, the
    # isotopes to neglect will be the same as in the test block above
    assert test_mat.isotopes_in_neutronics == [True, True, False]

    # Finally we run the exact same case but now we tell it that U238 must be
    # kept in the model no matter what. Therefore we expect the corresponding
    # value in test_mat.isotopes_in_neutronics to be True.
    test_mat.isotopes_to_keep_in_model = {'U238'}
    test_mat.determine_important_isotopes(lib)
    assert test_mat.isotopes_in_neutronics == [True, True, True]

    # Now lets repeat with a purely absorbing material
    # Create a material to test
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c", False), ["U234", "70c", True]]
    atom_fractions = [4., 1.e-8]
    is_depleting = False
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    lib = simple_lib_h1u234.clone()
    lib.set_isotope_indices()
    depl_libs = {0: simple_lib_h1u234, lib.name: lib}
    test_mat.depl_lib_name = lib.name
    test_mat.flux = np.array([1.E13])

    # Before we go ahead, prove that test_mat.isotopes_in_neutronics
    # is all true
    assert all(test_mat.isotopes_in_neutronics)

    # Now we expect that U234 will not be included but H1 will be. Lets check
    test_mat.determine_important_isotopes(lib)
    assert test_mat.isotopes_in_neutronics == [True, False]


def test_material_number_density_vectors(simple_lib):
    # Create a material to test
    name = "1"
    mat_id = 1
    density = 2.  # a/b-cm
    isotope_data = [("H1", "70c", False), ["U235", "70c", True],
                    ["U238", "72c", True]]
    atom_fractions = [4., 5., 1.]
    is_depleting = False
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)

    # First test getting the number density vector with a provided
    # simple_lib
    lib = simple_lib.clone()
    lib.set_isotope_indices()
    test_Ns = test_mat.get_library_number_density_vector(lib.isotope_indices)
    ref_N = np.array([0.5, 0.1]) * 2. * 1.E24
    np.testing.assert_array_equal(test_Ns, ref_N)

    # Repeat with simple_lib being assigned to the material
    test_mat.depl_lib_name = lib.name
    test_Ns = test_mat.get_library_number_density_vector(lib.isotope_indices)
    np.testing.assert_array_equal(test_Ns, ref_N)

    # Now we can move on to testing updating the material with a number
    # density vector
    # Since depletion_library is already set, we will do that one first
    n_in = np.array([2., 0.4]) * ref_N
    # Now if we get test_mat.number_densities, it should be the same
    # as n_in (we already tested this converts atom fractions and
    # density correctly)
    test_mat.update_from_number_densities(n_in, lib)
    ref_N = np.append(n_in, 8.E23)
    np.testing.assert_allclose(test_mat.number_densities, ref_N,
                               rtol=1.e-15)

    # Repeat with lib being provided
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    # Prove that there is no depletion lib
    assert test_mat.depl_lib_name is 0
    # And now do my test with a provided library
    test_mat.update_from_number_densities(n_in, lib)
    np.testing.assert_allclose(test_mat.number_densities, ref_N,
                               rtol=1.e-15)

    # Finally, we need to test the passing of a scaling constant
    test_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    # The ref soln starts with n_in and the non-depleting isotope (H1)
    ref_N = np.append(n_in, [test_mat.number_densities[0]])
    # Then we normalize (/np.sum()) and scale to 1.5 * current density (2)
    ref_N *= 1.5 * 2.e24 / np.sum(ref_N)
    # And now do my test with a provided library
    test_mat.update_from_number_densities(n_in, lib=lib, scale_constant=1.5)
    np.testing.assert_allclose(test_mat.number_densities, ref_N,
                               rtol=1.e-15)


def test_material_hdf5(simple_lib):
    # This test will be performed by initializing a test object
    # writing to an HDF5 file, reading it back in, and then comparing
    # values

    # Set the parameters we will use
    name = "1"
    mat_id = 1
    density = 2.
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    flux = np.array([10.])
    volume = 2.
    status = adder.constants.IN_CORE

    # initialize the material and write it to an hdf5 file
    init_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, thermal_xs_libraries, status)
    init_mat.flux = flux
    init_mat.volume = volume

    # Clone material an arbitrary number of times n_clones
    # to test the num_copies attribute
    n_clones = 7
    depl_libs = {0: simple_lib}
    for i in range(n_clones):
        init_mat.clone(depl_libs, f'{i+2}')

    with h5py.File("test.h5", "w") as temp_h5:
        temp_grp = temp_h5.create_group("materials")
        init_mat.to_hdf5(temp_grp)

    # Now reopen the file and recreate the material to test
    with h5py.File("test.h5", "r") as temp_h5:
        temp_grp = temp_h5["materials/"]
        test_mat = adder.Material.from_hdf5(temp_grp, name)

    assert test_mat.name == name
    assert test_mat.id == mat_id
    assert test_mat.density == density
    for i, test_iso in enumerate(test_mat.isotopes):
        assert test_iso.name == isotope_data[i][0]
        assert test_iso.xs_library == isotope_data[i][1]
    np.testing.assert_array_equal(test_mat.atom_fractions, [0.4, 0.5, 0.1])
    assert test_mat.is_depleting == is_depleting
    assert test_mat.default_xs_library == default_xs_library
    assert test_mat.num_groups == num_groups
    assert test_mat.thermal_xs_libraries == thermal_xs_libraries
    assert test_mat.isotopes_in_neutronics == [True, True, True]
    assert test_mat.volume == volume
    np.testing.assert_array_equal(test_mat.flux, flux)
    assert test_mat.Q == 0.
    ref_N = np.array([0.4, 0.5, 0.1]) * 2. * 1.E24
    np.testing.assert_array_equal(test_mat.number_densities, ref_N)
    assert test_mat.num_isotopes == 3
    assert test_mat.status == status
    assert test_mat.num_copies == n_clones

    os.remove("test.h5")

    # Finally, test a case where there are thermal xs libs
    # (since this would be a different code path)
    # initialize the material and write it to an hdf5 file
    init_mat = adder.Material(name, mat_id, density, isotope_data,
                              atom_fractions, is_depleting, default_xs_library,
                              num_groups, ["lwtr.20t"], status)
    init_mat.flux = flux
    init_mat.volume = volume

    with h5py.File("test.h5", "w") as temp_h5:
        temp_grp = temp_h5.create_group("materials")
        init_mat.to_hdf5(temp_grp)

    # Now reopen the file and recreate the material to test
    with h5py.File("test.h5", "r") as temp_h5:
        temp_grp = temp_h5["materials/"]
        test_mat = adder.Material.from_hdf5(temp_grp, name)

    assert test_mat.thermal_xs_libraries == ["lwtr.20t"]
