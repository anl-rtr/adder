import adder
from collections import OrderedDict
import pytest
import numpy as np
import logging


def test_reactor_init():
    # Tests the initialization of a Reactor object

    # Set the parameters we will frequently use
    name = "test"
    neutronics_solver = "test"
    depletion_solver = "test"
    mpi_cmd = "mpirun"
    neutronics_exec = "mcnp.EXE"
    depletion_exec = "origen2.EXE"
    base_neutronics_input_file = "sometext"
    h5_filename = "test.h5"
    num_threads = 2
    num_procs = 4
    chunksize = 20
    use_depletion_library_xs = True
    reactivity_thresh = 1.E-8
    reactivity_thresh_init = False

    # Check the type and value checks of each of the input parameters
    # Check name
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(1, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check neutronics_solver
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, "vim", depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, 1, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check depletion_solver
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, "burner",
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, 1,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check mpi_cmd, can be not string, or have num_procs > 1 and be empty
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                1, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                "", neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, 2, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check neutronics_exec
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, 1, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check depletion_exec
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, 1,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check base_neutronics_input_file
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                1, h5_filename, num_threads, num_threads,
                                num_procs,  chunksize, use_depletion_library_xs,
                                reactivity_thresh, reactivity_thresh_init)

    # Check h5_filename
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, 1,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check num_neut_threads
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                "1", num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                0, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    # Check num_depl_threads
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, "1", num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, 0, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    # Check num_procs
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, "1", chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, 0, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    # Check chunksize
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, "1",
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, 0,
                                use_depletion_library_xs, reactivity_thresh,
                                reactivity_thresh_init)

    # Check use_depletion_library_xs
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs,
                                chunksize, "true", reactivity_thresh,
                                reactivity_thresh_init)

    # Check reactivity_thresh
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, 1,
                                reactivity_thresh_init)
    with pytest.raises(ValueError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, -1.,
                                reactivity_thresh_init)

    # Check reactivity_thresh_init
    with pytest.raises(TypeError):
        test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                                mpi_cmd, neutronics_exec, depletion_exec,
                                base_neutronics_input_file, h5_filename,
                                num_threads, num_threads, num_procs, chunksize,
                                use_depletion_library_xs, reactivity_thresh,
                                'True')

    # Check num_procs: checked with neutronics (num_threads is temporary too)

    # Check that the attributes exist and their values are set correctly
    test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                            mpi_cmd, neutronics_exec, depletion_exec,
                            base_neutronics_input_file, h5_filename,
                            num_threads, num_threads, num_procs, chunksize,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)
    assert test_rx.name == name
    assert test_rx.neutronics.solver == neutronics_solver
    assert test_rx.depletion.solver == depletion_solver
    assert test_rx.materials is None
    assert test_rx.case_label == "Initial"
    assert test_rx.operation_label == "Initial"
    assert test_rx.case_idx == 0
    assert test_rx.operation_idx == 0
    assert test_rx.step_idx == 0
    assert test_rx.time == 0.0
    assert test_rx.power == 0.0
    assert test_rx.flux_level == 0.0
    assert test_rx.Q_recoverable == 200.0
    assert test_rx.keff == 0.0
    assert test_rx.keff_stddev == 0.0
    assert test_rx.h5_filename == h5_filename
    assert test_rx.neutronics.use_depletion_library_xs == \
        use_depletion_library_xs
    assert test_rx.neutronics.reactivity_threshold == reactivity_thresh
    assert test_rx.neutronics.reactivity_threshold_initial == \
        reactivity_thresh_init
    # Not going to check the begin time, since it just uses python
    # standard libraries, just make sure it exists
    assert hasattr(test_rx, "begin_time")
    assert test_rx.end_time is None
    # The contents of the HDF5 file will not be checked now;
    # The HDF5 File is not actually written to until update_hdf5 is
    # called, therefore it should still be None
    assert test_rx.h5_file is None
    assert test_rx.h5_initialized == False

    # Check the type checking of the setters that are not init params
    # materials
    with pytest.raises(TypeError):
        test_rx.materials = "test"
    # labels
    with pytest.raises(TypeError):
        test_rx.case_label = 1
    with pytest.raises(TypeError):
        test_rx.operation_label = 1
    # indices
    with pytest.raises(TypeError):
        test_rx.case_idx = "1"
    with pytest.raises(TypeError):
        test_rx.operation_idx = "1"
    with pytest.raises(TypeError):
        test_rx.step_idx = "1"
    # time
    with pytest.raises(TypeError):
        test_rx.time = "test"
    # power
    with pytest.raises(TypeError):
        test_rx.power = "test"
    # flux_level
    with pytest.raises(TypeError):
        test_rx.flux_level = "flux_level"
    # Q_recoverable
    with pytest.raises(TypeError):
        test_rx.Q_recoverable = "test"
    # keff
    with pytest.raises(TypeError):
        test_rx.keff = "test"
    # keff_stddev
    with pytest.raises(TypeError):
        test_rx.keff_stddev = "test"

    # Verify the __repr__ method
    assert str(test_rx) == "<Reactor test>"

    # Do a test to verify we can setup an MCNP solver case
    test_rx = adder.Reactor(name, "MCnP", depletion_solver,
                            mpi_cmd, neutronics_exec, depletion_exec,
                            base_neutronics_input_file, "test2.h5",
                            num_threads, num_threads, num_procs, chunksize,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)
    assert test_rx.neutronics.solver == "mcnp"

    # Do a test to verify we can setup an Origen solver case
    test_rx = adder.Reactor(name, neutronics_solver, "ORiGEN2.2",
                            mpi_cmd, neutronics_exec, depletion_exec,
                            base_neutronics_input_file, "test3.h5",
                            num_threads, num_threads, num_procs, chunksize,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)
    assert test_rx.depletion.solver == "origen2.2"


def test_reactor_init_materials_and_input(caplog, depletion_lib):
    root_logger = adder.init_root_logger("adder")
    depletion_lib.to_hdf5("test_lib.h5")

    # Create a reactor as our test platform
    name = "test"
    neutronics_solver = "test"
    depletion_solver = "test"
    neutronics_exec = "test"
    depletion_exec = "test"
    base_neutronics_input_file = "sometext"
    h5_filename = "test5.h5"
    neutronics_library_file = "test_lib_file.txt"
    depletion_lib_file = "test_lib.h5"
    depletion_lib_name = "test"
    user_mats_info = OrderedDict()
    user_mats_info[2] = OrderedDict()
    user_mats_info[2]["name"] = "mat 2"
    user_mats_info[2]["depleting"] = False
    user_mats_info[2]["density"] = None
    user_mats_info[2]["volume"] = 1.
    user_mats_info[2]["non_depleting_isotopes"] = ["U238"]
    user_mats_info[2]["use_default_depletion_library"] = True
    user_mats_info[2]["apply_reactivity_threshold_to_initial_inventory"] = True
    user_univ_info = OrderedDict()
    shuffled_mats = set()
    shuffled_univs = set()
    use_depletion_library_xs = False
    reactivity_thresh = 1.E-8
    reactivity_thresh_init = False

    test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                            "", neutronics_exec, depletion_exec,
                            base_neutronics_input_file, h5_filename, 1, 1, 1, 1,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)

    # Test the type checking of inputs
    # neutronics_library_file
    with pytest.raises(TypeError):
        test_rx.init_materials_and_input(1, depletion_lib_file,
                                         depletion_lib_name, user_mats_info,
                                         user_univ_info, shuffled_mats,
                                         shuffled_univs)
    # depletion_lib_file
    with pytest.raises(TypeError):
        test_rx.init_materials_and_input(neutronics_library_file,
                                         1, depletion_lib_name, user_mats_info,
                                         user_univ_info, shuffled_mats,
                                         shuffled_univs)
    # depletion_lib_name
    with pytest.raises(TypeError):
        test_rx.init_materials_and_input(neutronics_library_file,
                                         depletion_lib_file, 1,
                                         user_mats_info, user_univ_info,
                                         shuffled_mats, shuffled_univs)
    # user_mats_info
    with pytest.raises(TypeError):
        test_rx.init_materials_and_input(neutronics_library_file,
                                         depletion_lib_file,
                                         depletion_lib_name, "1",
                                         user_univ_info, shuffled_mats,
                                         shuffled_univs)

    # Run the default input case (all materials is_depleting)
    test_rx.init_materials_and_input(neutronics_library_file,
                                     depletion_lib_file, depletion_lib_name,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs)

    # Check the materials
    for i, test_mat in enumerate(test_rx.materials):
        if i == 0:
            assert test_mat.name == str(i + 1)
        else:
            assert test_mat.name == "mat 2"
        assert test_mat.density == 1.
        if i == 0:
            assert test_mat.volume is None
        else:
            assert test_mat.volume == 1.
        isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
        for j, test_iso in enumerate(test_mat.isotopes):
            assert test_iso.name == isotope_data[j][0]
            assert test_iso.xs_library == isotope_data[j][1]
            if j == 0:
                # The H1 is always non-depleting
                assert test_iso.is_depleting is False
            else:
                if i == 1 and j == 2:
                    assert test_iso.is_depleting is False
                else:
                    assert test_iso.is_depleting is True

        np.testing.assert_equal(test_mat.atom_fractions, [0.4, 0.5, 0.1])
        if i == 0:
            assert test_mat.is_depleting is True
        else:
            assert test_mat.is_depleting is False
        assert test_mat.default_xs_library == "71c"
        if i == 0:
            assert test_mat.is_default_depletion_library is False
            assert test_mat.depl_lib_name is "1"
        elif i == 1:
            assert test_mat.is_default_depletion_library is True
            assert test_mat.depl_lib_name is 0
        assert test_mat.thermal_xs_libraries == []
        assert test_mat.isotopes_in_neutronics == [True, True, True]
        assert test_mat.num_isotopes == 3
        if i == 0:
            assert isinstance(test_mat.isotopes_to_keep_in_model, set)
            assert len(test_mat.isotopes_to_keep_in_model) == 0
        elif i == 1:
            assert sorted(test_mat.isotopes_to_keep_in_model) == sorted(
                {'H1', 'U235', 'U238'})

    # Check the depletion library data
    # Verify the test library was implemented by seeing if the
    # init parameters made it through and were set, as well as checking
    # the default get_fission_xs is there.
    assert test_rx.depletion_libs[0].name == depletion_lib.name

    # Verify the neutronics input file was set
    assert isinstance(test_rx.neutronics.base_input, OrderedDict)
    assert test_rx.neutronics.base_input["options"] == ["option 1", "option 2"]
    assert test_rx.neutronics.base_input["runmode"] == ["fast"]

    # Test the output of log_materials based on manual writing of
    # expected output
    caplog.clear()
    test_rx.log_materials()
    ref_txt = """Material Naming:

         ID  |  Depl  |       Name        |  Vol [cc]
-----------------------------------------------------
          1  |  True  |         1         | None
          2  | False  |       mat 2       | 1.000000E+00"""

    # Convert ref_txt to a list of stripped lines
    # To avoid pytest version specific differences, we will remove whitespace
    # from before and after tables for the comparison.
    ref_txt = ref_txt.splitlines()
    for i in range(len(ref_txt)):
        ref_txt[i] = ref_txt[i].strip()
    # And put it back to gether
    ref_txt = "\n".join(ref_txt)

    # Repeat with caplog.text
    result_txt = caplog.text.splitlines()
    for i in range(len(result_txt)):
        result_txt[i] = result_txt[i].strip()
    # And put it back to gether
    result_txt = "\n".join(result_txt)

    # And now we can compare
    assert ref_txt in result_txt
    caplog.clear()

    # Now check the total volume property; we will do this
    # for the reactor as-is, with only one material depleting
    # and then we will set that material to depleting and re-call
    # total volume to make sure the new volume is incorporated
    # First we need to assign a volume to mat 1
    test_rx.materials[0].volume = 11.
    assert test_rx.total_volume == 11.

    test_rx.materials[1].is_depleting = True
    assert test_rx.total_volume == 11. + 1.

    # And finally we set the status of mat[0] to storage
    # And check volume
    test_rx.materials[0].status = 1
    assert test_rx.total_volume == 1.


def test_reactor_update_depletion_constants(simple_lib):
    # Note: this method also fully tests:
    # Reactor._flux_scaling_constant(...)
    # Reactor._update_material_fluxes(...)
    # Reactor._update_Q_recoverable()

    # Set up base reactor with 2 materials (same as in previous test)
    # Create two test material objects, one is_depleting, one not
    name = "1"
    mat_id = 1
    density = 1.0
    isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
    atom_fractions = [4., 5., 1.]
    is_depleting = True
    default_xs_library = "71c"
    num_groups = 1
    thermal_xs_libraries = []
    status = adder.constants.IN_CORE
    mat1 = adder.Material(name, mat_id, density, isotope_data,
                          atom_fractions, is_depleting, default_xs_library,
                          num_groups, thermal_xs_libraries, status)
    mat1.flux = np.array([2.5])
    mat1.volume = 1.
    mat1.depl_lib_name = 0

    name = "2"
    mat_id = 2
    is_depleting = False
    mat2 = adder.Material(name, mat_id, density, isotope_data,
                          atom_fractions, is_depleting, default_xs_library,
                          num_groups, thermal_xs_libraries, status)
    mat2.flux = np.array([1.37])
    mat2.volume = 2.

    # Now create a reactor we can assign the materials to
    name = "test"
    neutronics_solver = "test"
    depletion_solver = "test"
    neutronics_exec = "mcnp.EXE"
    depletion_exec = "origen2.EXE"
    base_neutronics_input_file = "sometext"
    h5_filename = "test4.h5"
    use_depletion_library_xs = True
    reactivity_thresh = 1.E-8
    reactivity_thresh_init = False

    # Initialize our reactor
    test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                            "", neutronics_exec, depletion_exec,
                            base_neutronics_input_file, h5_filename, 1, 1, 1, 1,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)
    test_rx.materials = [mat1, mat2]
    test_rx.power = 7.
    use_power = True

    # Now build our flux and volume dictionaries
    flux = OrderedDict()
    flux[1] = np.array([3.0])
    flux[2] = np.array([6.0])
    volumes = OrderedDict()
    volumes[1] = 0.5
    volumes[2] = 0.7

    # Set our constants to values to be used in updating our fluxes
    keff = 2.0
    nu = 2.4

    # Create a dummy depletion library
    depl_libs = {0: simple_lib}
    test_rx.depletion_libs = depl_libs

    # Get reference solution
    # first the Q
    # for volume 1
    v1_Q = 0.5 * (1.29927E-3 * 92. * 92. * np.sqrt(235.) + 33.12) * 10.0
    v1_Q += 0.1 * (1.29927E-3 * 92. * 92. * np.sqrt(238.) + 33.12) * 1.0
    # include volume and density
    v1_Q *= 0.5 * 1.0
    # And for volume 2
    v2_Q = 0.5 * (1.29927E-3 * 92. * 92. * np.sqrt(235.) + 33.12) * 10.0
    v2_Q += 0.1 * (1.29927E-3 * 92. * 92. * np.sqrt(238.) + 33.12) * 1.0
    # include volume and density
    v2_Q *= 0.7 * 2.0
    # Just set to 0 so not depletable (keeping above in case changed)
    v2_Q = 0.
    ref_tot_Q = v1_Q + v2_Q

    v1_fr = 0.5 * 10.0
    v1_fr += 0.1 * 1.0
    # include volume and density
    v1_fr *= 0.5 * 1.0
    # And for volume 2
    v2_fr = 0.5 * 10.0
    v2_fr += 0.1 * 1.0
    # include volume and density
    v2_fr *= 0.7 * 2.0
    # Just set to 0 so not depletable (keeping above in case changed)
    v2_fr = 0.
    ref_tot_fr = v1_fr + v2_fr
    ref_Q_recov = ref_tot_Q / ref_tot_fr

    def _reset_vals(test_rx):
        # Reset tested values
        test_rx.Q_recoverable = -1.
        test_rx.materials[0].volume = 0.
        test_rx.materials[1].volume = 0.
        test_rx.materials[0].flux[0] = 0.
        test_rx.materials[0].flux[0] = 0.

    # Now update our fluxes
    _reset_vals(test_rx)
    test_rx._update_depletion_constants(flux, nu, keff, use_power, volumes)
    assert abs(test_rx.Q_recoverable - ref_Q_recov) < 3.e-14
    ref_flux1 = 8.4 * 3.0 / 3.23194554407919E-17
    ref_flux1 = nu * test_rx.power * 3.0 / \
        (keff * 1.6021766208e-19 * ref_Q_recov)
    ref_flux2 = 0.0  # 0 since not depletable
    ref_flux_level = (ref_flux1 * volumes[1] + ref_flux2 * volumes[2]) / \
        (volumes[1])
    assert (abs(test_rx.materials[0].flux[0] - ref_flux1) / ref_flux1) < 1.e-15
    assert (abs(test_rx.materials[1].flux[0] - ref_flux2)) < 1.e-15
    assert (abs(test_rx.flux_level - ref_flux_level) / ref_flux_level) < 1.e-14
    assert abs(test_rx.materials[0].volume - volumes[1]) < 1.e-15
    assert abs(test_rx.materials[1].volume - volumes[2]) < 1.e-15

    # Try with a zero power case
    _reset_vals(test_rx)
    test_rx.power = 0.
    test_rx._update_depletion_constants(flux, nu, keff, use_power, volumes)
    assert abs(test_rx.Q_recoverable - ref_Q_recov) < 3.e-14
    assert test_rx.materials[0].flux[0] == 0.
    assert test_rx.materials[1].flux[0] == 0.
    assert abs(test_rx.materials[0].volume - volumes[1]) < 1.e-15
    assert abs(test_rx.materials[1].volume - volumes[2]) < 1.e-15
    assert test_rx.flux_level == 0.

    # Try with setting flux instead of power
    test_rx.flux_level = 10.0
    use_power = False
    Vtot = volumes[1]
    _reset_vals(test_rx)
    test_rx._update_depletion_constants(flux, nu, keff, use_power, volumes)
    assert abs(test_rx.Q_recoverable - ref_Q_recov) < 3.e-14
    assert abs(test_rx.materials[0].flux[0] - test_rx.flux_level) < 1.e-15
    assert test_rx.materials[1].flux[0] == 0.
    assert abs(test_rx.materials[0].volume - volumes[1]) < 1.e-15
    assert abs(test_rx.materials[1].volume - volumes[2]) < 1.e-15

    # Try with a zero power case (w/ flux_level in use)
    _reset_vals(test_rx)
    test_rx.flux_level = 0.
    test_rx._update_depletion_constants(flux, nu, keff, use_power, volumes)
    assert abs(test_rx.Q_recoverable - ref_Q_recov) < 3.e-14
    assert test_rx.materials[0].flux[0] == 0.
    assert test_rx.materials[1].flux[0] == 0.
    assert abs(test_rx.materials[0].volume - volumes[1]) < 1.e-15
    assert abs(test_rx.materials[1].volume - volumes[2]) < 1.e-15


def test_reactor_hdf5(depletion_lib):
    # Create a reactor with materials just as in
    # test_reactor_init_materials_and_input

    depletion_lib.to_hdf5("test_lib.h5")

    # Create a reactor as our test platform
    name = "test"
    neutronics_solver = "test"
    depletion_solver = "test"
    neutronics_exec = "test"
    depletion_exec = "test"
    base_neutronics_input_file = "sometext"
    h5_filename = "test_rx.h5"
    neutronics_library_file = "test_lib_file.txt"
    depletion_lib_file = "test_lib.h5"
    depletion_lib_name = "test"
    user_mats_info = OrderedDict()
    user_mats_info[2] = OrderedDict()
    user_mats_info[2]["name"] = "mat 2"
    user_mats_info[2]["depleting"] = False
    user_mats_info[2]["density"] = None
    user_mats_info[2]["volume"] = 1.
    user_mats_info[2]["non_depleting_isotopes"] = ["U238"]
    user_mats_info[2]["use_default_depletion_library"] = True
    user_univ_info = OrderedDict()
    shuffled_mats = set()
    shuffled_univs = set()
    use_depletion_library_xs = False
    reactivity_thresh = 1.E-8
    reactivity_thresh_init = False

    test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                            "", neutronics_exec, depletion_exec,
                            base_neutronics_input_file, h5_filename, 1, 1, 1, 1,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)
    test_rx.init_materials_and_input(neutronics_library_file,
                                     depletion_lib_file, depletion_lib_name,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs)

    # Ok, now we should write this to a file, re-read it, and compare
    # the resultant new reactor
    test_rx.update_hdf5()
    test_rx.finalize()

    new_test_rx = adder.Reactor.from_hdf5(h5_filename)

    # Now we compare to original
    assert new_test_rx.name == name
    assert new_test_rx.neutronics.solver == neutronics_solver
    assert new_test_rx.depletion.solver == depletion_solver
    assert len(new_test_rx.materials) == 2
    assert test_rx.case_label == "Initial"
    assert test_rx.operation_label == "Initial"
    assert test_rx.case_idx == 0
    assert test_rx.operation_idx == 0
    assert test_rx.step_idx == 0
    assert new_test_rx.time == 0.0
    assert new_test_rx.power == 0.0
    assert new_test_rx.flux_level == 0.0
    assert new_test_rx.Q_recoverable == 200.0
    assert new_test_rx.keff == 0.0
    assert new_test_rx.keff_stddev == 0.0
    assert new_test_rx.h5_filename == h5_filename
    assert new_test_rx.neutronics.use_depletion_library_xs == \
        use_depletion_library_xs
    assert new_test_rx.neutronics.reactivity_threshold == reactivity_thresh
    assert new_test_rx.neutronics.reactivity_threshold_initial == \
        reactivity_thresh_init
    # Not going to check the begin time, since it just uses python
    # standard libraries, just make sure it exists
    assert hasattr(new_test_rx, "begin_time")
    assert new_test_rx.end_time is None

    # Check the materials
    for i, test_mat in enumerate(test_rx.materials):
        if i == 0:
            assert test_mat.name == str(i + 1)
        else:
            assert test_mat.name == "mat 2"
        assert test_mat.density == 1.
        if i == 0:
            assert test_mat.volume is None
        else:
            assert test_mat.volume == 1.
        isotope_data = [("H1", "70c"), ["U235", "70c"], ["U238", "72c"]]
        for j, test_iso in enumerate(test_mat.isotopes):
            assert test_iso.name == isotope_data[j][0]
            assert test_iso.xs_library == isotope_data[j][1]
            if j == 0:
                # The H1 is always non-depleting
                assert test_iso.is_depleting is False
            else:
                if i == 1 and j == 2:
                    assert test_iso.is_depleting is False
                else:
                    assert test_iso.is_depleting is True

        np.testing.assert_allclose(test_mat.atom_fractions,
                                   [0.4, 0.5, 0.1], rtol=1.e-16, atol=1.e-16)
        if i == 0:
            assert test_mat.is_depleting is True
        else:
            assert test_mat.is_depleting is False
        assert test_mat.default_xs_library == "71c"
        if i == 0:
            assert test_mat.is_default_depletion_library is False
            assert test_mat.depl_lib_name is "1"
        elif i == 1:
            assert test_mat.is_default_depletion_library is True
            assert test_mat.depl_lib_name is 0
        assert test_mat.thermal_xs_libraries == []
        assert test_mat.isotopes_in_neutronics == [True, True, True]
        assert test_mat.num_isotopes == 3


def test_reactor_init_library(caplog):
    # Set the parameters we will frequently use
    name = "test"
    neutronics_solver = "test"
    depletion_solver = "test"
    mpi_cmd = "mpirun"
    neutronics_exec = "mcnp.EXE"
    depletion_exec = "origen2.EXE"
    base_neutronics_input_file = "sometext"
    h5_filename = "test.h5"
    num_threads = 2
    num_procs = 4
    chunksize = 5
    use_depletion_library_xs = True
    reactivity_thresh = 1.E-8
    reactivity_thresh_init = False

    test_rx = adder.Reactor(name, neutronics_solver, depletion_solver,
                            mpi_cmd, neutronics_exec, depletion_exec,
                            base_neutronics_input_file, h5_filename,
                            num_threads, num_threads, num_procs, chunksize,
                            use_depletion_library_xs, reactivity_thresh,
                            reactivity_thresh_init)

    # Initialize our depletion library for a single 2-isotope system
    depllib = adder.DepletionLibrary("test", np.array([0., 0.01, 1., 20.]))

    # The MAGIC nuclide will be Xe135
    magicdk = adder.DecayData(6., "hr", 0.)
    magicdk.add_type("beta-", 1., "Cs135")
    depllib.add_isotope("Xe135", decay=magicdk)

    # The dump nuclide for MAGIC is Cs135
    cs135dk = adder.DecayData(None, "s", 0.)
    depllib.add_isotope("Cs135", decay=cs135dk)

    depllib.to_hdf5("test.h5")

    # Now initialize the library
    test_rx.init_library("test.h5", "test")

    assert test_rx.depletion_libs[0].name == depllib.name

    # Now lets do a test where we perform a library init with a bad name
    test_rx.depletion_libs = {}
    caplog.clear()
    try:
        test_rx.init_library("gobledy_gook.h5", "test")
    except SystemExit:
        # Dont let the SystemExit bring us down and then we check for
        # the error we want
        last_log = caplog.record_tuples[-1]
        assert last_log == ('adder.reactor', logging.ERROR,
                            'Depletion Library gobledy_gook.h5 Was Not Found!')
    caplog.clear()
