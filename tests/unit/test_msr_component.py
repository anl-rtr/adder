import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import pytest

from adder.msr import MSRComponent, MSRTank


def test_comp_init(depletion_lib):
    # Tests the init method of MSRComponent
    # No need to test MSRTank here since it uses MSRComponent's function

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.    # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")
    amu_vec = np.array([1.00782503, 4.00260325, 16.00610192, 15.99491462,
                        231.03630285, 235.04392819, 238.050787, 235.04392819])
    decay_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., -2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 0., 0., -1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.]])

    # Check the type and value checks of each of the input parameters
    with pytest.raises(TypeError):
        test_c = MSRComponent(1, mat_name, mass_flowrate, density, vol,
                              removal_rates, in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, 1, mass_flowrate, density, vol,
                              removal_rates, in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, "1", density, vol,
                              removal_rates, in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, mass_flowrate, "1", vol,
                              removal_rates, in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, mass_flowrate, density, "vol",
                              removal_rates, in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                              'removal_rates', in_core, lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                              removal_rates, "in_core", lib)
    with pytest.raises(TypeError):
        test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                              removal_rates, in_core, "lib")

    # Check that the attributes exist and their values are set correctly
    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)
    assert test_c.name == name
    assert test_c.mat_name == mat_name
    assert test_c.mass_flowrate == mass_flowrate
    assert test_c.density == density
    assert test_c.volume == vol
    assert test_c.removal_rates == removal_rates
    assert test_c.in_core == in_core
    assert test_c.library == lib
    assert test_c._A_matrix is None
    assert test_c._T_matrix is None
    assert test_c._last_dt is None
    assert test_c._last_flux is None
    np.testing.assert_allclose(test_c.library.atomic_mass_vector, amu_vec)
    np.testing.assert_allclose(test_c.decay_matrix, decay_mat)

    # Check the properties
    assert test_c.delta_t == (vol * density * 1000 / mass_flowrate)
    # now set the mass_flowrate to 0 and make sure we get a large #
    test_c.mass_flowrate = 0.
    assert test_c.delta_t > 1E30


def test_comp_init_A_matrix(depletion_lib):
    # Tests the init method of MSRComponent
    # No need to test MSRTank here since it uses MSRComponent's function

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.    # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")
    decay_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., -2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 0., 0., -1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.]])
    flux_const = 0.5
    flux = np.array([flux_const])
    xs_mat = np.zeros_like(decay_mat)
    # Add in the 2 fission x/s
    i_235, i_1, i_16 = 5, 0, 3
    xs_mat[i_235, i_235] = -flux_const * 100.e-24
    xs_mat[i_1, i_235] = flux_const * 0.5 * 100.e-24
    xs_mat[i_16, i_235] = flux_const * 0.5 * 100.e-24
    i_238, i_1, i_16 = 6, 0, 2
    xs_mat[i_238, i_238] = -flux_const * 10.e-24
    xs_mat[i_1, i_238] = flux_const * 1. * 10.e-24
    xs_mat[i_16, i_238] = flux_const * 1. * 10.e-24

    A_mat = decay_mat + xs_mat

    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)

    # Without flux, and with an in-core component should receive error
    with pytest.raises(ValueError):
        test_c.init_A_matrix("brute")
    with pytest.raises(ValueError):
        test_c.init_A_matrix("tmatrix")

    # Now on to the cases that will actually run
    # First just test brute decay-only matrix
    test_c.in_core = False
    test_c.init_A_matrix("brute")
    assert isinstance(test_c._A_matrix, ss.csr_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), decay_mat)

    # Now t-matrix decay-only matrix
    test_c.init_A_matrix("tmatrix")
    assert isinstance(test_c._A_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), decay_mat)

    # As a simple test of including xs, use a 0 flux for both
    test_c.in_core = True
    test_c.init_A_matrix("brute", flux=np.zeros_like(flux))
    assert isinstance(test_c._A_matrix, ss.csr_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), decay_mat)

    # Now t-matrix
    test_c.init_A_matrix("tmatrix", flux=np.zeros_like(flux))
    assert isinstance(test_c._A_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), decay_mat)

    # Now include the flux
    test_c.init_A_matrix("brute", flux=flux)
    assert isinstance(test_c._A_matrix, ss.csr_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), A_mat)

    # Now t-matrix
    test_c.init_A_matrix("tmatrix", flux=flux)
    assert isinstance(test_c._A_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._A_matrix.todense()), A_mat)


def test_comp_init_T_matrix(depletion_lib):
    # Tests the init method of MSRComponent
    # No need to test MSRTank here since it uses MSRComponent's function

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.    # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")
    decay_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., -2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 0., 0., -1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.]])
    flux_const = 0.5
    flux = np.array([flux_const])
    xs_mat = np.zeros_like(decay_mat)
    # Add in the 2 fission x/s
    i_235, i_1, i_16 = 5, 0, 3
    xs_mat[i_235, i_235] = -flux_const * 100.e-24
    xs_mat[i_1, i_235] = flux_const * 0.5 * 100.e-24
    xs_mat[i_16, i_235] = flux_const * 0.5 * 100.e-24
    i_238, i_1, i_16 = 6, 0, 2
    xs_mat[i_238, i_238] = -flux_const * 10.e-24
    xs_mat[i_1, i_238] = flux_const * 1. * 10.e-24
    xs_mat[i_16, i_238] = flux_const * 1. * 10.e-24

    A_mat = decay_mat + xs_mat

    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)

    # Build the A matrix
    test_c.init_A_matrix("tmatrix", flux=flux)

    # And now we can test init_T_matrix
    # Will have to do:
    # 1. A standard case
    # 2. long dt compared to time_step
    # 3. 0 dt to get an identity matrix
    # Note ref solns will just be from ssl.expm, since validation cases
    # will verify the analytical solutions
    def dummy(mat, vec, dt, units):
        return ssl.expm(mat * dt)

    # 1. Standard case
    dt = (vol * density * 1000 / mass_flowrate)
    time_step = 1e8
    ref_T = ssl.expm(A_mat * dt)
    test_c.init_T_matrix(time_step, dummy)
    assert isinstance(test_c._T_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._T_matrix.todense()), ref_T)

    # 2. Long dt compared to time_step
    time_step = 10.
    ref_T = ssl.expm(A_mat * time_step)
    test_c.init_T_matrix(time_step, dummy)
    assert isinstance(test_c._T_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._T_matrix.todense()), ref_T)

    # 3. 0 dt to get identity matrix
    time_step = 0.
    ref_T = np.eye(A_mat.shape[0])
    test_c.init_T_matrix(time_step, dummy)
    assert isinstance(test_c._T_matrix, ss.csc_matrix)
    np.testing.assert_allclose(np.array(test_c._T_matrix.todense()), ref_T)


def test_comp_transmute(depletion_lib):
    # Tests the init method of MSRComponent
    # No need to test MSRTank here since it uses MSRComponent's function

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.    # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")
    decay_mat = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 2., 0., 0., 0., 0.],
                          [0., 0., 0., -2., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.],
                          [0., 0., 0., 0., 0., -1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.5, 0., 0.]])
    flux_const = 0.5
    flux = np.array([flux_const])
    xs_mat = np.zeros_like(decay_mat)
    # Add in the 2 fission x/s
    i_235, i_1, i_16 = 5, 0, 3
    xs_mat[i_235, i_235] = -flux_const * 100.e-24
    xs_mat[i_1, i_235] = flux_const * 0.5 * 100.e-24
    xs_mat[i_16, i_235] = flux_const * 0.5 * 100.e-24
    i_238, i_1, i_16 = 6, 0, 2
    xs_mat[i_238, i_238] = -flux_const * 10.e-24
    xs_mat[i_1, i_238] = flux_const * 1. * 10.e-24
    xs_mat[i_16, i_238] = flux_const * 1. * 10.e-24

    A_mat = decay_mat + xs_mat

    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)

    # Build the A matrix
    test_c.init_A_matrix("brute", flux=flux)

    # Now we can transmute with a dummy solver
    def dummy(mat, vec, dt, units):
        return dt * vec

    # Do first with normal dt
    n_out = test_c.transmute(1e8, dummy, np.ones(A_mat.shape[0]))
    np.testing.assert_allclose(n_out, 20. * np.ones(A_mat.shape[0]))

    # And time step < dt
    n_out = test_c.transmute(10., dummy, np.ones(A_mat.shape[0]))
    np.testing.assert_allclose(n_out, 10. * np.ones(A_mat.shape[0]))


def test_comp_update_density(depletion_lib):
    # Tests the init method of MSRComponent
    # No need to test MSRTank here since it uses MSRComponent's function

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.    # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")

    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)

    assert test_c.density == density

    # This next one should do nothing since test_c.variable_density is False
    test_c.update_density(20.)
    assert test_c.density == density

    # Now do it while allowing variable density
    test_c.variable_density = True
    test_c.update_density(20.)
    assert test_c.density == 20. * density


def test_comp_update_volume(depletion_lib):
    # Tests the init method of MSRComponent

    # set thee default parameters we will use
    name = "comp 1"
    mat_name = "mat 1"
    mass_flowrate = 1000.   # kg/s
    density = 2.   # g/cc
    vol = 10.   # m^3
    removal_rates = {"U235": 0.5}
    in_core = True
    lib = depletion_lib.clone(new_name="clone")
    feed_density = 2.   # g/cc

    test_c = MSRComponent(name, mat_name, mass_flowrate, density, vol,
                          removal_rates, in_core, lib)

    assert test_c.volume == vol

    # As this is not a tank, this should do nothing
    test_c.update_volume(20., 10. * 1e3 / feed_density)
    assert test_c.volume == vol

    # Now repeat, with a tank
    lib = depletion_lib.clone(new_name="clone")
    test_c = MSRTank(name, mat_name, mass_flowrate, density, vol,
                     removal_rates, in_core, lib)

    assert test_c.volume == vol
    assert test_c.delta_t == 20.

    # Now we have a tank, we should see the volume grow
    test_c.update_volume(20., 10. * 1e3 / feed_density)
    new_vol = vol + 20. * 10. / (2000.)
    assert test_c.volume == new_vol

    # We should also see delta_t grow proportionally
    assert test_c.delta_t == 20.0 * 1.01
