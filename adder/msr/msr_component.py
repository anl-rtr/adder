import math

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

from adder.depletionlibrary import DecayData, DepletionLibrary
from adder.type_checker import *
import adder.data
from adder.constants import MAX_METASTABLE_STATE


class MSRComponent(object):
    """Representation of a system component in a MSR that is used to
    define the depleting fluid flowpath.

    Parameters
    ----------
    name : str
        The name of the component
    mat_name : str or None
        If this is an in-core component, this is the name of the
        adder.Material object
    mass_flowrate : float
        The mass flowrate through this component, in units of kilograms
        per second.
    density : float
        The initial density of this component, in units of g/cc. If
        mat_name is None, this will be used.
    volume : float
        The volume of this component, in units of cubic meters.
    removal_rates : Dict
        The removal rates in a dictionary with the isotope names
        (in GND format) provided as the str key, and the removal rate
        provided as the corresponding value (a float). The removal rates
        are assumed to be in units of fraction per second. Note that
        this should have been processed upstream to replace elemental
        names with isotopic constituents of that element
    in_core : bool
        Whether or not this is an in-core component (and thus is subject to
        neutron flux)
    library : DepletionLibrary
        The depletion library data, note that this should have been
        pre-processed such that the removal isotopes were already
        incorporated.

    Attributes
    ----------
    name : str
        The name of the component
    mat_name : str or None
        If this is an in-core component, this is the name of the
        adder.Material object
    mass_flowrate : float
        The mass flowrate through this component, in units of kilograms
        per second.
    volume : float
        The volume of this component, in units of cubic meters.
    removal_rates : Dict
        The removal rates in a dictionary with the isotope/element names
        (in GND format) provided as the str key, and the removal rate
        provided as the corresponding value (a float). If an element
        name is provided, then it is used for all that element's
        isotopes. If an element is provided and an isotope provided,
        the isotopic specific data is used instead of the elemental for
        that isotope only. The removal rates are assumed to be in units
        of fraction per second.
    in_core : bool
        Whether or not this is an in-core component (and thus is subject to
        neutron flux)
    library : DepletionLibrary
        The depletion library data
    variable_density : bool
        Whether or not the density of the component will vary with life,
        changing the residence time in the component
    """

    def __init__(self, name, mat_name, mass_flowrate, density, volume,
                 removal_rates, in_core, library):
        self.name = name
        self.mat_name = mat_name
        self.mass_flowrate = mass_flowrate
        self.density = density
        self.volume = volume
        self.in_core = in_core
        self.library = library
        self.removal_rates = removal_rates
        self.variable_density = False

        self._init_decay_matrix()
        # Initialize atomic mass vector as we will need it
        self.library.set_atomic_mass_vector()
        self._A_matrix = None
        self._T_matrix = None
        self._last_dt = None
        self._last_flux = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def mat_name(self):
        return self._mat_name

    @mat_name.setter
    def mat_name(self, mat_name):
        if mat_name is not None:
            check_type("mat_name", mat_name, str)
        self._mat_name = mat_name

    @property
    def mass_flowrate(self):
        return self._mass_flowrate

    @mass_flowrate.setter
    def mass_flowrate(self, mass_flowrate):
        check_type("mass_flowrate", mass_flowrate, float)
        self._mass_flowrate = mass_flowrate

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        check_type("density", density, float)
        self._density = density

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        check_type("volume", volume, float)
        self._volume = volume

    @property
    def in_core(self):
        return self._in_core

    @in_core.setter
    def in_core(self, in_core):
        check_type("in_core", in_core, bool)
        self._in_core = in_core

    @property
    def variable_density(self):
        return self._variable_density

    @variable_density.setter
    def variable_density(self, variable_density):
        check_type("variable_density", variable_density, bool)
        self._variable_density = variable_density

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, library):
        check_type("library", library, DepletionLibrary)
        self._library = library

    @property
    def removal_rates(self):
        return self._removal_rates

    @removal_rates.setter
    def removal_rates(self, removal_rates):
        check_type("removal_rates", removal_rates, dict)
        for k, v in removal_rates.items():
            # Check the key is a string and a valid GND name
            check_type("removal_rates key", k, str)
            if not adder.data.is_isotope(k):
                msg = "{} is not a valid Isotope or Element Name!".format(k)
                raise ValueError(msg)
            # Now we can check the value and ensure its a float
            # theoretically negative values are acceptable, so we wont
            # check for that
            check_type("removal_rates value", v, float)

        self._removal_rates = removal_rates

    @property
    def delta_t(self):
        if self._mass_flowrate > 0.:
            # [s] =    [m^3]    *     [g/cc] * [kg/m^3 / g/cc] / [kg/s]
            return self._volume * self._density * 1000. / self._mass_flowrate
        else:
            return np.finfo(np.float64).max

    def _init_decay_matrix(self):
        # As stated in the class docstring, the removal rates should
        # already have elements expanded to their isotopic constituents,
        # and the library should have all removal isotopes (from all)
        # components added

        # Now go through each library isotope, if it is in
        # removal_rates, incorporate the removal rate in the library
        isos_to_add = []
        for i_name in self._library.isotopes.keys():
            z, a, m = adder.data.zam(i_name)
            if m >= MAX_METASTABLE_STATE:
                # Skip over the isotopes which are our removal
                # recipients since if they are an element in
                # removal_rates, we will end up trying to remove these
                continue
            e_name = adder.data.ATOMIC_SYMBOL[z]
            # Create a removal entry for this isotope,
            # converting the removal rate to a half-life
            if e_name in self._removal_rates:
                rem = DecayData(math.log(2.) / self._removal_rates[e_name],
                                "s", 0.)
            elif i_name in self._removal_rates:
                rem = DecayData(math.log(2.) / self._removal_rates[i_name],
                                "s", 0.)
            else:
                rem = None
            if rem is not None:
                child = adder.data.gnd_name(z, a, m + MAX_METASTABLE_STATE)
                iso_data = self._library.isotopes[i_name]
                rem.add_type("removal", 1., child)
                # And assign it
                iso_data.removal = rem

                # And accrue the list of isos to add
                if child not in self._library.isotopes:
                    isos_to_add.append(child)

        if len(isos_to_add) > 0:
            stable = DecayData(None, "s", 0.)
            for iso in isos_to_add:
                # Now create the child isotope as well, it will be
                # stable
                self._library.add_isotope(iso, decay=stable)
            self._library._isotopes_ordered = False

        # Now we can build the decay matrix
        self._library.set_atomic_mass_vector()
        self.decay_matrix = self._library.build_decay_matrix()

    def init_A_matrix(self, solve_method, flux=None):
        if self._in_core and flux is None:
            msg = "A flux is required for an in-core component"
            raise ValueError(msg)

        if solve_method == "brute":
            mat_func = ss.csr_matrix
            mat_fmt = "csr"
        else:
            mat_func = ss.csc_matrix
            mat_fmt = "csc"

        if flux is None:
            self._A_matrix = mat_func(self.decay_matrix, dtype=np.float64)
        else:
            self._A_matrix = \
                self._library.build_depletion_matrix(
                    flux, matrix_format=mat_fmt, dk_matrix=self.decay_matrix)

    def init_T_matrix(self, time_step, solver_func, use_cram=True):
        dt = self.delta_t

        if dt > time_step:
            # If the mass flowrate was zero, then the fluid is
            # stationary and we need to have the residence time simply
            # be the time step length (as opposed to infinite)
            dt = time_step

        if not self._in_core:
            # Then lets see if we can re-use the T-matrix, if dt is same
            if self._last_dt is not None and self._T_matrix is not None:
                if math.isclose(dt, self._last_dt):
                    # Then the T matrix hasnt changed, dont recompute
                    return

        # Ok, guess we must compute
        if dt > 0.:
            if use_cram:
                self._T_matrix = self._T_from_cram(dt, solver_func)
            else:
                self._T_matrix = ssl.expm(self._A_matrix * dt)
        else:
            self._T_matrix = ss.eye(self.library.num_isotopes, format="csc")

        # Update our dt so we can check against this next time
        self._last_dt = dt

    def _T_from_cram(self, dt, solver_func):
        T = solver_func(self._A_matrix, np.eye(self._library.num_isotopes),
                        dt, "s")
        return ss.csc_matrix(T)

    def transmute(self, time_step, solver_func, n_in):
        # Solve the Bateman equations for this system for the brute
        # forced approach

        # Get the duration in units of days
        dt = self.delta_t
        if dt > time_step:
            # If the mass flowrate was zero, then the fluid is
            # stationary and we need to have the residence time simply
            # be the time step length (as opposed to infinite)
            dt = time_step

        return solver_func(self._A_matrix, n_in, dt, "s")

    def update_density(self, density_ratio):
        # If we cant update the density as the fluid flows (i.e., in the
        # t-matrix approach), this is used to keep the density updated
        # as feed and removal can change it
        if self._variable_density:
            self._density *= density_ratio

    def update_volume(self, delta_t, feed_v_rate):
        # Implemented only for the volume-variant Tank class. In this
        # base class, nothing will happen as the volume does not change
        pass


class MSRTank(MSRComponent):
    """Representation of an MSRComponent with varying volume and thus
    varying residence time duration."""

    def update_volume(self, delta_t, feed_v_rate):
        # deltaV = dt [s] * feed_v_rate [cc/s]
        # This yields deltaV in terms of cm^3. To convert to m^3, we
        # multiply by 1E-6.
        # Therefore V has units of m^3
        self.volume += delta_t * feed_v_rate * 1e-6
