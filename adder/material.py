from copy import deepcopy
from collections import OrderedDict

import numpy as np
import h5py

from adder.data import atomic_mass
from adder.depletionlibrary import DecayData, ReactionData
from adder.isotope import *
import adder.constants as constants
from adder.type_checker import *
from adder.utils import get_id


class Material(object):
    """This class contains relevant information about a material.

    Parameters
    ----------
    name : str
        The name of the material.
    id_ : int
        The identifier of the material, as referred to in the neutronics
        solver.
    density : float
        The material density in units of a/b-cm
    isotope_data : Iterable of tuples
        This contains the arguments for the Isotope objects for each
        isotope in the material.
    atom_fractions : Iterable of float
        The fractions of the density of the constituent isotopes.
    is_depleting : bool
        Whether or not this material is to be depleted.
    default_xs_library : str
        The default xs library to apply to isotopes as introduced by
        depletion or otherwise.
    num_groups : int
        The number of groups in the depletion library, and thus the
        size of the flux array
    thermal_xs_libraries : str
        This provides the name of the associated thermal scattering
        library, if needed. An empty list indicates no thermal data is
        assigned to this material.
    status : {constants.IN_CORE, constants.STORAGE, or constants.SUPPLY}
        Whether the material is in-core, from the supply chain or in
        storage.
    check : bool, optional
        This flag is only intended for use for testing purposes. When set to
        True the Material.check(...) method is called and the Material is
        checked for correctness. When False the method is not called.

    Attributes
    ----------
    name : str
        The name of the material.
    id : int
        The identifier of the material, as referred to in the neutronics
        solver.
    is_depleting : bool
        Whether or not this material is to be depleted.
    density : float
        The material density in units of a/b-cm
    isotopes : Iterable of adder.Isotopes
        The isotopes within the material.
    atom_fractions : Iterable of float
        The fractions of the density of the constituent isotopes.
    number_densities : numpy.ndarray
        The number densities of each isotope
    volume : float
        The volume of the material in units of cm^3.
    status : {constants.IN_CORE, constants.STORAGE, or constants.SUPPLY}
        Whether the material is in-core, from the supply chain or in
        storage.
    flux : np.ndarray of float
        The group-wise flux
    default_xs_library : str
        The default xs library to apply to isotopes as introduced by
        depletion or otherwise.
    num_groups : int
        The number of groups in the depletion library, and thus the
        size of the flux array
    thermal_xs_libraries : List of str.
        This provides the names of the associated thermal scattering
        libraries, if needed. Defaults to a blank list, indicating no
        thermal data is assigned to this material.
    isotopes_in_neutronics : Iterable of bool
        Whether or not the neutronics solver should incorporate each
        isotope (indexed the same as isotopes, atom_fraction, etc.).
    depl_lib_name : int or str
        The name of the library to use with this material
    isotopes_to_keep_in_model : set of str
        The set of isotope names which will not be subject to the reactivity
        threshold determination. This is initialized empty and is set
        with the establish_initial_isotopes method.
    """

    _USED_IDS = set([])

    def __init__(self, name, id_, density, isotope_data,
                 atom_fractions, is_depleting, default_xs_library,
                 num_groups, thermal_xs_libraries, status, check=True):
        self.name = name
        self.id = id_
        self.density = density
        self.isotopes = [isotope_factory(*data) for data in isotope_data]
        self.atom_fractions = atom_fractions
        self.is_depleting = is_depleting
        # Volumes will be over-written after the neutronics solver runs
        self.volume = None
        self.status = status
        self.is_default_depletion_library = False
        self.depl_lib_name = constants.BASE_LIB
        self.num_groups = num_groups
        self.flux = np.zeros(self.num_groups)
        self.default_xs_library = default_xs_library
        self.thermal_xs_libraries = thermal_xs_libraries
        self.isotopes_in_neutronics = True
        self.Q = 0.
        self.num_copies = 0
        self.isotopes_to_keep_in_model = set()
        # _h5_path is the path of the previous location in the HDF5 file
        # (if any) that this material was printed. h5_status_change is a flag
        # to denote if the state has changed since then. These are
        # used to determine if there is a reason to explicitly re-write the
        # material to the HDF5 file, or if a soft-link will suffice.
        # If a soft-link is needed. This saves results writing time and space.
        # The state change will come from depletion or a change in volume
        self._h5_path = None
        self.h5_status_change = True
        # self.logs contains log messages as a 3-tuple of:
        # (log type, msg, None or # of indents). This is then called by
        # LoggedClass.update_logs by a class that inherits from LoggedClass
        # like Reactor or Neutronics. This is done because parallelized
        # objects (as Materials definitely will be) best not alter system state
        # which a log file inherently is. Therefore the Materials have no
        # direct logging capability but we still want them to have an ability
        # to write logs in the same format.
        self.logs = []

        if check:
            self.check()

    def __getstate__(self):
        selfdict = self.__dict__.copy()
        nope = ['_id', '_is_depleting', '_volume', '_status',
            '_depl_lib_name', '_num_groups', '_is_default_depletion_library',
            '_thermal_xs_libraries', '_isotopes_in_neutronics', 'Q',
            'num_copies', '_h5_path', 'h5_status_change', 'logs', '_USED_IDS']
        for key in nope:
            try:
                del selfdict[key]
            except:
                pass
        return selfdict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def depl_lib_name(self):
        return self._depl_lib_name

    @depl_lib_name.setter
    def depl_lib_name(self, depl_lib_name):
        check_type("depl_lib_name", depl_lib_name, (int, str))
        self._depl_lib_name = depl_lib_name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if id_ is None:
            # Then we have to auto-generate an ID, use our generalized function
            self._id = get_id(Material._USED_IDS, constants.MATL_MAX_ID)
        else:
            check_type("id", id_, int)
            check_greater_than("id", id_, 0, equality=False)
            check_less_than("id", id_, constants.MATL_MAX_ID, equality=True)
            self._id = id_
            Material._USED_IDS.add(self._id)

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        check_type("density", density, float)
        check_greater_than("density", density, 0., equality=True)
        self._density = density

    @property
    def neutronics_density(self):
        # This returns the density for the material without the isotopes
        # that will not make it to the neutronics model.
        # This will be in units of atom/b-cm
        rho = np.sum(self.number_densities[self.isotopes_in_neutronics])
        return rho * 1.E-24

    @property
    def mass_density(self):
        # Returns the density of this material in units of g/cc
        n = self.number_densities
        n_dot_amu = 0.
        for i in range(len(n)):
            n_dot_amu += n[i] * atomic_mass(self._isotopes[i].name)
        mass_density = n_dot_amu / constants.AVOGADRO
        return mass_density

    @property
    def isotopes(self):
        return self._isotopes

    @isotopes.setter
    def isotopes(self, isotopes):
        check_iterable_type("isotopes", isotopes, Isotope)
        self._isotopes = isotopes

    @property
    def atom_fractions(self):
        return self._atom_fractions

    @atom_fractions.setter
    def atom_fractions(self, atom_fractions):
        check_iterable_type("atom_fractions", atom_fractions, float)
        check_length("atom_fractions", atom_fractions, self.num_isotopes)
        for val in atom_fractions:
            check_greater_than("atom_fractions", val, 0., equality=True)
        # Modify the atom fractions to sum to 1.0
        tot_frac = np.sum(atom_fractions)
        norm_frac = [user_frac / tot_frac for user_frac in atom_fractions]
        self._atom_fractions = norm_frac[:]

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        check_value("status", status, constants.ALLOWED_STATUSES)
        self._status = status

    @property
    def default_xs_library(self):
        return self._default_xs_library

    @default_xs_library.setter
    def default_xs_library(self, default_xs_library):
        check_type("default_xs_library", default_xs_library, str)
        self._default_xs_library = default_xs_library

    @property
    def thermal_xs_libraries(self):
        return self._thermal_xs_libraries

    @thermal_xs_libraries.setter
    def thermal_xs_libraries(self, thermal_xs_libraries):
        # Since check_iterable_type allows a string as an iterable,
        # check to ensure it is *not* a string first
        if isinstance(thermal_xs_libraries, str):
            msg = 'Unable to set "thermal_xs_libraries" ' \
                'which is not of type "str"'
            raise TypeError(msg)
        # Ok, it isnt a string, carry on
        check_iterable_type("thermal_xs_libraries", thermal_xs_libraries, str)
        self._thermal_xs_libraries = thermal_xs_libraries

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        if volume is not None:
            check_type("volume", volume, float)
            check_greater_than("volume", volume, 0., equality=True)
        self._volume = volume
        # Update the hdf5 status change flag
        self.h5_status_change = True

    @property
    def is_depleting(self):
        return self._is_depleting

    @is_depleting.setter
    def is_depleting(self, is_depleting):
        check_type("is_depleting", is_depleting, bool)
        self._is_depleting = is_depleting

    @property
    def is_default_depletion_library(self):
        return self._is_default_depletion_library

    @is_default_depletion_library.setter
    def is_default_depletion_library(self, is_default_depletion_library):
        check_type("is_default_depletion_library",
                   is_default_depletion_library, bool)
        self._is_default_depletion_library = is_default_depletion_library

    @property
    def isotopes_in_neutronics(self):
        return self._isotopes_in_neutronics

    @isotopes_in_neutronics.setter
    def isotopes_in_neutronics(self, isotopes_in_neutronics):
        if isinstance(isotopes_in_neutronics, bool):
            self._isotopes_in_neutronics = \
                [isotopes_in_neutronics] * self.num_isotopes
        else:
            check_iterable_type("isotopes_in_neutronics",
                                isotopes_in_neutronics, bool)
            check_length("isotopes_in_neutronics", isotopes_in_neutronics,
                         self.num_isotopes)
            self._isotopes_in_neutronics = isotopes_in_neutronics[:]

    @property
    def num_groups(self):
        return self._num_groups

    @num_groups.setter
    def num_groups(self, num_groups):
        check_type("num_groups", num_groups, int)
        self._num_groups = num_groups

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, flux):
        check_iterable_type("flux", flux, float)
        self._flux = flux
        self._flux_1g = np.sum(flux)
        self.h5_status_change = True

    @property
    def flux_1g(self):
        return self._flux_1g

    @property
    def number_densities(self):
        N = self._density * np.asarray(self._atom_fractions) * 1.E24
        return N

    @property
    def num_isotopes(self):
        return len(self._isotopes)

    @property
    def isotope_names(self):
        return [iso.name for iso in self._isotopes]

    def __repr__(self):
        return "<Material Name: {}, Id: {}, is_depleting: {}>".format(
            self._name, self._id, self._is_depleting)

    def check(self):
        """This method adds log messages relating to the completeness of the
        material definition at the end of initialization."""

        # Check that, if depleting, a default xs library is defined
        if self._is_depleting and not \
                (self._default_xs_library
                    and self._default_xs_library.strip()):
            # Then the string is empty or blank. This is the best we can do
            # without knowledge of valid xslib identifiers which is code
            # dependent.
            msg = f'Material {self._name} (id: {self._id}) is depleting ' \
                'without a default cross section library defined!'
            self.logs.append(("error", msg, None))

        # Make sure we have no 0-fraction isotopes
        to_remove = []
        for i in range(self.num_isotopes):
            if self._atom_fractions[i] == 0.:
                to_remove.append(i)
        for i in to_remove:
            msg = f'{self._isotopes[i].name} in Material {self._name} ' \
                f'(id: {self._id}) was removed as it has a zero atom fraction'
            self.logs.append(("info_file", msg, None))
            self.remove_isotope_by_index(i)

    def remove_isotope_by_index(self, idx):
        # Remove an isotope by deleting the entries in:
        # isotopes, atom_fractions, and isotopes_in_neutronics
        for items in (self._isotopes, self._atom_fractions,
                      self._isotopes_in_neutronics):
            if isinstance(items, list):
                items.pop(idx)
            else:
                raise NotImplementedError(
                    "remove_isotope_by_index "
                    "currently only implemented for lists")

    def establish_initial_isotopes(self, apply_threshold_to_initial):
        """Sets the list of isotopes that should not be subject to the
        reactivity threshold check based on the apply_threshold_to_initial
        input.
        """

        if not apply_threshold_to_initial:
            # Then store the isotope names in isotopes_to_keep_in_model
            self.isotopes_to_keep_in_model.update(self.isotope_names)

    def update_isotope_is_depleting(self, lib):
        """Updates the status of each isotope to reflect if it is
        depleting or not based on the information available in the
        depletion library. Specifically, if an isotope is not in the
        depletion library, then it is not depleting. This is primarily
        intended to ensure elements aren't depleting unless the library
        has otherwise included the element in the decay path, but it
        should work fine for all cases (i.e., isotopes too) with the
        same logic.
        """

        # If this material is not depleting, do nothing
        if not self._is_depleting:
            return

        # If we make it here, then we have the information we need
        stable = DecayData(None, "s", 0.)
        zero_xs = ReactionData("b", lib.num_neutron_groups)
        for i, iso in enumerate(self._isotopes):
            if iso.name not in lib.initial_isotopes:
                # The isotope is not in the depletion library, flag it as
                # non-depleting
                msg = "Setting Isotope {} ".format(iso.name) + \
                    "in Material {} to non-depleting ".format(self._name) + \
                    "since it is not in the depletion library"
                self.logs.append(("info_file", msg, None))
                self._isotopes[i] = update_isotope_depleting_status(iso, False)

                if iso.name not in lib.isotopes:
                    # If this isotope wasnt already added to the list of isos,
                    # then do it witha stable, 0 cross section version of
                    # the isotope in the library so that indexing works
                    # later on.
                    lib.add_isotope(iso.name, xs=zero_xs, decay=stable)
        # Since we may have added isotopes, update the indices
        if lib.num_isotopes != len(lib.isotope_indices):
            lib.set_isotope_indices()

    def clone(self, depl_libs, new_name=None):
        """Create a copy of this material with a new unique ID, updated
        name.

        Parameters
        ----------
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        new_name : None or str, optional
            A specific name to apply to this; if None, then an
            incremented number will be added to the end. e.g., "name"
            will become "name_1", and "name_1" would become "name_2".

        Returns
        -------
        clone : adder.Material
            The clone of this material

        """

        # Get the init parameters from self except for name, and id_
        density = deepcopy(self._density)
        if self._is_depleting:
            # Spend the memory and make copies of the isotopic data as it
            # will change in time
            isotopes = np.empty(self.num_isotopes, dtype=object)
            for i, iso in enumerate(self._isotopes):
                isotopes[i] = (iso.name, iso.xs_library, iso.is_depleting)
            atom_fractions = np.copy(self._atom_fractions)
            is_depleting = True
            flux = np.copy(self._flux)
            Q = deepcopy(self.Q)
            iso_in_neut = deepcopy(self._isotopes_in_neutronics)
        else:
            # Save the memory and only assign references from the original
            # as these will be constant in time. We also want to save the time
            # of creating the isotopes array by making it blank for now and
            # then overwriting later with just a reference vice a copy
            isotopes = []
            atom_fractions = []
            is_depleting = False
            flux = self._flux
            Q = self.Q
            iso_in_neut = self._isotopes_in_neutronics
        default_xs_library = deepcopy(self._default_xs_library)
        num_groups = deepcopy(self._num_groups)
        thermal_xs_libraries = deepcopy(self._thermal_xs_libraries)
        status = deepcopy(self._status)

        # The isotopes_to_keep_in_model parameter will be constant in time so
        # lets also not make copies of it
        isotopes_to_keep_in_model = self.isotopes_to_keep_in_model

        # Set the new id
        id_ = None

        # Update the number of clones so we can keep our names up to date
        self.num_copies += 1

        # Now modify the name
        if new_name is None:
            name = self._name + "[{}]".format(self.num_copies)
        else:
            name = new_name

        clone = Material(name, id_, density, isotopes, atom_fractions,
                         is_depleting, default_xs_library, num_groups,
                         thermal_xs_libraries, status)

        if not clone._is_depleting:
            # Then we have to go back and add back in the references to the
            # isotopic data
            clone._isotopes = self._isotopes
            clone._atom_fractions = self._atom_fractions

        # Update the other information
        clone._volume = deepcopy(self._volume)
        clone._is_default_depletion_library = \
            deepcopy(self._is_default_depletion_library)
        if clone._is_depleting and not clone._is_default_depletion_library:
            # Make a copy of the library as this will see differing
            # flux spectra (if it has been initialized so far)
            new_lib = depl_libs[self._depl_lib_name].clone(
                new_name=clone._name)
            depl_libs[new_lib.name] = new_lib
            clone._depl_lib_name = new_lib.name
        else:
            clone._depl_lib_name = self._depl_lib_name
        clone._flux = flux
        clone.isotopes_in_neutronics = iso_in_neut
        clone.Q = Q
        clone.isotopes_to_keep_in_model = isotopes_to_keep_in_model
        clone.logs = []

        return clone

    def compute_Q(self, depl_lib):
        """Computes the total energy release rate and the total fission
        rate for this material.

        This is used when determining the reactor's average recoverable
        energy per fission.

        This calculation implicitly makes the assumption that only
        materials that are depleting are producing our fissions and thus
        non-depleting materials are ignored in the Q determination.

        The calculation also relies upon the ORIGEN2.2 correlation for
        the recoverable energy per fission for actinides.

        Parameters
        ----------
        depl_lib : DepletionLibrary
            The depletion library to use

        Returns
        -------
        tot_Q : float
            The total energy release rate for this material
        tot_fiss_rate :
            The total fission rate for this material
        """

        tot_Q = 0.
        tot_fiss_rate = 0.

        # If this doesnt exist in the model, we just want to return
        # our zeros
        if not self._is_depleting:
            return tot_Q, tot_fiss_rate

        Q, FR = depl_lib.get_composition_Q_fiss_rate(self.isotope_names,
                                                     self._atom_fractions,
                                                     self._flux)

        tot_Q += self._density * self._volume * Q
        tot_fiss_rate += self._density * self._volume * FR

        return tot_Q, tot_fiss_rate

    def determine_important_isotopes(self, depl_lib, rho_threshold=1.E-8):
        """Set the :attrib:`isotopes_in_neutronics` attribute which
        defines which isotopes contribute a non-negligible reactivity
        effect.

        This is performed by performing a one-group perturbation
        theory computation to determine each isotopes contribution to
        the eigenvalue of an infinite homogeneous medium of this
        material. The one-group data is obtained from however many
        groups is contained in the depletion library by using the flux
        from the most recent neutronics solve.

        Parameters
        ----------
        depl_lib : DepletionLibrary
            The depletion library to use
        rho_threshold : float, optional
            The reactivity threshold for which any isotopes which
            contribute less than this to the infinite homogeneous
            medium's reactivity are not included in the neutronics
            computation. Defaults to 1.E-5 (1 pcm).
        """
        # Check the types
        check_type("rho_threshold", rho_threshold, float)

        # Pre-set all isotopes to be included, we will deviate from here only
        # as needed
        self.isotopes_in_neutronics = True

        # Now if the user doesnt want to filter, dont.
        if rho_threshold == 0.:
            return

        # Now if the flux is 0 (i.e., after a 0-power depletion or this just
        # came from storage) and its a multigroup problem, then we will take
        # minimal efforts to filter
        # If its 1-group, then we can still perform this work as a unit flux
        # is sufficient to compute the reactivity worth of the isotopes since
        # we arent actually doing group weighting.
        if self._flux_1g <= 0. and self.num_groups > 1:
            # We will only preclude the isotopes of the smallest concentrations
            for i in range(len(self._isotopes)):
                if self._atom_fractions[i] < 1.E-10 and \
                        (self._isotopes[i].name
                         not in self.isotopes_to_keep_in_model):
                    self._isotopes_in_neutronics[i] = False
            return
        elif self._flux_1g <= 0.:
            # Then this a one-group case that just has a 0 flux.
            # Since a 1-grp problem has no need of flux-weighting, we can
            flux = np.ones(1)
        else:
            # Then this is a case of any groups that has a flux
            flux = self._flux

        lib = depl_lib
        # delta_rhos is the eigenvalue change per isotope for each
        # isotope
        delta_rhos = np.zeros(len(self._isotopes))
        # Step through each isotope and determine delta_rho
        iso_names = [iso.name for iso in self._isotopes]
        N = self.number_densities
        base_nufiss = lib.get_1g_macro_xs(iso_names, N, "nu-fission", flux)
        base_abs = lib.get_1g_macro_xs(iso_names, N, "absorb", flux)
        base_kinf = base_nufiss / base_abs

        for i, iso_name in enumerate(iso_names):
            conc = N[i]
            iso_abs = conc * lib.get_1g_micro_xs(iso_name, "absorb", flux)

            if base_kinf > 0.:
                # Then this has fissile material and we can look for a
                # reactivity perturbation
                iso_nufiss = conc * lib.get_1g_micro_xs(iso_name, "nu-fission",
                                                        flux)
                # Figure out what k-infinite is without this isotope
                if base_abs - iso_abs != 0.:
                    kinf = (base_nufiss - iso_nufiss) / (base_abs - iso_abs)
                    if kinf != 0.:
                        delta_rhos[i] = np.abs((base_kinf - kinf) / kinf)
                else:
                    # Then we definitely need to keep this bad boy as
                    # it contains all of the absorption
                    delta_rhos[i] = 1.
                # Otherwise, this is an isotope w/ no xs data, and is
                # an isotope considered to have no absorptive property
                # so leaving delta_rhos[i] = 0 will remove it from
                # the neutronics calc.

            else:
                # This material has no fissile inventory and we should just
                # look at the fractional worth of the absorption
                delta_rhos[i] = iso_abs / base_abs

        # We now have all the reactivity impact of removing the isotopes
        # We want to allow up to threshold total reactivity impact
        # So sort these delta_rhos from small to large, the isotopes
        # which sum to less than threshold, we can safely neglect
        sorted_indices = np.argsort(delta_rhos)
        # Get the sorted cumulative sum
        cumulative_sum = np.cumsum(delta_rhos[sorted_indices])
        # Find where the cumulative sum crosses the threshold
        threshold_index = np.searchsorted(cumulative_sum, rho_threshold)

        # Now we can set our isotopes_in_neutronics values to false
        for neglect_index in sorted_indices[:threshold_index]:
            # But dont neglect isotopes which we do not deplete
            # (otherwise they wouldnt be in the model), or those which are
            # identified as to be kept in the model
            if self._isotopes[neglect_index].is_depleting and \
                (self._isotopes[neglect_index].name
                 not in self.isotopes_to_keep_in_model):
                self._isotopes_in_neutronics[neglect_index] = False

    def get_library_number_density_vector(self, iso_indices):
        """Gets the number density array, ordered as it would be needed
        for depletion matrix operations.

        Parameters
        ----------
        iso_indices : dict
            The indices of the A_matrix, keyed by the isotope name
        Returns
        -------
        n_vector : np.ndarray
            The number density vector, ordered as the depletion library
            expects
        """

        n_vector = np.zeros(len(iso_indices))

        mat_N = self.number_densities
        for i, iso in enumerate(self._isotopes):
            if iso.is_depleting:
                # Then we include it so its effect on the decay chain is
                # captured
                n_vector[iso_indices[iso.name]] = mat_N[i]

        return n_vector

    def calc_composition_from_number_densities(self, n_vector, iso_indices,
        inv_iso_indices, scale_constant=None):
        """Computes a new material inventory based on a number density
        returned from a depletion calculation using this depletion
        library.

        The difference between this and Material.update_from_number_densities
        is that this does not change the environment's state, as is needed for
        parallel processing.

        Parameters
        ----------
        n_vector : np.ndarray
            The number density vector itself
        iso_indices : dict
            The indices of the A_matrix, keyed by the isotope name
        inv_iso_indices : dict
            The inverse of iso_indices, where the key is the index
        scale_constant : None or float, optional
            If scale_constant is None, then the number density from
            n_vector will directly be applied to this material. If a
            value is provided, then the density will instead be
            scaled by this value.

        Returns
        -------
        new_isotopes : Iterable of 3-tuple (str, str, bool)
            A list of the isotope name (str), the xs library (str), and whether
            it is depleting (bool). There is one of these 3-tuples per isotope.
        new_fractions : np.ndarray
            A 1-D vector containing the atom fractions for each of the isotopes
            in new_isos
        new_density : float
            The new material density in units of a/b-cm

        """
        # Make a copy as may need to modify n_vector
        n_in = n_vector.copy()

        # Get the attributes of the starting material
        original_iso_metadata = OrderedDict()
        original_N = self.number_densities
        non_depleting_indices = []
        non_depleting_non_library = []
        for i, isotope in enumerate(self._isotopes):
            name = isotope.name
            # Using the same loop we will also create a dictionary of the
            # xs_libraries so we can re-set these later
            original_iso_metadata[name] = (isotope.xs_library,
                                           isotope.is_depleting, original_N[i])
            if not isotope.is_depleting:
                if name in iso_indices:
                    non_depleting_indices.append(iso_indices[name])
                else:
                    non_depleting_non_library.append(name)

        iso_indices = set(np.where(n_in > 0.)[0].tolist() +
                          non_depleting_indices)
        new_isotopes = [None] * len(iso_indices)
        new_fractions = np.zeros(len(iso_indices))
        new_density = 0.

        j = 0
        # First deal with all the isotopes that are in the library
        # (depleting or not)
        for i in iso_indices:
            iso_name = inv_iso_indices[i]

            # We want to use the same xs library as the nuclide had
            # originally, so if this isnt a new isotope, use the
            # previous xs library
            if iso_name in original_iso_metadata:
                xs_library, is_depleting, orig_N = \
                    original_iso_metadata[iso_name]
            else:
                # Then this is new so use the default
                xs_library = self._default_xs_library
                is_depleting = True
                orig_N = None
            new_isotopes[j] = (iso_name, xs_library, is_depleting)

            # Now, if this isotope was not depleting, we need to get
            # back the original value
            if not is_depleting:
                n_in[i] = orig_N

            # Get the fractions, we will normalize it later
            new_fractions[j] = n_in[i]

            new_density += n_in[i]
            j += 1

        # And now repeat for those isotopes that are not depleting
        # whether it be bc they are not in the library or otherwise
        for iso_name in non_depleting_non_library:
            # Get the starting info
            xs_library, is_depleting, orig_N = original_iso_metadata[iso_name]
            new_isotopes.append((iso_name, xs_library, is_depleting))
            new_fractions = np.append(new_fractions, orig_N)
            new_density += orig_N

        # Assign the new_fractions while accruing n_in and new_density
        new_fractions /= new_density
        new_density *= 1.E-24

        if scale_constant is not None:
            # Then the density change is scaled with a reference density being
            # the original density; this is useful for getting the same
            # atom fractions across materials but having different densities
            # from different temperatures, for example.
            new_density = scale_constant * self._density

        return new_isotopes, np.clip(new_fractions, 0., None), new_density

    def apply_new_composition(self, new_isos, new_fracs, new_density):
        """Applies the composition data from
        calc_composition_from_number_densities to the Material object itself.

        This is intended to be run AFTER the parallel depletion operations as
        this changes the Material's state and the global Isotope state via the
        isotope factory call.

        Parameters
        ----------
        new_isos : Iterable of 3-tuple (str, str, bool)
            A list of the isotope name (str), the xs library (str), and whether
            it is depleting (bool). There is one of these 3-tuples per isotope.
        new_fracs : np.ndarray
            A 1-D vector containing the atom fractions for each of the isotopes
            in new_isos
        new_density : float
            The new material density in units of a/b-cm
        """

        self._isotopes = [isotope_factory(*iso) for iso in new_isos]
        # Convert the atom fractions to an array
        self._atom_fractions = new_fracs
        self._density = new_density
        self.h5_status_change = True

    def update_from_number_densities(self, n_vector, lib, scale_constant=None):
        """Updates the material inventory based on a number density
        returned from a depletion calculation using this depletion
        library

        Parameters
        ----------
        n_vector : np.ndarray
            The number density vector itself
        lib : DepletionLibrary
            The depletion library to use
        scale_constant : None or float, optional
            If scale_constant is None, then the number density from
            n_vector will directly be applied to this material. If a
            value is provided, then the density will instead be
            scaled by this value.
        """

        new_isos, new_fracs, new_density = \
            self.calc_composition_from_number_densities(n_vector,
                lib.isotope_indices, lib.inverse_isotope_indices,
                scale_constant)

        self.apply_new_composition(new_isos, new_fracs, new_density)

    def to_hdf5(self, group):
        """Writes the material to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        # If this is a non-depleting material that has never been stored, we
        # simply save a link to its previous instance instead of re-copying the
        # data
        if self._h5_path is not None and not self.h5_status_change:
            group[self._name] = h5py.SoftLink(self._h5_path)
            return

        # Gather/re-format the data we need
        num_iso = self.num_isotopes
        if self._volume is not None:
            vol = self._volume
        else:
            vol = -1.
        if self._thermal_xs_libraries:
            xslibs = np.array([np.string_(xs)
                               for xs in self._thermal_xs_libraries])
            num_therm = len(xslibs)
        else:
            xslibs = [""]
            num_therm = 1

        # Now create data set to write. The HDF5 data structure produced by
        # this is based on minimizing the number of writes to the file which
        # helps locally (based on profile) and on networked drives.

        # To do this we have created a struct of the data in the materials obj.

        # Now the isotopic data
        iso_struct = [
            ('name', 'S9'), ('is_depleting', np.bool_), ('xs_lib', 'S6')]
        iso_dtype = np.dtype(iso_struct)
        struct = [
            ('id', np.int32),
            ('is_depleting', np.bool_),
            ('status', np.int32),
            ('num_copies', np.int32),
            ('density', np.float64),
            ('volume', np.float64),
            ('flux', np.float64, (len(self._flux),)),
            ('default_lib', 'S6'),
            ('thermal_libs', 'S20', (num_therm,)),
            ('atom_fractions', np.float64, (num_iso,)),
            ('iso_data', iso_dtype, (num_iso,))]
        # Now assign these values
        iso_vals = [None] * num_iso
        for i, iso in enumerate(self._isotopes):
            iso_vals[i] = (iso.name, iso.is_depleting, iso.xs_library)
        vals = \
            [(self._id, self._is_depleting, self._status, self.num_copies,
             self._density, vol, self._flux,
             self._default_xs_library, xslibs, self._atom_fractions,
             iso_vals)]
        data = np.array(vals, dtype=np.dtype(struct))
        dset = group.create_dataset(self._name, data=data)

        self._h5_path = dset.name
        self.h5_status_change = False

        # TODO: When a full restart is implemented, this must be modified to
        # write items such as self.isotopes_to_keep_in_model

    @classmethod
    def from_hdf5(cls, group, name):
        """Initializes a Material object from an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        this : Material
            A Material object initialized from the HDF5 file

        """

        # Lets get the dataset from the group
        dset = group[name]
        data = dset[()]

        # Now extract the data we need
        id_ = int(data['id'][0])
        is_depleting = bool(data['is_depleting'][0])
        status = int(data['status'][0])
        num_copies = int(data['num_copies'][0])
        density = float(data['density'][0])
        volume = float(data['volume'][0])
        if volume == -1:
            volume = None
        flux = data['flux'][0]
        num_groups = flux.shape[0]
        default_xs_library = data['default_lib'][0].decode()
        thermal_xs_libraries = [s.decode() for s in data['thermal_libs'][0]]
        if thermal_xs_libraries == ['']:
            thermal_xs_libraries = []
        atom_fractions = [float(f) for f in data["atom_fractions"][0]]
        iso_names = data['iso_data'][0]['name']
        iso_is_depl = data['iso_data'][0]['is_depleting']
        iso_xs_lib = data['iso_data'][0]['xs_lib']

        # Now we can create the isotopes
        iso_data = []
        for iso, xslib, depleting in zip(iso_names, iso_xs_lib, iso_is_depl):
            iso_data.append((iso.decode(), xslib.decode(), bool(depleting)))

        this = cls(name, id_, density, iso_data, atom_fractions,
                   is_depleting, default_xs_library, num_groups,
                   thermal_xs_libraries, status)

        # Set the attributes that are set after initialization
        this.flux = flux
        this.volume = volume
        # We will disreagrd isotopes_in_neutronics data since that will be
        # re-populated whenever the information is written to the neutronics
        # input
        this.isotopes_in_neutronics = True
        this.num_copies = num_copies

        # TODO: When a full restart is implemented, this must be modified to
        # read items such as self.isotopes_to_keep_in_model

        return this

    def clear_logs(self):
        self.logs = []
