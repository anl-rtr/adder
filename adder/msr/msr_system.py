import math
from collections import deque

import numpy as np

from adder.depletionlibrary import DepletionLibrary, DecayData
from adder.material import Material
import adder.data
from adder.constants import MAX_METASTABLE_STATE, AVOGADRO, BASE_LIB
from adder.type_checker import *
from .msr_component import MSRComponent, MSRTank


class MSRSystem(object):
    """The collection of components that, together, make up an MSR fluid
    system.

    Parameters
    ----------
    solve_method : {"brute", "tmatrix", "tmatrix_expm", "rxn_rate_avg"}
        The methodology to use when performing the depletion
    system_data : dict
        The dictionary of input information from the input file
    depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
    materials : List of Material
        The reactor materials

    Attributes
    ----------
    name : str
        The name of the component
    method : {"brute", "tmatrix", "tmatrix_expm", "rxn_rate_avg"}
        The methodology to use when performing the depletion
    mass_flowrate : float
        The mass flowrate through this system, in units of kg/sec
    mass_feed_rate : list of floats
        The mass flowrate of feed into the system, in unit of kg/sec,
        for each depletion step
    num_original_isotopes : int
        The number of isotopes in the upstream data library, that is,
        before chemical removal processes are incorporated in to the
        system.
    component_network : dict
        The graph of the components making up the flowpath. The key
        is the MSRComponent object, the associated value is a list of
        the connected nodes (which are also MSRComponent objects).
    component_network_weights : dict
        The edge weight of the component network. They key again is the
        MSRComponent object, the associated value is a list of the
        weights (as a float) to the downstream nodes.
    starting_component : MSRComponent
        The start of the network. This component must not have any
        parallel paths around it.
    paths : List of Lists of MSRComponents
        The flowpaths through each of the system, from the starting
        component to each of the components immediately upstream of the
        starting component.
    feed_vector : list of np.ndarrays
        The feed vector, in units of atoms/sec scaled to the mass
        feed rate, for each depletion step
    feed_v_rate: list of floats
        The volumetric feed rate, in units of cc/s, for each
        depletion step
    feed_density : list of floats
        The density of the feed, in units of g/cc, for each depletion
        step
    path_T_matrices : List of scipy.sparse.csc_matrix
        The transition matrices for each of the paths; only populated
        if method is "tmatrix" or "tmatrix_expm.
    """

    def __init__(self, solve_method, system_data, depl_libs, materials):
        check_type("system_data", system_data, dict)
        check_iterable_type("materials", materials, Material)
        self.name = system_data["name"]
        self.method = solve_method
        self.mass_flowrate = system_data["flowrate"]
        # Make a new lib for this system and register it upstream
        new_lib = depl_libs[BASE_LIB].clone(
            new_name=self.name + "_" + depl_libs[BASE_LIB].name)
        self.library = new_lib
        depl_libs[new_lib.name] = new_lib
        self.path_T_matrices = []

        # Now we have to add isotopes to the library for all the removal
        # isotopes in all the components, that way each and every
        # matrix in the loop has the same size for simplified calcs.
        # First store the number of isotopes in the original library
        # So we know where we can pull things off from
        self.num_original_isotopes = depl_libs[BASE_LIB].num_isotopes
        _add_removal_isos(self._library, system_data)
        self._library.set_atomic_mass_vector()

        # Initialize the components
        comp_by_name = {}
        comp_data_by_name = {}
        for key, c_info in system_data.items():
            if not key.startswith("component_"):
                continue

            # In these we set the component mass flowrate to the system
            # flowrate until we have the hierarchy down and then we can
            # apply the right values
            if c_info["type"].lower() == "generic":
                new_lib = self._library.clone(
                    new_name=c_info["name"] + '_' + self._library.name)
                depl_libs[new_lib.name] = new_lib
                comp = MSRComponent(c_info["name"], None, 0.,
                                    c_info["density"], c_info["volume"],
                                    c_info["removal_vector"],
                                    False, new_lib)
            elif c_info["type"].lower() == "in-core":
                # the only difference is the material name exists and
                # the density comes from our materials
                found_it = False
                for mat in materials:
                    if mat.name == c_info["mat_name"]:
                        # Get the mass density for initialization
                        mass_density = mat.mass_density
                        # Now we need to make sure this depletion lib
                        # includes our removal isotopes
                        # But we can't also change the problem-wide
                        # library, so if this is problem-wide, clone it
                        if mat.is_default_depletion_library:
                            new_lib = depl_libs[mat.depl_lib_name].clone(
                                new_name=c_info["name"] + '_' + self._library.name)
                            depl_libs[new_lib.name] = new_lib
                            mat.depl_lib_name = new_lib.name
                        # Now we can add it
                        matlib = depl_libs[mat.depl_lib_name]
                        _add_removal_isos(matlib, system_data)

                        # Ok, now we can proceed
                        found_it = True
                        if not mat.is_depleting:
                            msg = "Cannot assign non-depleting material to" + \
                                " an MSR system component!"
                            raise ValueError(msg)
                        break
                if not found_it:
                    msg = "Could not find material {} from system diagram!"
                    raise ValueError(msg.format(c_info["mat_name"]))

                comp = MSRComponent(c_info["name"], c_info["mat_name"], 0.,
                                    mass_density, c_info["volume"],
                                    c_info["removal_vector"], True, matlib)
            elif c_info["type"].lower() == "tank":
                new_lib = self._library.clone(
                    new_name=c_info["name"] + '_' + self._library.name)
                depl_libs[new_lib.name] = new_lib
                comp = MSRTank(c_info["name"], None, 0., c_info["density"],
                               c_info["volume"], c_info["removal_vector"],
                               False, new_lib)
            comp_by_name[comp.name] = comp
            comp_data_by_name[comp.name] = \
                (c_info["downstream_components"],
                 c_info["downstream_mass_fractions"])

        # Now re-arrange into a network
        self.component_network = {}
        self.component_network_weights = {}
        for comp in comp_by_name.values():
            self.component_network[comp] = \
                [comp_by_name[dc] for dc in comp_data_by_name[comp.name][0]]
            self.component_network_weights[comp_by_name[comp.name]] = \
                comp_data_by_name[comp.name][1]

        # Now we can determine the component flowrates; we need the
        # paths from start to end
        start = None
        for k in self.component_network.keys():
            if k.name == system_data["flow_start"]:
                start = k
                break
        if start is None:
            msg = "In system {}, the starting component {}, was not found"
            raise ValueError(msg.format(self.name, system_data["flow_start"]))
        self.starting_component = start
        # Get the ends
        ending_nodes = []
        for k, v in self.component_network.items():
            if self.starting_component in v:
                if len(v) > 1:
                    msg = "Starting node cannot have any other nodes at " + \
                        "the same level"
                    raise ValueError(msg)
                else:
                    ending_nodes.append(k)
        if len(ending_nodes) == 0:
            msg = "System graph must be cyclic! The starting node must be " + \
                "a child of some other node!"
            raise ValueError(msg)

        # To set the mass flowrates of all the components we will have to
        # traverse the system hierarchy in a recursive fashion and accrue the
        # total mass flow in each component as we go
        def _traverse(parent, net, net_wgts, top, m_dot):
            children = net[parent]
            for child in children:
                if child == top:
                    break
                c_idx = children.index(child)
                child.mass_flowrate += m_dot * net_wgts[parent][c_idx]
                _traverse(child, net, net_wgts, top,
                          m_dot * net_wgts[parent][c_idx])
        start.mass_flowrate = self._mass_flowrate
        _traverse(start, self._component_network,
                  self._component_network_weights, start, self._mass_flowrate)

        self.paths = []
        for end in ending_nodes:
            self.paths.extend(_find_all_paths(self._component_network, start,
                                              end))

        # Now get the product of mass flow fractions in each path so we
        # can use it as a weight for each path
        path_wgts = []
        for path in self.paths:
            parent = path[0]
            wgt = 1.
            for step in path[1:]:
                idx = self._component_network[parent].index(step)
                wgt *= self._component_network_weights[parent][idx]
                parent = step
            path_wgts.append(wgt)
        self.path_weights = np.array(path_wgts)

        self._init_feed(system_data["feed"])
        self._concentration_history = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        check_value("method", method, adder.constants.MSR_SOLN_METHODS)
        self._method = method

    @property
    def starting_material_name(self):
        return self.starting_component.mat_name

    @property
    def mass_flowrate(self):
        return self._mass_flowrate

    @mass_flowrate.setter
    def mass_flowrate(self, mass_flowrate):
        check_type("mass_flowrate", mass_flowrate, float)
        self._mass_flowrate = mass_flowrate

    @property
    def mass_feed_rate(self):
        return self._mass_feed_rate

    @mass_feed_rate.setter
    def mass_feed_rate(self, mass_feed_rate):
        check_iterable_type("mass_feed_rate", mass_feed_rate, float)
        self._mass_feed_rate = mass_feed_rate

    @property
    def feed_density(self):
        return self._feed_density

    @feed_density.setter
    def feed_density(self, feed_density):
        check_iterable_type("feed_density", feed_density, float)
        self._feed_density = feed_density

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, library):
        check_type("library", library, DepletionLibrary)
        self._library = library

    @property
    def num_original_isotopes(self):
        return self._num_original_isotopes

    @num_original_isotopes.setter
    def num_original_isotopes(self, num_original_isotopes):
        check_type("num_original_isotopes", num_original_isotopes, int)
        self._num_original_isotopes = num_original_isotopes

    @property
    def component_network(self):
        return self._component_network

    @component_network.setter
    def component_network(self, component_network):
        check_type("component_network", component_network, dict)
        self._component_network = component_network

    @property
    def component_network_weights(self):
        return self._component_network_weights

    @component_network_weights.setter
    def component_network_weights(self, component_network_weights):
        check_type("component_network_weights", component_network_weights,
                   dict)
        self._component_network_weights = component_network_weights

    @property
    def starting_component(self):
        return self._starting_component

    @starting_component.setter
    def starting_component(self, starting_component):
        check_type("starting_component", starting_component, MSRComponent)
        self._starting_component = starting_component

    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, paths):
        check_type("paths", paths, list)
        self._paths = paths

    @property
    def feed_vector(self):
        return self._feed_vector

    @feed_vector.setter
    def feed_vector(self, feed_vector):
        if feed_vector is not None:
            check_iterable_type("feed_vector", feed_vector, np.ndarray)
        self._feed_vector = feed_vector

    @property
    def feed_v_rate(self):
        return self._feed_v_rate

    @feed_v_rate.setter
    def feed_v_rate(self, feed_v_rate):
        if feed_v_rate is not None:
            check_iterable_type("feed_v_rate", feed_v_rate, float)
        self._feed_v_rate = feed_v_rate

    @property
    def path_T_matrices(self):
        return self._path_T_matrices

    @path_T_matrices.setter
    def path_T_matrices(self, path_T_matrices):
        check_type("path_T_matrices", path_T_matrices, list)
        self._path_T_matrices = path_T_matrices

    def update_time_properties(self):
        # This method sets the path_offsets, path_times, and
        # min_transport_time properties after the component volumes are
        # altered.
        self.path_times = np.array(
            [sum([c.delta_t for c in path]) for path in self.paths])

        self.min_transport_time = np.min(self.path_times)

        self.path_offsets = \
            np.subtract(np.floor_divide(self.path_times,
                                        self.min_transport_time),
                        1).astype(np.int)

    def update_path_weights(self):
        # This method updates the weighting of the paths based on mass flow
        # fractions; used by the rxn-rate-avg method only

        # First get the product of mass flow fractions in each path so we
        # can use it as a weight for each path
        # (Note, this is the same as what is in the System initialization
        # but it needs to be performed here since we cant rely on the old vals)
        # after one time step
        original_path_wgts = []
        for path in self.paths:
            parent = path[0]
            wgt = 1.
            for step in path[1:]:
                idx = self._component_network[parent].index(step)
                wgt *= self._component_network_weights[parent][idx]
                parent = step
            original_path_wgts.append(wgt)

        # Get the step-wise weights so we can use this in vol-weighting
        step_total_weight = {
            comp: 0.0 for comp in self.component_network.keys()}
        for p, path in enumerate(self.paths):
            old_wgt = original_path_wgts[p]
            for step in path:
                step_total_weight[step] += old_wgt

        total_volume = 0.0
        new_wgt = []
        for p, path in enumerate(self.paths):
            old_wgt = original_path_wgts[p]
            wgt = 0.0
            for step in path:
                wgt += old_wgt / step_total_weight[step] * step.volume
                total_volume += old_wgt / step_total_weight[step] * step.volume
            new_wgt.append(wgt)

        # Store the results
        self.path_weights = np.array(new_wgt) / total_volume

    def _init_history_storage(self, n_vector):
        offsets = self.path_offsets
        self._concentration_history = []
        for p in range(len(self.paths)):
            if offsets[p] == 0:
                # Just set it to something that is not None so that we
                # do not need to keep running this init
                self._concentration_history.append(1)
            else:
                self._concentration_history.append(
                    deque([self.path_weights[p] * n_vector.copy()]) *
                    offsets[p])

    def _get_and_set_histories(self, vectors):
        # This method modifies the nuclidic vectors in-place by
        # replacing the computed concentrations with those that make it
        # to the loop exit at the current time. Those which will arrive
        # in the future are stored in the concentration history param.
        for o, offset in enumerate(self.path_offsets):
            if offset != 0:
                while len(self._concentration_history[o]) < offset:
                    # Then the offset has changed since running
                    # _init_history_storage, as it could for a tank,
                    # and so to catch up, we need to add duplicates
                    # of the vector on the RHS to catch up
                    # This will be a source of error, though likely
                    # only minimally so
                    self._concentration_history[o].append(vectors[o, :].copy())
                # Add the vector to the history deck
                self._concentration_history[o].append(vectors[o, :].copy())
                # And replace with the correct vector
                vectors[o, :] = self._concentration_history[o].popleft()

    def _init_feed(self, feed_data):
        """Initializes the feed data including unit conversion, etc"""

        library = self._library
        library.set_isotope_indices()
        library.set_atomic_mass_vector()

        if not feed_data:
            # Then there is no vector data, and so there is no feed.
            # Lets leave a 0 feed vector and move on
            self.feed_vector = []
            self.feed_vector.append(np.zeros(library.num_isotopes))
            self.feed_density = []
            self.feed_density.append(np.zeros(1))
            self.mass_feed_rate = []
            self.mass_feed_rate.append(np.zeros(1))
            self.feed_v_rate = []
            self.feed_v_rate.append(np.zeros(1))
            return

        # Check the feed vector data
        check_iterable_type("feed_vector", feed_data["feed_vector"], dict)
        check_iterable_type("feed_rate", feed_data["feed_rate"], float)

        # create list of mass feed rate floats and feed vector dictionaries
        self.mass_feed_rate = []
        self.feed_vector = []
        self.feed_v_rate = []
        for i in range(len(feed_data["feed_vector"])):
            for k, v in feed_data["feed_vector"][i].items():
                # Check the key is a string and a valid GND name
                check_type("feed_vector key", k, str)
                if not adder.data.is_isotope(k):
                    msg = \
                        "{} is not a valid Isotope or Element Name!".format(k)
                    raise ValueError(msg)
                # Now we can check the value and ensure its a float
                # theoretically negative values are acceptable, so we wont
                # check for that
                check_type("feed_vector value", v, float)

            # Now expand the elements into their isotopes
            # Define the feed vector
            expanded_feed_vector = np.zeros(library.num_isotopes)
            # Incorporate the feed data
            for iso_name in library.isotopes.keys():
                lib_idx = library.isotope_indices[iso_name]
                if iso_name in feed_data["feed_vector"][i]:
                    # Then we simply add this bad boy in
                    expanded_feed_vector[lib_idx] += \
                        feed_data["feed_vector"][i][iso_name]
                else:
                    # We should check to see if the element is there
                    # Get the element name
                    z, a, m = adder.data.zam(iso_name)
                    elem_name = adder.data.ATOMIC_SYMBOL[z]
                    if elem_name in feed_data["feed_vector"][i]:
                        # Check to see if this is a valid element
                        # (i.e., one with abundances)
                        if elem_name not in adder.data.NATURAL_ABUNDANCE_BY_ELEM:
                            msg = "Cannot define a feed vector with a" + \
                                " non-naturally occuring element." + \
                                " This element's feed vector should" + \
                                " instead be defined isotopically."
                            raise ValueError(msg)

                        # if so, add in the feed, weighted by abundance
                        if iso_name in adder.data.NATURAL_ABUNDANCE:
                            expanded_feed_vector[lib_idx] += \
                                feed_data["feed_vector"][i][elem_name] * \
                                adder.data.NATURAL_ABUNDANCE[iso_name]

            # Now lets normalize the feed vector
            expanded_feed_vector /= np.sum(expanded_feed_vector)

            # If our units are weight percent, convert to atom-percent
            if feed_data["vector_units"] == "wo":
                # Then we need to convert from wo to ao
                # Convert expanded_feed_vector to moles
                expanded_feed_vector /= library.atomic_mass_vector
                # Now renormalize to come up with mole fractions
                # (which are the same thing as atom fractions)
                expanded_feed_vector /= np.sum(expanded_feed_vector)

            # If our feed rate units are a mass flowrate, we need to convert
            # to an atom rate
            if feed_data["feed_rate_units"] == "kg/sec":
                # Ugh, best go back to w/o for this
                wo_vector = expanded_feed_vector * library.atomic_mass_vector
                wo_vector /= np.sum(wo_vector)
                # Divvy up the masses amongst the isotopes
                mass_vector = feed_data["feed_rate"][i] * 1000. * wo_vector
                # Now convert from masses to atoms
                expanded_feed_vector = \
                    mass_vector * AVOGADRO / library.atomic_mass_vector
                # The mass feed rate is already in kg/s, so no need to do
                # anything
                mass_feed_rate = feed_data["feed_rate"][i]
            else:
                expanded_feed_vector *= feed_data["feed_rate"][i]
                # Need to get a mass feed rate in units of kg/s. Now it is
                # in atoms/sec, so multipy each atom type by its mass and
                # convert to kg/s
                mass_feed_rate = \
                    np.dot(expanded_feed_vector,
                           library.atomic_mass_vector) / \
                    AVOGADRO / 1000.

            self.mass_feed_rate.append(mass_feed_rate)
            # Finally, populate self.feed_vector, in units of atoms/sec
            self.feed_vector.append(expanded_feed_vector)
            # volumetric feed rate
            #       [cc / s]  =       [kg / s]   * [g / kg] /     [g / cc]
            self.feed_v_rate.append(mass_feed_rate * 1e3 / feed_data["density"][i])
        self.feed_density = feed_data["density"]

    def add_feed(self, n_vector, feed_dt, in_depletion_step):
        # Returns the feed vector normalized to the correct adjustment
        # to a number density vector

        # If there is no feed, do nothing
        if self._mass_feed_rate[in_depletion_step] == 0. or \
            self._feed_density[in_depletion_step] == 0.:
            return n_vector

        # To find the new concentration after feed, we do a volumetric
        # mixing calculation. To do that we need the number densities
        # of the system and feed salts, and the volumes of each.
        # From there it is simply
        # N_new = N_sys * V_sys + N_feed * V_feed / V_new
        # where sys refers to the component.

        # To get V_feed, we multiply the volumetric feed rate by
        # the duration that feed was added to this component (feed_dt)
        # duration in units of [s], V_rate in units of [cc/s]
        # [cc] =      [s]      *    [cc / s]
        V_feed = feed_dt * self._feed_v_rate[in_depletion_step]

        # Now we just need the component and new volume (in cc)
        # [cc] =            [m^3]              * [cc / m^3]
        V_sys = self.starting_component.volume * 1.e6
        V_new = V_feed + V_sys

        # Now compute volume fractions by normalizing by the
        # new volume (V_new above)
        V_sys_over_V_new = V_sys / V_new
        # Since feed_vector is in terms of atoms/sec, we just need to
        # multiply by the duration of feed. feed_dt_over_V_new
        # is that duration divided by V_new to save flops
        feed_dt_over_V_new = feed_dt / V_new

        # Now incorporate the feed vector by volume-weighting
        n_with_feed = V_sys_over_V_new * n_vector + feed_dt_over_V_new * \
            self._feed_vector[in_depletion_step]

        return n_with_feed

    def solve_rxn_rate_avg_method(self, materials_by_name, duration,
                                  depletion_step, solver_func, n_i, amu_vec):
        """This method performs the flux-averaging methodology.
        """

        # In this method, the xs * flux and decay constants are replaced with
        # time-averaged values when creating the depletion matrix.
        # This time-averaging will be done for each of the linearized paths
        # in the system.
        # To estimate the transmutation over the requested duration, the
        # depletion will be performed over each of these linearized paths
        # and combined with the resultant weights.

        # First create the libraries for each path
        num_groups = self._library.num_neutron_groups
        libs = [self._library.clone(new_name="Path {}".format(p + 1))
                for p, path in enumerate(self.paths)]

        # Update path weights so its available for the first iteration.
        self.update_path_weights()

        # First get the path-wise total flux
        pathflux = [np.zeros(num_groups) for p in range(len(self.paths))]
        for p, path in enumerate(self.paths):
            for step in path:
                if step.mat_name is not None:
                    mat = materials_by_name[step.mat_name]
                    pathflux[p] += mat.flux * step.delta_t
            pathflux[p] /= self.path_times[p]

        for p, path in enumerate(self.paths):
            for iso_name, iso in libs[p].isotopes.items():
                # Compute the time-averaged rxn rate for this isotope across
                # all libs
                iso_xs = iso.neutron_xs
                if iso_xs is not None:
                    # Iterate over every rxn channel
                    for type_ in iso_xs.keys():
                        rxn = np.zeros(num_groups)
                        # Now get the data for each component
                        for step in path:
                            if step.mat_name is not None:
                                mat = materials_by_name[step.mat_name]
                                mat_n_xs = \
                                    step.library.isotopes[iso_name].neutron_xs
                                ref_xs = mat_n_xs._products[type_][0]
                                # All xs will be in consistent units as the
                                # libraries will be sourced from the same lib,
                                # so unit conversions/checks are absent here.
                                rxn += ref_xs * mat.flux * step.delta_t

                                # TODO: when ADDER supports energy-dependent
                                # fission yields, then this block of code
                                # should also combine fission yields
                        # Now normalize rxn by the flux and total time
                        rxn /= (pathflux[p] * self.path_times[p])
                        # Store the xs in our path's library
                        _, t, y, q = iso_xs[type_]
                        iso_xs._products[type_] = (rxn, t, y, q)

                # Repeat for the decay constants. In this code we are making
                # the pretty bulletproof assumption that the decay constants
                # are the same for each library, the only differences will be
                # in the removal rates of MSR purification systems/processes.
                decay_const = 0.
                for step in path:
                    step_removal = step.library.isotopes[iso_name].removal
                    if step_removal is not None:
                        decay_const += step_removal.decay_constant * \
                            step.delta_t

                # Now normalize the removal decay constant
                decay_const /= self.path_times[p]
                if decay_const != 0.:
                    iso.removal = DecayData(math.log(2) / decay_const, "s", 0.)
                    z, a, m = adder.data.zam(iso_name)
                    child = adder.data.gnd_name(z, a, m + MAX_METASTABLE_STATE)
                    iso.removal.add_type("removal", 1., child)

        # Now create the depletion matrices
        matrices = [lib.build_depletion_matrix(pathflux[p], matrix_format="csr")
                    for p, lib in enumerate(libs)]

        # Create some initial information we need for all depletions
        dt = duration * 86400.
        n_in = np.zeros((len(libs), self._library.num_isotopes))
        n_in[:, :self.num_original_isotopes] = n_i[:]
        n_path_out = np.zeros((len(libs), self._library.num_isotopes))

        # Deplete each path and scale by the weight of this path
        for p, matrix in enumerate(matrices):
            n_path_out[p, :] = self.path_weights[p] * \
                solver_func(matrix, n_in[p, :], dt, "s")

        # Now combine all the paths into one
        n_end = np.sum(n_path_out, axis=0)

        # And incorporate feed
        total_volume = sum(comp.volume
            for comp in self.component_network.keys())
        feed_dt = dt * self.starting_component.volume / total_volume
        # if there is only 1 feed rate value, assume constant feed rate
        if len(self._mass_feed_rate) == 1:
            in_depletion_step = 0
        else:
            in_depletion_step = depletion_step
        n_end = self.add_feed(n_end, feed_dt, in_depletion_step)

        # Now since feed_vector can be negative, make sure the data stays >= 0
        n_end[n_end < 0.] = 0.

        # Strip off the removed isotopes from our return vector
        n_o = n_end[:self.num_original_isotopes]

        # Now we have to update the densities and the volumes
        # To do that we need to figure out the density ratio from
        # before and after
        density_ratio = np.dot(n_o, amu_vec) / np.dot(n_i, amu_vec)
        for comp in self.component_network.keys():
            comp.update_density(density_ratio)
            comp.update_volume(dt, self._feed_v_rate[in_depletion_step])

        # after updating volumes, update path weights
        self.update_path_weights()

        return n_o

    def compute_matrices(self, materials_by_name, duration, solver_func):
        """Initializes the matrices we need to deplete this step"""

        # Convert the step duration to a time in seconds
        dt = duration * 86400.

        # Go through each component, find any relevant flux, and pass
        # it to MSRComponent.init_A_matrix
        for component in self.component_network.keys():
            if component.in_core and component.mat_name is not None:
                flux = materials_by_name[component.mat_name].flux
            else:
                flux = None
            component.init_A_matrix(self.method, flux)

            # Now if we are using the t-matrix approach, we should also
            # compute the t-matrix
            if self.method == "tmatrix":
                component.init_T_matrix(dt, solver_func, True)
            elif self.method == "tmatrix_expm":
                component.init_T_matrix(dt, solver_func, False)

        # Now we can go through each path to create an aggregate
        # T-matrix for each path
        if self.method.startswith("tmatrix"):
            # re-init the path matrices
            self.path_T_matrices = []
            for p, path in enumerate(self.paths):
                T = None
                for step in path:
                    if T is None:
                        # Then this is our first time through so dont
                        # multiply, just assign
                        T = step._T_matrix
                    else:
                        # "Accrue" operations via left-mult of next T
                        T = step._T_matrix @ T

                self.path_T_matrices.append(self.path_weights[p] * T)

    def solve(self, duration, depletion_step, n_i, solver_func, amu_vec):
        """Solves the depletion system for the given duration"""

        # Convert the duration to a time in seconds
        dt = duration * 86400.

        # Compute the number of loops
        num_loops = int(np.around(dt / self.min_transport_time))

        # Begin processing
        path_vectors = np.zeros((len(self._paths), self._library.num_isotopes))
        n_end = np.zeros(self._library.num_isotopes)
        n_end[:self.num_original_isotopes] = n_i[:]

        # if there is only 1 feed rate value, assume constant feed rate
        if len(self._mass_feed_rate) == 1:
            in_depletion_step = 0
        else:
            in_depletion_step = depletion_step

        for step_index in range(num_loops):
            n_start = n_end
            if self._method == "brute":
                for p, path in enumerate(self._paths):
                    n_enter = n_start.copy()
                    for step in path:
                        n_exit = step.transmute(dt, solver_func, n_enter)
                        n_enter = n_exit
                    path_vectors[p, :] = self.path_weights[p] * n_exit
            elif self._method.startswith("tmatrix"):
                for p in range(len(self._paths)):
                    path_vectors[p, :] = self._path_T_matrices[p] @ n_start

            # Now put the path_vectors at the right place in history
            if self._concentration_history is None:
                # Initialize it first by placing n_start at all
                # the previous timesteps (indicating the fluid is
                # homogenously mixed throughout)
                self._init_history_storage(n_start)

            # Now we can set the data that left each path in the queue
            # and replace it with the data from the queue.
            self._get_and_set_histories(path_vectors)

            # At this point, path_vectors contains the data we need
            # and we just need to merge them
            n_end = np.sum(path_vectors, axis=0)

            # And finally, incorporate the feed
            n_end = self.add_feed(n_end, self.starting_component.delta_t,
                                  in_depletion_step)

            # Now since feed_vector can be negative, lets make sure the
            # data stays >= 0
            n_end[n_end < 0.] = 0.

        # Strip off the removed isotopes from our return vector
        n_o = n_end[:self.num_original_isotopes]

        # Now we have to update the densities and the volumes
        # To do that we need to figure out the density ratio from
        # before and after
        density_ratio = np.dot(n_o, amu_vec) / np.dot(n_i, amu_vec)

        for comp in self.component_network.keys():
            comp.update_density(density_ratio)
            comp.update_volume(dt, self._feed_v_rate[in_depletion_step])

        return n_o


def _find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = _find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def _add_removal_isos(lib, system_data):
    # Add all of the isotopes we wish to have removal processes for into the
    # library. Since the names can be elements or isotopes, we will iterate
    # through the library's isotopes to see if the data is in the
    # removal vector
    # Before we do that, get the names of the library isos
    orig_iso_names = list(lib.isotopes.keys())

    # Now, to speed up iteration, lets get all the component keys
    c_keys = [k for k in system_data.keys() if k.startswith("component_")]
    for name in orig_iso_names:
        for key in c_keys:
            c_info = system_data[key]

            # Now we have the component, analyze it's removal vector
            if name in c_info["removal_vector"]:
                # Create the child isotope
                z, a, m = adder.data.zam(name)
                child = adder.data.gnd_name(z, a, m + MAX_METASTABLE_STATE)
                if child not in lib.isotopes:
                    lib.add_isotope(child,
                                    decay=DecayData(None, "s", 0.))
                break
            else:
                # It still could be an element
                z, a, m = adder.data.zam(name)
                elem_name = adder.data.ATOMIC_SYMBOL[z]
                if elem_name in c_info["removal_vector"]:
                    # It is, so figure out the child name
                    child = \
                        adder.data.gnd_name(z, a, m + MAX_METASTABLE_STATE)
                    if child not in lib.isotopes:
                        lib.add_isotope(child,
                                        decay=DecayData(None, "s", 0.))
                    break
