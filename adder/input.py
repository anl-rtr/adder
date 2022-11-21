import os.path
from collections import OrderedDict
import re

import numpy as np
from configobj import ConfigObj

import adder.input_validation
from adder.reactor import Reactor
from adder.control_group import ControlGroup
from adder.loggedclass import LoggedClass
from adder.constants import *
from adder.msr_reactor import MSRReactor


logger = LoggedClass(0, __name__)
_INDENT = 2


def get_input(input_file):
    """Reads the input file located at the input_file path and returns
    an initialized adder.Reactor object.

    Parameters
    ----------
    input_file : str
        Path to the input file

    Returns
    -------
    rx : adder.Reactor
        Initialized Reactor object
    ops : List of 3-Tuple
        The state name, method name to call, and the arguments for that
        method based on the operations information within the input file
    """

    # Check the file is a valid string and a valid file
    if not isinstance(input_file, str):
        logger.log("error", "input_file must be a string")
    if not os.path.isfile(input_file):
        logger.log("error", "{} does not exist".format(input_file))

    # Use the configobj package to read the input and put it in a dict
    # named config
    logger.log("info", "Processing ADDER Input")
    config = ConfigObj(input_file, raise_errors=True,
                       list_values=True, stringify=True)
    adder.input_validation.validate(config)
    input_echo(config)

    rx = setup_reactor(config)

    mat_aliases = OrderedDict()
    uni_aliases = OrderedDict()
    if "materials" in config:
        mat_aliases = setup_material_aliases(config["materials"], rx)
    if "universes" in config:
        uni_aliases = setup_universe_aliases(config["universes"], rx)

    ops = setup_operations(config, mat_aliases, uni_aliases, rx)

    logger.log("info", "Completed ADDER Input Processing")

    return rx, ops


def setup_reactor(config):
    """Given the config dictionary, this method initializes the
    Adder.Reactor object.

    Parameters
    ----------
    config : dict
        ConfigObj-processed and validated data

    Returns
    -------
    rx : adder.Reactor
        Initialized Reactor object
    """

    # Get all our parameters of interest
    name = config["case_name"]
    neutronics_solver = config["neutronics_solver"]
    depletion_solver = config["depletion_solver"]
    neutronics_exec = config["neutronics_exec"]
    mpi_command = config["mpi_command"]
    depletion_exec = config["depletion_exec"]
    base_neutronics_input_filename = config["neutronics_input_file"]
    h5_filename = config["output_hdf5"]
    num_neut_threads = config["num_neutronics_threads"]
    num_depl_threads = config["num_depletion_threads"]
    num_mpi_procs = config["num_mpi_processes"]
    depletion_chunksize = config["depletion_chunksize"]
    use_depletion_library_xs = config["use_depletion_library_xs"]
    neutronics_reactivity_threshold = config["neutronics_reactivity_threshold"]
    neutronics_reactivity_threshold_initial = \
        config["apply_reactivity_threshold_to_initial_inventory"]

    if mpi_command is None:
        # Then the user didnt provide an MPI command, so do not use it
        mpi = ""
    else:
        mpi = mpi_command

    # Get either a base Reactor class or a MSRRector class depending on
    # if the msr section is in the configuration
    if "msr" in config:
        rx = MSRReactor(name, neutronics_solver, depletion_solver, mpi,
                        neutronics_exec, depletion_exec,
                        base_neutronics_input_filename, h5_filename,
                        num_neut_threads, num_depl_threads, num_mpi_procs,
                        depletion_chunksize, use_depletion_library_xs,
                        neutronics_reactivity_threshold,
                        neutronics_reactivity_threshold_initial)
    else:
        rx = Reactor(name, neutronics_solver, depletion_solver, mpi,
                     neutronics_exec, depletion_exec,
                     base_neutronics_input_filename, h5_filename,
                     num_neut_threads, num_depl_threads, num_mpi_procs,
                     depletion_chunksize, use_depletion_library_xs,
                     neutronics_reactivity_threshold,
                     neutronics_reactivity_threshold_initial)

    # Continue with the additional reactor initialization (same for both
    # types of reactors)
    # First get the additional data we will need from the user input
    neutronics_library_file = config["neutronics_library_file"]
    depletion_library_file = config["depletion_library_file"]
    depletion_library_name = config["depletion_library_name"]

    # Get the materials info and create the info needed to be
    # passed to Reactor.init_materials_and_input
    materials_data = OrderedDict()
    universe_data = OrderedDict()
    if "materials" in config:
        if "metadata" in config["materials"]:
            materials_data = \
                get_matuni_metadata(config["materials"]["metadata"], True)
    if "universes" in config:
        if "metadata" in config["universes"]:
            universe_data = \
                get_matuni_metadata(config["universes"]["metadata"], False)
    shuffled_mats, shuffled_univs = get_mat_univ_shuffles(config)

    init_args = (neutronics_library_file, depletion_library_file,
                 depletion_library_name, materials_data, universe_data,
                 shuffled_mats, shuffled_univs)

    # Get and apply the additional MSR info if available
    if "msr" in config:
        logger.log("warning", "Since an [msr] block is present, "
                   "MSR Functionality has been enabled; "
                   "This functionality has not yet been reviewed for QA "
                   "Compliance!")
        msr_data = config["msr"]

        # Now format the information in msr_data into the parameters
        # needed for MSRReactor's init_materials_and_input
        smoothing = msr_data["flux_smoothing_method"]
        method = msr_data["solve_method"]

        sys_data = []
        for key in msr_data:
            if key.lower().startswith("system") or key.lower() == "feed":
                sys_data.append(msr_data[key])

        # Now update init_args
        init_args = init_args + (smoothing, method, sys_data)

    # Now we can run init_materials_and_input
    logger.log("info", "Identifying Materials from Neutronics Input File",
               indent=_INDENT)
    rx.init_materials_and_input(*init_args)

    # After we have this info then let's do what we need to for the
    # definition of supply materials and universes
    if "materials" in config:
        if "supply" in config["materials"]:
            create_mat_supply_storage(config["materials"]["supply"],
                                      rx.materials, rx.depletion_libs,
                                      "[materials][[supply]]")
        if "storage" in config["materials"]:
            create_mat_supply_storage(config["materials"]["storage"],
                                      rx.materials, rx.depletion_libs,
                                      "[materials][[storage]]")
    # And now copy the universes from supply
    if "universes" in config:
        if "supply" in config["universes"]:
            create_univ_supply_storage(config["universes"]["supply"],
                                       rx, "[universes][[supply]]")
        if "storage" in config["universes"]:
            create_univ_supply_storage(config["universes"]["storage"],
                                       rx, "[universes][[storage]]")

    # Setup the control group information
    if "control_groups" in config:
        for group_name, data in config["control_groups"].items():
            rx.control_groups[group_name] = ControlGroup(group_name, data)

    return rx


def get_mat_univ_shuffles(config):
    # This parses the config block to figure out which materials are shuffled
    # so that we can determine which materials to duplicate
    # First get the material aliases (with no checking, that's done later)
    aliases = OrderedDict()
    if "aliases" in config:
        aliases_info = config["aliases"]
        for alias_data in aliases_info.values():
            aliases[alias_data["name"]] = alias_data["set"][:]

    # Now get all the material shuffles
    shuffled_mats = set()
    shuffled_univs = set()
    if "operations" in config:
        cfg_ops = config["operations"]
        for case_data in cfg_ops.values():
            for subsection, ssec_data in case_data.items():
                if subsection.startswith("shuffle") or subsection.startswith("revolve"):
                    obj_type = ssec_data["type"]
                    if obj_type == "material":
                        worker = shuffled_mats
                    elif obj_type == "universe":
                        worker = shuffled_univs

                    if subsection.startswith("shuffle"):
                        moves = ssec_data["moves"]
                    else:
                        moves = ssec_data["set"]

                    # Now add all the moves to our set, replacing with
                    # aliases as needed
                    for piece in moves:
                        if piece in aliases:
                            worker.update(aliases[piece])
                        else:
                            worker.add(piece)
    return shuffled_mats, shuffled_univs


def get_matuni_metadata(user_info, is_mat):
    # We have three material inputs: ranges, lists, individuals
    # First we set from the ranges, then the lists, and then individuals
    # From these inputs, we want to get the specific ids, names,
    # values for if it is depleting, the ex_core_status, and the
    # non_depleting_isotopes
    data = OrderedDict()

    # This will go through all ranges then all lists in order of
    # appearance
    for ssec, subcfg in user_info.items():

        # Handle the names, which are the same for range and list
        if ssec.startswith("range") or ssec.startswith("list"):
            cfg_names = subcfg["names"]

        # Handle just range
        if ssec.startswith("range"):
            neutronics_id_start = subcfg["neutronics_id_start"]
            neutronics_id_end = subcfg["neutronics_id_end"]
            exclude = subcfg["exclude_neutronics_ids"]
            density = subcfg["density"]
            if exclude is None:
                exclude = []

            # Convert to a list of neutronics_ids and names
            neutronics_ids = []
            names = []
            densities = []
            i = 0
            for id_ in range(neutronics_id_start, neutronics_id_end + 1):
                if id_ not in exclude:
                    neutronics_ids.append(id_)

                    # Set the name
                    if cfg_names is None:
                        names.append(str(id_))
                    elif len(cfg_names) == 1:
                        names.append(cfg_names[0] + "_{}".format(id_))
                    else:
                        names.append(cfg_names[i])
                    i += 1

                    densities.append(density)

        elif ssec.startswith("list"):
            # Handle the list
            neutronics_ids = subcfg["neutronics_ids"]
            densities = subcfg["densities"]

            # Convert to a list of names
            names = []
            for i, id_ in enumerate(neutronics_ids):
                # Set the name
                if cfg_names is None:
                    names.append(str(id_))
                elif len(cfg_names) == 1:
                    names.append(cfg_names[0] + "_{}".format(id_))
                else:
                    names.append(cfg_names[i])

        elif ssec.startswith("item"):
            # End with the individuals
            neutronics_ids = [subcfg["neutronics_id"]]
            name = subcfg["name"]
            if name is None:
                names = [str(neutronics_ids[0])]
            else:
                names = [name]

            densities = [subcfg["density"]]

        # Get the remaining data that doesnt depend on type
        if is_mat:
            volume = subcfg["volume"]
            depleting = subcfg["depleting"]
            depl_lib_flag = subcfg["use_default_depletion_library"]
            non_depleting_isotopes = subcfg["non_depleting_isotopes"]
            if non_depleting_isotopes is None:
                non_depleting_isotopes = []
            apply_rho_thresh_to_initial = \
                subcfg["apply_reactivity_threshold_to_initial_inventory"]

            # And add these neutronics_ids to the dictionary
            for id_, name, density in zip(neutronics_ids, names, densities):
                data[id_] = \
                    OrderedDict(
                        name=name, depleting=depleting, density=density,
                        non_depleting_isotopes=non_depleting_isotopes,
                        use_default_depletion_library=depl_lib_flag,
                        apply_reactivity_threshold_to_initial_inventory= \
                            apply_rho_thresh_to_initial)
                if volume is not None:
                    # Now we dont set this if its the default value
                    data[id_]["volume"] = volume
        else:
            # Universes dont have depleting statuses, just add to dict
            for id_, name in zip(neutronics_ids, names):
                data[id_] = OrderedDict(name=name)

    return data


def create_mat_supply_storage(cfg, materials, depl_libs, level):
    # This modifies materials in place

    # We have three types of supply to deal with:
    # "redefine", "copy", and "new".
    # New has yet to be implemented
    # "redefine" and "copy" each accept ranges, lists, and items
    # just like in [materials]
    # We want to get all the data from each and process as needed

    # To do this we will store the ids and names of each
    # we will use name[i] = None to denote that this is a reassignment
    # not a copy
    mat_ids = []
    mat_names = []
    if level == "[materials][[supply]]":
        status = SUPPLY
    elif level == "[materials][[storage]]":
        status = STORAGE

    # We have either redefine, copy, or new subsections
    for my_ssec, subcfg in cfg.items():
        if my_ssec == "new":
            raise NotImplementedError("new not yet supported")

        sublevel = level + "[[{}]]".format(my_ssec)
        for my_sssec, subsubcfg in subcfg.items():
            subsublevel = sublevel + "[[[{}]]]".format(my_sssec)

            # Handle just range
            if my_sssec.startswith("range"):
                n_id_start = subsubcfg["neutronics_id_start"]
                n_id_end = subsubcfg["neutronics_id_end"]
                exclude = subsubcfg["exclude_neutronics_ids"]
                if exclude is None:
                    exclude = []

                neut_ids = []
                for id_ in range(n_id_start, n_id_end + 1):
                    if id_ not in exclude:
                        neut_ids.append(id_)

            elif my_sssec.startswith("list"):
                # Handle just list
                neut_ids = subsubcfg["neutronics_ids"]

            elif my_sssec.startswith("item"):
                # Handle just individual items
                neut_ids = [subsubcfg["neutronics_id"]]

            # Figure out what to do with names
            if my_ssec == "copy":
                # Then we need to know the new names
                if my_sssec.startswith("range") or my_sssec.startswith("list"):
                    these_names = subsubcfg["names"]
                    if len(these_names) == 1:
                        these_names = [these_names[0]
                                       for i in range(len(neut_ids))]
                    elif len(these_names) != len(neut_ids):
                        msg = "In {},\n\tthe number of".format(subsublevel) + \
                            " neutronics ids does not match the number of " + \
                            "\n\tnames or just one global value"
                        logger.log("error", msg)
                else:
                    these_names = [subsubcfg["name"]
                                   for i in range(len(neut_ids))]
            else:
                these_names = [None for i in range(len(neut_ids))]

            # Now add our data to the big set
            mat_ids.extend(neut_ids)
            mat_names.extend(these_names)

    # Now we have the list of materials to operate on;
    # we need to go through each and make the changes to put them
    # in supply
    # First, lets set up a dictionary of material ids vs index in array
    mat_id_dict = OrderedDict()
    for i, material in enumerate(materials):
        mat_id_dict[material.id] = i

    # Step through each material provided and either copy or reassign
    for i, mat_id in enumerate(mat_ids):
        if mat_id not in mat_id_dict:
            msg = "In {}, neutronics_id {} ".format(level, mat_id) + \
                "is not in the model and so was not modified or copied."
            logger.log("warning", msg)
        orig_mat = materials[mat_id_dict[mat_id]]
        if mat_names[i] is None:
            # Then this is just a reassignment
            mod_mat = orig_mat
            if mat_id in mat_id_dict:
                if mod_mat.status == IN_CORE:
                    msg = "Cannot move material {}".format(mat_id) + \
                        " from the core to supply;\n\t" + \
                        "Use {}[[copy]] to make a copy".format(level)
                    logger.log("error", msg)

                msg = "Material {} Status Changed from {} to {}"
                logger.log("info", msg.format(mod_mat.name,
                                              STATUS_STRING[mod_mat.status],
                                              STATUS_STRING[status]),
                           indent=_INDENT)
                mod_mat.status = status

            else:
                msg = "In {}, neutronics_id {} ".format(level, mat_id) + \
                    "is not in the model;\n" + \
                    "The material was not copied."
                logger.log("warning", msg)
        else:
            # Then this is a copy
            mod_mat = \
                orig_mat.clone(depl_libs,
                               new_name=mat_names[i].format(orig_mat.name))
            mod_mat.status = status
            msg = "Material {} cloned as material {}".format(orig_mat.name,
                                                             mod_mat.name)
            logger.log("info_file", msg)

            # Now we have our copied material, update materials and
            # our mat_id_dict
            materials.append(mod_mat)
            mat_id_dict[mod_mat.id] = len(materials) - 1


def create_univ_supply_storage(cfg, rx, level):
    # This modifies the neutronics universe info in place

    # We have three types of supply to deal with:
    # "redefine", "copy", and "new".
    # New has yet to be implemented
    # "redefine" and "copy" each accept ranges, lists, and items
    # just like in [universes][[metadata]]
    # We want to get all the data from each and process as needed

    if level == "[universes][[supply]]":
        status = SUPPLY
    elif level == "[universes][[storage]]":
        status = STORAGE

    # We have either redefine, copy, or new subsections, step through em
    u_ids_to_change = []
    univs_to_clone = []
    for my_ssec, subcfg in cfg.items():
        if my_ssec == "new":
            raise NotImplementedError("new not yet supported")

        sublevel = level + "[[{}]]".format(my_ssec)
        for my_sssec, subsubcfg in subcfg.items():
            subsublevel = sublevel + "[[[{}]]]".format(my_sssec)

            # Now gather all the universe ids we need to worry about
            # Handle just range
            if my_sssec.startswith("range"):
                n_id_start = subsubcfg["neutronics_id_start"]
                n_id_end = subsubcfg["neutronics_id_end"]
                exclude = subsubcfg["exclude_neutronics_ids"]
                if exclude is None:
                    exclude = []

                neut_ids = []
                for id_ in range(n_id_start, n_id_end + 1):
                    if id_ not in exclude:
                        neut_ids.append(id_)

            elif my_sssec.startswith("list"):
                # Handle just list
                neut_ids = subsubcfg["neutronics_ids"]

            elif my_sssec.startswith("item"):
                # Handle just individual items
                neut_ids = [subsubcfg["neutronics_id"]]

            # Figure out what to do with names
            if my_ssec == "copy":
                # Then we need to know the new names
                if my_sssec.startswith("range") or my_sssec.startswith("list"):
                    names = subsubcfg["names"]
                    if len(names) != len(neut_ids):
                        msg = "In {},\n\tthe number of".format(subsublevel) + \
                            " neutronics ids does not match the number of " + \
                            "\n\tnames"
                        logger.log("error", msg)
                else:
                    names = [subsubcfg["name"] for i in range(len(neut_ids))]
                to_clone = [(id_, name) for id_, name in zip(neut_ids, names)]
                univs_to_clone.extend(to_clone)
            else:
                u_ids_to_change.extend(neut_ids)

    # Now we have the list of universes to operate on;
    # we need to go through each and make the changes to put them
    # in supply or storage if possible

    # For convenience and speed, create easy lookups into the
    # materials by id; we have to do this with every move as the
    # move can create new materials (with clones of storage) so we
    # dont pre-determine this
    mat_by_id = {}
    for _, mat in enumerate(rx.materials):
        mat_by_id[mat.id] = mat

    # Now go through the universes to change first
    for u_id in u_ids_to_change:
        univ = rx.neutronics.universes[u_id]
        if univ.status == IN_CORE:
            msg = "Universe {} is in-core;".format(univ.name) + \
                "Cannot change an in-core universe to ex-core; use" + \
                "the [[[copy]]] section!"
            logger.log("error", msg)
        univ.status = status

    # Now go ahead and clone universes
    # Only Neutronics subclasses with universes will actually have this
    # clone_universe method (and universes dict). However, upstream we
    # have already ensured that this will be the case.
    for u_id, new_name in univs_to_clone:

        # Check for complement operators during cloning.
        rx.neutronics.universes[u_id].update_nested_children()
        univ_cells = rx.neutronics.universes[u_id].nested_cells.keys()

        warning_cell_list = []
        for cell in rx.neutronics.complement_operator_cells:
            if cell in univ_cells:
                warning_cell_list.append(cell)

        if warning_cell_list:
            msg = "ADDER does not modify complement operators on cells during" \
                  " cloning. Cells {} are members of a universe (universe {})" \
                  " being cloned and are attached to a complement operator in" \
                  " the definition of another cell. The user must confirm " \
                  "that cloning of this universe still results in a correct " \
                  "region definition with the original complement." \
                  "".format(warning_cell_list, u_id)
            rx.neutronics.log("warning", msg)

        clone = rx.neutronics.clone_universe(u_id, rx.depletion_libs,
            rx.materials, new_name=new_name, new_status=status)


def setup_material_aliases(config, rx):
    """Given the config dictionary, this method creates the
    list of aliases to be used with fuel management operations.

    Parameters
    ----------
    config : dict
        ConfigObj-processed and validated data
    rx : adder.Reactor
        Initialized Reactor object

    Returns
    -------
    aliases : OrderedDict of List of str
        A dictionary containing material alias names and the
        list of material names contained within that alias set.
    """

    # Get the alias information
    aliases = OrderedDict()
    if "aliases" not in config:
        return aliases

    aliases_info = config["aliases"]
    for alias_data in aliases_info.values():
        name = alias_data["name"]

        # First we make sure this alias name does not exist in the model
        # as a material
        for model_material in rx.materials:
            if name == model_material.name:
                msg = "Alias {} cannot have the same ".format(name) + \
                    "name as a material"
                logger.log("error", msg)

        # The validator made sure there was a list of length 1
        # for the set attribute, all that is left is to make sure it is
        # a valid material name and that all materials in the set are
        # the same type (supply, storage, in_core)
        alias_set = alias_data["set"]
        statuses = []
        for alias_material in alias_set:
            match = False
            for model_material in rx.materials:
                if model_material.name == alias_material:
                    statuses.append(model_material.status)
                    match = True
                    break
            if not match:
                msg = "{} from alias {}".format(alias_material, name) + \
                    " is an invalid material!"
                logger.log("error", msg)
            else:
                aliases[name] = alias_set[:]

        # Make sure the statuses are all the same
        if not _all_equal(statuses):
            msg = "All materials referred to by alias {} ".format(name) + \
                "need to have the same status (in-core, storage, supply)!"
            logger.log("error", msg)

    return aliases


def setup_universe_aliases(config, rx):
    """Given the config dictionary, this method creates the
    list of aliases to be used with fuel management operations.

    Parameters
    ----------
    config : dict
        ConfigObj-processed and validated data
    rx : adder.Reactor
        Initialized Reactor object

    Returns
    -------
    aliases : OrderedDict of List of str
        A dictionary containing universe alias names and the
        list of universe names contained within that alias set.
    """

    # Get the alias information
    aliases = OrderedDict()
    if "aliases" not in config:
        return aliases

    aliases_info = config["aliases"]
    for alias_data in aliases_info.values():
        name = alias_data["name"]

        # First we make sure this alias name does not exist in the model
        # as a material
        for model_material in rx.materials:
            if name == model_material.name:
                msg = "Alias {} cannot have the same ".format(name) + \
                    "name as a universe"
                logger.log("error", msg)

        # The validator made sure there was a list of length 1
        # for the set attribute, all that is left is to make sure it is
        # a valid material name and that all materials in the set are
        # the same type (supply, storage, in_core)
        alias_set = alias_data["set"]
        statuses = []
        for alias_material in alias_set:
            match = False
            for model_material in rx.materials:
                if model_material.name == alias_material:
                    statuses.append(model_material.status)
                    match = True
                    break
            if not match:
                msg = "{} from alias {}".format(alias_material, name) + \
                    " is an invalid universe!"
                logger.log("error", msg)
            else:
                aliases[name] = alias_set[:]

        # Make sure the statuses are all the same
        if not _all_equal(statuses):
            msg = "All universes referred to by alias {} ".format(name) + \
                "need to have the same status (in-core, storage, supply)!"
            logger.log("error", msg)

    return aliases


def setup_operations(config, mat_aliases, uni_aliases, rx):
    """Given the config dictionary, this method creates the
    list of operations to be performed on the reactor.

    Parameters
    ----------
    config : dict
        ConfigObj-processed and validated data
    mat_aliases : OrderedDict of List of str
        A dictionary containing material alias names and the
        list of material names contained within that alias set.
    uni_aliases : OrderedDict of List of str
        A dictionary containing universe alias names and the
        list of universe names contained within that alias set.
    rx : adder.Reactor
        Initialized Reactor object

    Returns
    -------
    ops : List of 3-Tuple or 4-tuple
        The state name, method name to call, and the arguments for that
        method based on the operations information within the input file. If
        this is the first entry of a case block, the first entry will be the
        case name (and hence, we will have a 4-tuple)
    """

    ops = []

    if "operations" in config:
        cfg_ops = config["operations"]
    else:
        # Nothing to do, just return the blank operations
        return ops

    # Now we can go through and process each case
    # keep track of the cumulative time steps in multiple depletion blocks.
    cumulative_time_steps = 0
    for case_data in cfg_ops.values():

        # Get the name for this case
        label = case_data["label"]

        # Now we can progress through each type of operation
        # and create the data we need;
        # we had to get "label" first because we need it to
        # set up the deplete case
        i = -1
        for subsection, ssec_data in case_data.items():
            if subsection != "label":
                i += 1

            if subsection.startswith("deplete"):
                # Create arguments for the deplete method
                method_name = "deplete"
                delta_ts = ssec_data["durations"]
                if "powers" in ssec_data:
                    powers = ssec_data["powers"]
                    fluxes = None
                else:
                    fluxes = ssec_data["fluxes"]
                    powers = None
                depletion_method = ssec_data["depletion_method"]
                depletion_substeps = ssec_data["depletion_substeps"]
                execute_endpoint = ssec_data["execute_endpoint"]

                method_args = (delta_ts, cumulative_time_steps, powers,
                               fluxes, depletion_substeps, depletion_method,
                               execute_endpoint)
                cumulative_time_steps += len(delta_ts)
                # Store the results
                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))
            elif subsection.startswith("shuffle"):
                # First inject our alias data before we process
                obj_type = ssec_data["type"]
                if obj_type == "material":
                    aliases = mat_aliases
                elif obj_type == "universe":
                    aliases = uni_aliases
                new_data = _expand_aliases_in_shuffle(ssec_data, aliases,
                                                      subsection)
                for shuffle, shuffle_data in new_data.items():
                    # Create arguments for the shuffle method
                    method_name = "shuffle"
                    move_start = shuffle_data["moves"][0]
                    move_remainder = shuffle_data["moves"][1:]
                    method_args = (move_start, move_remainder, obj_type)

                    # Store the results
                    if i == 0:
                        ops.append(
                            (label, subsection, method_name, method_args))
                    else:
                        ops.append((subsection, method_name, method_args))

            elif subsection.startswith("revolve"):
                sssec_str = "[operations][[{}]][[[revolve]]]".format(label)
                # NOTE: revolves/flips assume (0,0,0) is the upper left,
                # top plane; the last entry is lower right, bottom plane
                # and data are provided in increasing x order, then
                # down a row, increase x again, ... . Then repeat for
                # the next plane down.

                # Pull out the data we need
                obj_type = ssec_data["type"]
                geometry_type = ssec_data["geometry"]
                obj_list = ssec_data["set"]
                obj_list_shape = ssec_data["shape"]
                xy_degrees = ssec_data["xy_degrees"]
                z_flip = ssec_data["z_flip"]

                if obj_type == "material":
                    aliases = mat_aliases
                elif obj_type == "universe":
                    aliases = uni_aliases

                # First inject our alias data
                full_obj_list = _inject_alias(obj_list, aliases)
                # Check that the shape is correct
                expected_flat_length = np.prod(obj_list_shape)
                if expected_flat_length != len(full_obj_list):
                    msg = sssec_str + \
                        " shape ({}) ".format(obj_list_shape) + \
                        "will not fit all data, after " + \
                        "accounting for aliased entries."
                    logger.log("error", msg)
                # Convert to a numpy array with the right
                # shape since numpy provides convenient
                # rotation facilities
                objects = \
                    np.array(full_obj_list).reshape(obj_list_shape)
                # Now we will revolve as needed
                if geometry_type == "cartesian":
                    num_rotations = xy_degrees // 90
                    # If we have a 0, 360, or 180 degree rotation, then the
                    # matrix doesnt need to be square in XY.
                    if num_rotations in [1, 3]:
                        # Error if the shape is invalid
                        if objects.shape[1] != objects.shape[2]:
                            msg = sssec_str + " shape({})".format(obj_list_shape) + \
                                "must have the same Y- and X- dimensions for " + \
                                "a 90 or 270 degree rotation."
                            logger.log("error", msg)
                    # Do the rotation
                    rot_objects = np.rot90(objects, num_rotations, axes=(1, 2))
                elif geometry_type == "hexagonal":
                    logger.log("error", "Hexagonal rotations are not "
                               "yet supported")

                # Finally, if we are z-flipping, do that with flipud
                # Note this doesnt care about the matrix shape
                if z_flip:
                    rot_objects = np.flipud(rot_objects)

                paths = _convert_shuffle(objects, rot_objects)

                # Finally set the method function info
                method_name = "shuffle"
                for move_start, move_remainder in paths.items():
                    method_args = (move_start, move_remainder, obj_type)

                    # Store the results
                    if i == 0:
                        ops.append(
                            (label, subsection, method_name, method_args))
                    else:
                        ops.append((subsection, method_name, method_args))

            elif subsection.startswith("transform"):
                # Pull out the data we need
                if "group_name" in ssec_data:
                    obj_type = "group"
                    full_obj_set = ssec_data["group_name"]
                    yaw = None
                    pitch = None
                    roll = None
                    angle_units = None
                    matrix = None
                    displacement = [ssec_data["value"], 0., 0.]
                else:
                    obj_type = ssec_data["type"]
                    obj_set = ssec_data["set"]
                    yaw = ssec_data["yaw"]
                    pitch = ssec_data["pitch"]
                    roll = ssec_data["roll"]
                    angle_units = ssec_data["angle_units"]
                    matrix = ssec_data["matrix"]
                    if matrix is not None:
                        matrix = np.array(matrix,
                                          dtype=np.float64).reshape((3, 3))
                    displacement = ssec_data["displacement"]

                    aliases = uni_aliases

                    # Take our set and modify it so aliases are expanded
                    full_obj_set = _inject_alias(obj_set, aliases)

                # Now create a transform for each in the set
                method_name = "transform"
                method_args = (full_obj_set, yaw, pitch, roll, angle_units,
                               matrix, displacement, obj_type)

                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

            elif subsection.startswith("geometry_sweep"):
                # Pull out the data we need
                sweep_group = ssec_data["group_name"]
                sweep_values = ssec_data["values"]

                # Now create a transform for each in the set
                method_name = "geom_sweep"
                method_args = (sweep_group, sweep_values)

                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

            elif subsection.startswith("geometry_search"):
                # Pull out the data we need
                search_set = ssec_data["group_name"]
                grp = rx.control_groups[search_set]
                if grp.type in ["cells", "universes"]:
                    msg = "Using cell or universe-based groups in a " + \
                        "geometry search is a developmental feature " + \
                        "and should not be used for production analyses!"
                    logger.log("warning", msg)

                k_target = ssec_data["k_target"]
                bracket_interval = ssec_data["bracket_interval"]
                target_interval = ssec_data["target_interval"]
                initial_guess = ssec_data["initial_guess"]
                min_act_batches = ssec_data["min_active_batches"]
                max_iterations = ssec_data["max_iterations"]
                uncertainty_fraction = ssec_data["uncertainty_fraction"]

                # Now create a transform for each in the set
                method_name = "geom_search"
                method_args = (search_set, k_target, bracket_interval,
                               target_interval, uncertainty_fraction,
                               initial_guess, min_act_batches, max_iterations)

                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

            elif subsection.startswith("write_input"):
                # This is primarily intended for debugging/unit testing
                # purposes, however it allows the user to dump the
                # neutronics solver's input at this particular stage of the
                # computation
                fname = ssec_data["filename"]
                user_tallies = ssec_data["include_user_tallies"]
                user_output = ssec_data["include_user_output"]
                adder_tallies = ssec_data["include_adder_tallies"]

                method_name = "write_neutronics_input"
                method_args = (fname, adder_tallies, user_tallies, user_output)
                # Store the results
                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

            elif subsection.startswith("write_depletion_lib"):
                # Pull out the data we need
                filename = ssec_data["filename"]
                mat_names = ssec_data["materials"]
                lib_names = ssec_data["lib_names"]
                mode = ssec_data["mode"]
                # Take our set and modify it so aliases are expanded
                mat_names = _inject_alias(mat_names, mat_aliases)
                if lib_names is None:
                    lib_names = [None] * len(mat_names)

                method_name = "write_depletion_library_to_hdf5"
                method_args = (filename, mat_names, lib_names, mode)

                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

            elif subsection.startswith("calc_volume"):
                # Get the bounding box information
                target_unc = ssec_data["target_uncertainty"]
                max_hist = ssec_data["maximum_histories"]
                vol_data = {"type": ssec_data["vol_type"]}
                if ssec_data["vol_type"] == "box":
                    lower_left = ssec_data["lower_left"]
                    upper_right = ssec_data["upper_right"]
                    vol_data["lower_left"] = lower_left
                    vol_data["upper_right"] = upper_right
                elif ssec_data["vol_type"] == "cylinder":
                    bottom = ssec_data["cylinder_bottom"]
                    ro = ssec_data["cylinder_radius"]
                    h = ssec_data["cylinder_height"]
                    vol_data["bottom"] = bottom
                    vol_data["radius"] = ro
                    vol_data["height"] = h
                only_depleting = ssec_data["only_depleting"]
                method_name = "calc_volumes"
                method_args = (vol_data, target_unc, max_hist, only_depleting)
                if i == 0:
                    ops.append((label, subsection, method_name, method_args))
                else:
                    ops.append((subsection, method_name, method_args))

    return ops


def _inject_alias(data, aliases):
    """Given a list of material or universe name strings, parse it to
    see if an alias should be replaced with what it is an alias for. If
    so, do that replacement.
    """
    new_data = []
    for item in data:
        # We will first check directly for an alias, then for
        # an indexed alias, and otherwise, no alias

        if item in aliases:
            # Then we have an alias, so what we want to do is put in
            # each entry of the alias' targets.
            new_data.extend(aliases[item])
        elif re.search(r"\[\d+\]", item):
            # Then we have an indexed alias, implying the [#] should be
            # applied to all the individual items as well
            unbracket, bracket = item.split("[")
            # Add the "[" back in to bracket
            bracket = "[" + bracket
            if unbracket in aliases:
                new_items = [a_item + bracket for a_item in aliases[unbracket]]
                new_data.extend(new_items)
            else:
                new_data.append(item)
        else:
            # Nothing to be done, just include this new data
            new_data.append(item)
    return new_data


def _expand_aliases_in_shuffle(shuffle_cfg, aliases, key):
    """Adds configuration items for each alias shuffle"""

    expanded_cfg = OrderedDict()
    orig_moves = shuffle_cfg["moves"]

    # We do this by creating a 2d list of lists ("moves") where the
    # outer dimension is the index in the original list from the config
    # and the inner dimension is the aliased index
    moves = [None for orig_move in orig_moves]

    # Now replace the orig_moves with their alias versions if applicable
    for i, orig_move in enumerate(orig_moves):
        # _inject_alias will figure out if this is an alias, an
        # indexed alias, or no alias and return our inner list
        # accordingly
        moves[i] = _inject_alias([orig_move], aliases)

    # Check to make sure that the inner dimensions of all entries match
    for move in moves[1:]:
        if len(move) != len(moves[0]):
            msg = "The number of items to shuffle, after incorporating " \
                  "aliases must be consistent!"
            logger.log("error", msg)

    # Now we want to make a config entry for each of the inner
    # dimension of moves
    # This is more easily done with a transpose of the original
    moves_T = np.array(moves).T.tolist()

    # now we can iterate over the outer dimension of moves_T
    for i in range(len(moves_T)):
        move_set = []
        for j in range(len(moves_T[i])):
            move_set.append(moves_T[i][j])

        # Now we are ready to store the new information as a new config
        # entry
        if len(moves_T[0]) == 0:
            # Then there actually was no alias
            new_key = key
        else:
            new_key = "{}_alias_{}".format(key, i)
        # First make a copy of the old config entry
        expanded_cfg[new_key] = OrderedDict()
        for k, v in shuffle_cfg.items():
            expanded_cfg[new_key][k] = v

        # And just change the moves
        expanded_cfg[new_key]["moves"] = move_set

    return expanded_cfg


def _convert_shuffle(orig, new):
    """Create shuffle paths to use when using the Neutronics.shuffle
    method based on having a 3D array of original and post-rotation/flip
    material or universe names.

    Parameters
    ----------
    orig : np.ndarray of str
        Three-dimensional array of the mat/uni names in their original
        locations.
    new : np.ndarray of str
        Three-dimensional array of the mat/uni names in their final
        locations.

    Returns
    -------
    paths : OrderedDict of List of str
        The set of moves to make.
    """

    logger.log("debug", "Converting rotations or flips to shuffles")
    # Make sure orig and new are correct type and have the right number
    # of entries
    if orig.size != new.size:
        msg = "Shapes do not match!"
        logger.log("error", msg)

    # Singular moves captures where each location in orig moved to; this
    # will later be reduced to paths
    singular_moves = OrderedDict()

    # Go through each location in orig, find where it is in new,
    # take the indices of its location in new, use those indices on
    # orig to find what it replaced.
    for k in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            for i in range(orig.shape[2]):
                orig_val = orig[k, j, i]
                # Find where orig_val ended up in new
                new_idx = np.where(new == orig_val)
                # There can only be one place this moved to, so look for
                # duplicates
                if len(new_idx[0]) > 1:
                    msg = "Cannot move single piece to multiple locations!"
                    logger.log("error", msg)
                new_k, new_j, new_i = new_idx[0][0], new_idx[1][0], \
                    new_idx[2][0]
                replaced_by_orig_val = orig[new_k, new_j, new_i]
                singular_moves[orig_val] = replaced_by_orig_val

    # Now we know where everything moved, all that is left is to append
    # the sets together
    path_lists = []
    for start, end in singular_moves.items():
        # Get the path for this start point
        movement_set = []
        # Add our first connection
        movement_set.extend([start, end])
        # Get the rest
        next_start = end
        while True:
            next_end = singular_moves[next_start]
            if next_end != start:
                movement_set.append(next_end)
                next_start = next_end
            else:
                # Then we finished our ring cycle
                break
        # Ok we dont actually want a path for each entry, instead we
        # only keep the ones that are unique
        # We will find uniqueness by converting the list to a set, which
        # doesnt care about order only values, and compare that way
        if len(path_lists) > 0:
            no_matches = True
            for path_list in path_lists:
                if set(movement_set) == set(path_list):
                    no_matches = False
            if no_matches:
                path_lists.append(movement_set)
        else:
            path_lists.append(movement_set)

    # The 'hard' stuff is done, now just convert to the expected format
    # where its paths[start] = rest of path
    paths = OrderedDict()
    for path_list in path_lists:
        paths[path_list[0]] = path_list[1:]

    return paths


def input_echo(config):
    import json
    msg = json.dumps(config.dict(), indent=2)
    logger.log("info_file", "INPUT ECHO:\n\n" + msg + "\n")


def _all_equal(iterable):
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
