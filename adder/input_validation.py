from collections import OrderedDict

import adder.constants as constants
from adder.data import is_isotope
from adder.type_checker import *
from adder.loggedclass import LoggedClass
logger = LoggedClass(0, __name__)


def validate(config):
    """ Validates the nested dictionary present in config,
    converts types, and introduces default values to simplify downstream
    processing."""

    _validate_metadata(config)
    if "control_groups" in config:
        group_names = _validate_control_groups(config["control_groups"])
    else:
        group_names = []
    if "materials" in config:
        _validate_materials(config["materials"])
        # No else needed since this block is optional
    if "universes" in config:
        _validate_universes(config["universes"])
        # No else needed since this block is optional
    if "operations" in config:
        num_time_steps = _validate_ops(config["operations"], config,
                                       group_names)
        # No else needed since this block is technically optional
    if "msr" in config:
        if config["depletion_solver"].startswith("msr"):
            _validate_msr(config["msr"], num_time_steps)
        else:
            msg = "The [msr] block was provided, however " + \
                "the depletion solver is not `msr`, `msr16`, or 'msr48'"
            raise ValueError(msg)


def _validate_metadata(config):
    level = "metadata"
    # Check the required values
    keys = ["case_name", "neutronics_solver", "neutronics_exec",
            "neutronics_input_file", "depletion_solver",
            "depletion_library_file", "depletion_library_name"]
    req_type = str
    check_required(config, keys, req_type, level)

    # Check the strings with blanks (None) as default
    keys = ["mpi_command", "neutronics_library_file"]
    req_type = str
    default = None
    check_optional(config, keys, req_type, level, default)

    # Check the strings which have other defaults
    keys = ["output_hdf5"]
    req_type = str
    default = constants.DEFAULT_H5_OUTPUT
    check_optional(config, keys, req_type, level, default)

    # Check those which depend on other values
    keys = ["depletion_exec"]
    req_type = str
    if config["depletion_solver"].lower() in constants.INTERNAL_DEPLETION_SOLVERS:
        # Then depletion_exec is superflous
        config["depletion_exec"] = ""
    else:
        # Then it is required since it is an external solver
        check_required(config, keys, req_type, level)

    # Check those required to be in a set
    keys = ["depletion_method"]
    req_vals = constants.VALID_DEPLETION_METHODS
    default = constants.DEFAULT_DEPLETION_METHOD
    check_in_set(config, keys, req_vals, level, default)

    # Check logical entries
    keys = ["use_depletion_library_xs"]
    default = constants.USE_DEPLETION_LIBRARY_XS
    check_optional(config, keys, str_to_bool, level, default)

    keys = ["apply_reactivity_threshold_to_initial_inventory"]
    default = constants.DEFAULT_REACTIVITY_THRESH_TO_INITIAL
    check_optional(config, keys, str_to_bool, level, default)

    # If they only give num_threads, then set that to be what num_neut_threads
    # and num_depl_threads are
    if "num_threads" in config:
        check_num_and_range(config, ["num_threads"], int, level, 1, min_val=1)
        num_threads = config["num_threads"]
        config["num_neutronics_threads"] = num_threads
        config["num_depletion_threads"] = num_threads
        del config["num_threads"]
    else:
        # Then check each of them
        keys = ["num_neutronics_threads", "num_depletion_threads"]
        check_num_and_range(config, keys, int, level, 1, min_val=1)
    # Check the integers
    keys = ["num_mpi_processes", "depletion_chunksize", "depletion_substeps"]
    defaults = [1, 1000, 10]
    req_type = int
    min_val = 1
    for key, default in zip(keys, defaults):
        check_num_and_range(config, [key], req_type, level, default,
                            min_val=min_val)

    # And finally the floating points
    keys = ["neutronics_reactivity_threshold"]
    req_type = float
    default = constants.DEFAULT_REACTIVITY_THRESH
    check_optional(config, keys, req_type, level, default)


def _validate_control_groups(config):
    # We expect subsections that have the group name
    group_names = []
    for name, subcfg in config.items():
        level = "[control_groups][[{}]]".format(name)
        group_names.append(name)
        # There should be the following entries:
        # type, set, axis. Angle is optional
        keys = ["type", "axis"]
        check_required(subcfg, keys, str, level)
        # Verify that the type is one of the supported types
        check_in_set(subcfg, ["type"], constants.VALID_GEOM_SWEEP_TYPES,
                     level, None)
        # Verify that the type is one of the supported types
        check_in_set(subcfg, ["axis"], constants.VALID_GEOM_SWEEP_AXES,
                     level, None)
        # And the required list of ids
        keys = ["set"]
        check_lists(subcfg, keys, str, level, min_num=1, required=True)
        # Get the optional angle_units
        check_in_set(subcfg, ["angle_units"], constants.VALID_ANGLE_TYPES,
                     level, constants.DEFAULT_ANGLE_TYPE)
    return group_names


def _validate_materials(config):
    if "metadata" in config:
        _validate_matuni_metadata(config["metadata"], "[materials]")
    if "aliases" in config:
        _validate_aliases(config["aliases"], "[materials]")
    if "supply" in config:
        _validate_storage_and_supply(config["supply"], "[materials][[supply]]")
    if "storage" in config:
        _validate_storage_and_supply(config["storage"],
                                     "[materials][[storage]]")


def _validate_universes(config):
    if "metadata" in config:
        _validate_matuni_metadata(config["metadata"], "[universes]")
    if "aliases" in config:
        _validate_aliases(config["aliases"], "[universes]")
    if "supply" in config:
        _validate_storage_and_supply(config["supply"], "[universes][[supply]]")
    if "storage" in config:
        _validate_storage_and_supply(config["storage"],
                                     "[universes][[storage]]")


def _validate_matuni_metadata(config, top_level):
    level = top_level + "[[metadata]]"

    # The subsections here are [range_#], [list_#], or [item_#]
    # We treat each separately as they have different needs
    for mat_ssec, subcfg in config.items():
        sublevel = level + "[[[{}]]]".format(mat_ssec)

        # Handle the names, which are the same for range and list
        if mat_ssec.startswith("range") or mat_ssec.startswith("list"):
            # Check the names
            keys = ["names"]
            check_lists(subcfg, keys, str, sublevel, required=False,
                        set_None=True)

        # Handle just range
        if mat_ssec.startswith("range"):
            # Check required integers
            keys = ["neutronics_id_start", "neutronics_id_end"]
            req_type = int
            check_required(subcfg, keys, req_type, sublevel)
            check_num_and_range(subcfg, keys, req_type, sublevel, None,
                                min_val=1)

            # Check the exclude_neutronics_ids list
            keys = ["exclude_neutronics_ids"]
            check_lists(subcfg, keys, int, sublevel, required=False,
                        set_None=True, min_val=1)

            keys = ["density"]
            req_type = float
            check_optional(subcfg, keys, req_type, sublevel, None)

        elif mat_ssec.startswith("list"):
            # Handle just list
            # Check the neutronics_ids list
            keys = ["neutronics_ids"]
            check_lists(subcfg, keys, int, sublevel, required=True, min_val=1)

            # Check densities
            keys = ["densities"]
            check_lists(subcfg, keys, float, sublevel, required=False,
                        set_None=True)
            if subcfg["densities"] is None:
                subcfg["densities"] = \
                    [None for i in range(len(subcfg["neutronics_ids"]))]

        elif mat_ssec.startswith("item"):
            # Handle just individual items
            keys = ["neutronics_id"]
            req_type = int
            check_required(subcfg, keys, req_type, sublevel)
            check_num_and_range(subcfg, keys, req_type, sublevel, None,
                                min_val=1)

            keys = ["name"]
            req_type = str
            check_optional(subcfg, keys, req_type, sublevel, None)

            keys = ["density"]
            req_type = float
            check_optional(subcfg, keys, req_type, sublevel, None)

        else:
            msg = "Invalid {} subsection name: {}".format(top_level, sublevel)
            raise ValueError(msg)

        # Now handle the values that are the same regardless of item/list/range
        if top_level == "[materials]":
            # Do those only in materials
            keys = ["volume"]
            check_optional(subcfg, keys, float, sublevel, None)

            keys = ["depleting"]
            default = constants.DEFAULT_DEPLETING
            check_optional(subcfg, keys, str_to_bool, sublevel, default)

            keys = ["non_depleting_isotopes"]
            check_lists(subcfg, keys, str, sublevel, required=False,
                        set_None=True)

            keys = ["use_default_depletion_library"]
            default = constants.DEFAULT_MATL_USE_DEFAULT_DEPL_LIB
            check_optional(subcfg, keys, str_to_bool, sublevel, default)

            keys = ["apply_reactivity_threshold_to_initial_inventory"]
            # A "None" indicates use global default
            check_optional(subcfg, keys, str_to_bool, sublevel, None)


def _validate_aliases(config, top_level):
    level = top_level + "[[aliases]]"
    # aliases can start with any subsection name

    # Step through each subsection
    for alias_ssec, subcfg in config.items():
        sublevel = level + "[[[{}]]]".format(alias_ssec)
        check_optional(subcfg, ["name"], str, sublevel, alias_ssec)
        # Verify that the name is not a digit
        if subcfg["name"].isdigit():
            _integral_errormsg("name", sublevel)
        check_lists(subcfg, ["set"], str, sublevel, min_num=1, required=True)


def _validate_storage_and_supply(config, level):

    # We have either redefine, copy, or new subsections
    for my_ssec, subcfg in config.items():
        if my_ssec == "new":
            raise NotImplementedError("new not yet supported")
        if my_ssec not in ["copy", "redefine"]:
            raise ValueError("Invalid Storage/Supply "
                             "Subsection: {}".format(my_ssec))

        sublevel = level + "[[[{}]]]".format(my_ssec)

        # Now handle redefine and copy together since they are mostly
        # the same (except for presence of name/names in copy)

        # The subsubsections here are [range_#], [list_#], or [item_#]
        # We treat each separately as they have different needs
        for my_sssec, subsubcfg in subcfg.items():
            subsublevel = sublevel + "[[[[{}]]]]".format(my_sssec)

            # For copy, handle the names, which are the same for range
            # and list
            if my_ssec == "copy":
                if my_sssec.startswith("range") or my_sssec.startswith("list"):

                    # Check the names
                    keys = ["names"]
                    check_lists(subsubcfg, keys, str, subsublevel,
                                required=True, set_None=True)
                elif my_sssec.startswith("item"):
                    keys = ["name"]
                    req_type = str
                    check_required(subsubcfg, keys, req_type, subsublevel)

            # Handle just range
            if my_sssec.startswith("range"):
                # Check required integers
                keys = ["neutronics_id_start", "neutronics_id_end"]
                req_type = int
                check_required(subsubcfg, keys, req_type, subsublevel)
                check_num_and_range(subsubcfg, keys, req_type, subsublevel,
                                    None, min_val=1)

                # Check the exclude_neutronics_ids list
                keys = ["exclude_neutronics_ids"]
                check_lists(subsubcfg, keys, int, subsublevel, required=False,
                            set_None=True, min_val=1)

            elif my_sssec.startswith("list"):
                # Handle just list
                # Check the neutronics_ids list
                keys = ["neutronics_ids"]
                check_lists(subsubcfg, keys, int, subsublevel, required=True,
                            min_val=1)

            elif my_sssec.startswith("item"):
                # Handle just individual items
                keys = ["neutronics_id"]
                req_type = int
                check_required(subsubcfg, keys, req_type, subsublevel)
                check_num_and_range(subsubcfg, keys, req_type, subsublevel,
                                    None, min_val=1)

            else:
                msg = "Invalid material subsection name: {}".format(
                    subsublevel)
                raise ValueError(msg)


def _validate_ops(config, top_config, group_names):
    # The subsections are the cases and each case contains a number of
    # operation subsubsections that will be validated in their own funcs
    level = "[operations]"
    # counter to keep track of number of duration steps
    num_time_steps = 0
    for case, case_cfg in config.items():
        case_level = level + "[[{}]]".format(case)
        # At this level we only have a case label and then the subsects
        check_optional(case_cfg, ["label"], str, case_level, case)

        # Now process the case subsections
        for ssec, ssec_cfg in case_cfg.items():
            sublevel = case_level + "[[[{}]]]".format(ssec)

            if ssec.startswith("deplete"):
                num_time_steps += _validate_ops_deplete(ssec_cfg, top_config,
                                                        sublevel)
            elif ssec.startswith("shuffle"):
                _validate_ops_shuffle(ssec_cfg, top_config, sublevel)
            elif ssec.startswith("revolve"):
                _validate_ops_revolve(ssec_cfg, top_config, sublevel)
            elif ssec.startswith("write_input"):
                _validate_ops_write_input(ssec_cfg, top_config, sublevel)
            elif ssec.startswith("write_depletion_lib"):
                _validate_ops_write_depllib(ssec_cfg, top_config, sublevel)
            elif ssec.startswith("calc_volume"):
                _validate_ops_calc_volume(ssec_cfg, top_config, sublevel)
            elif ssec.startswith("transform"):
                _validate_ops_transform(ssec_cfg, top_config, sublevel,
                                        group_names)
            elif ssec.startswith("geometry_sweep"):
                _validate_ops_geom_sweep(ssec_cfg, top_config, sublevel,
                                         group_names)
            elif ssec.startswith("geometry_search"):
                _validate_ops_geom_search(ssec_cfg, top_config, sublevel,
                                          group_names)
    return num_time_steps


def _validate_ops_deplete(config, top_config, level):
    # "durations" are always required
    keys = ["durations"]
    check_lists(config, keys, float, level, min_num=1, required=True,
                min_val=0.)

    # Either powers or fluxes must be provided
    if ("powers" in config and "fluxes" in config) or \
        ("powers" not in config and "fluxes" not in config):
        msg = "One of 'powers' or 'fluxes'"
        raise ValueError(_reqd_errormsg(msg, level))
    if "powers" in config:
        check_lists(config, ["powers"], float, level, min_num=1, required=True,
                    min_val=0.)
    if "fluxes" in config:
        check_lists(config, ["fluxes"], float, level, min_num=1, required=True,
                    min_val=0.)

    # The next two are optional values that can overwrite the global
    # defaults from metadata, so get that metadata value for the default
    # Check those required to be in a set
    keys = ["depletion_method"]
    req_vals = constants.VALID_DEPLETION_METHODS
    default = top_config[keys[0]]
    check_in_set(config, keys, req_vals, level, default)

    # Check depletion_substeps integer
    keys = ["depletion_substeps"]
    req_type = int
    default = top_config[keys[0]]
    min_val = 1
    check_num_and_range(config, keys, req_type, level, default,
                        min_val=min_val)

    # Check if the user wants the endpoint to be calculated or not
    keys = ["execute_endpoint"]
    check_optional(config, keys, str_to_bool, level, True)

    return len(config['durations'])


def _validate_ops_shuffle(config, top_config, level):
    # Get the type
    keys = ["type"]
    # The valid move types is set by the type of neutronics solver, so
    # get the valid list now
    solver = top_config["neutronics_solver"].lower()
    req_vals = constants.VALID_MOVE_TYPES[solver]
    default = constants.DEFAULT_MOVE_TYPE[solver]
    check_in_set(config, keys, req_vals, level, default)

    # Set the move list
    keys = ["moves"]
    check_lists(config, keys, str, level, min_num=2, required=True)


def _validate_ops_revolve(config, top_config, level):
    # Get the type
    keys = ["type"]
    # The valid move types is set by the type of neutronics solver, so
    # get the valid list now
    solver = top_config["neutronics_solver"].lower()
    req_vals = constants.VALID_MOVE_TYPES[solver]
    default = constants.DEFAULT_MOVE_TYPE[solver]
    check_in_set(config, keys, req_vals, level, default)

    # Set the set list
    keys = ["set"]
    check_lists(config, keys, str, level, min_num=1, required=True)

    # Set the shape
    keys = ["shape"]
    check_lists(config, keys, int, level, min_num=3, max_num=3,
                required=True, min_val=1)

    # Geometry type
    keys = ["geometry"]
    req_vals = ["cartesian", "hexagonal"]
    default = "cartesian"
    check_in_set(config, keys, req_vals, level, default)

    # xy degrees
    keys = ["xy_degrees"]
    if config["geometry"] == "cartesian":
        req_vals = [0, 90, 180, 270, 360]
    else:
        req_vals = [0, 60, 120, 180, 240, 300, 360]
    check_in_set(config, keys, req_vals, level, None, cast_to=int)

    # z flip
    keys = ["z_flip"]
    check_optional(config, keys, str_to_bool, level, False)


def _validate_ops_write_input(config, top_config, level):
    # filename string
    keys = ["filename"]
    check_required(config, keys, str, level)

    # file writing information
    keys = ["include_user_tallies", "include_user_output",
            "include_adder_tallies"]
    check_optional(config, keys, str_to_bool, level, True)


def _validate_ops_write_depllib(config, top_config, level):
    # filename string
    keys = ["filename"]
    check_required(config, keys, str, level)

    keys = ["materials"]
    check_lists(config, keys, str, level, min_num=1, required=True)

    keys = ["lib_names"]
    check_lists(config, keys, str, level, min_num=1, required=False,
                set_None=True)

    keys = ["mode"]
    check_in_set(config, keys, ['r', 'r+', 'w', 'w-', 'x', 'a'], level,
                 default='w')


def _validate_ops_calc_volume(config, top_config, level):
    # Check the optional float
    check_num_and_range(config, ["target_uncertainty"], float, level,
                        constants.DEFAULT_VOL_TARGET_UNC, min_val=0.)
    check_num_and_range(config, ["maximum_histories"], int, level,
                        constants.DEFAULT_VOL_MAX_HIST, min_val=1)

    # Figure out our volume type
    if "lower_left" in config and "upper_right" in config:
        vol_type = "box"
        # Now ensure there is not the cylindrical inputs
        for param in ["cylinder_bottom", "cylinder_radius", "cylinder_height"]:
            if param in config:
                msg = "If `lower_left` and `upper_right` are specified," + \
                    "{} cannot be specified as well".format(param)
                raise ValueError(msg)
    elif "cylinder_bottom" in config and "cylinder_radius" in config and \
        "cylinder_height" in config:
        vol_type = "cylinder"
        # Now ensure there is not any box inputs
        for param in ["lower_left", "upper_right"]:
            if param in config:
                msg = "If `lower_left` or `upper_right` cannot be " + \
                    "specified with cylinder parameters"
                raise ValueError(msg)

    if vol_type == "box":
        keys = ["lower_left", "upper_right"]
        check_lists(config, keys, float, level, min_num=3, required=True)
    elif vol_type == "cylinder":
        keys = ["cylinder_bottom"]
        check_lists(config, keys, float, level, min_num=3, required=True)
        keys = ["cylinder_radius", "cylinder_height"]
        check_required(config, keys, float, level)

    config["vol_type"] = vol_type

    keys = ["only_depleting"]
    default = True
    check_optional(config, keys, bool, level, default)


def _validate_ops_transform(config, top_config, level, group_names):
    # Figure out if this is a group or non-group move
    if "group_name" in config:
        # Then we just need to get the group
        check_in_set(config, ["group_name"], group_names, level, default=None)
        check_required(config, ["value"], float, level)
    else:
        # Get the type
        keys = ["type"]
        # The valid transform types is set by the type of neutronics solver, so
        # get the valid list now
        solver = top_config["neutronics_solver"].lower()
        req_vals = constants.VALID_TRANSFORM_TYPES[solver]
        default = constants.DEFAULT_TRANSFORM_TYPE[solver]
        check_in_set(config, keys, req_vals, level, default)

        # Set the set list
        keys = ["set"]
        check_lists(config, keys, str, level, min_num=1, required=True)

        # Angles
        keys = ["yaw", "pitch", "roll"]
        default = 0.
        check_optional(config, keys, float, level, default)

        # angle_units
        check_in_set(config, ["angle_units"], constants.VALID_ANGLE_TYPES, level,
                    constants.DEFAULT_ANGLE_TYPE)

        # Matrix
        keys = ["matrix"]
        check_lists(config, keys, float, level, min_num=9, max_num=9,
                    required=False, set_None=True)
        if config[keys[0]] is not None:
            # It is present so set the angles to None
            config["yaw"] = None
            config["pitch"] = None
            config["roll"] = None
            config["angle_units"] = "radians"

        # displacement
        keys = ["displacement"]
        check_lists(config, keys, float, level, min_num=3, max_num=3,
                    required=False, set_None=True)
        if config[keys[0]] is None:
            config[keys[0]] = [0., 0., 0.]


def _validate_ops_geom_sweep(config, top_config, level, group_names):
    check_in_set(config, ["group_name"], group_names, level, default=None)

    # And now we can check the range and list blocks
    values = []
    for key in config:
        sublevel = level + "[[[[{}]]]]".format(key)
        # First we process the range;
        # Note we check to see if it startswith range as then the user can
        # do multiple ranges with range_1, range_2, etc as a form of mesh
        # refinement perhaps
        if key.lower().startswith("range"):
            # Then we require start, end, and number to be provided.
            keys = ["start", "end"]
            check_required(config[key], keys, float, sublevel)
            check_required(config[key], ["number"], int, sublevel)
            # Make sure number is > 1
            check_num_and_range(config[key], ["number"], int, sublevel, 1,
                                min_val=2)
            # endpoint is optional and defaults to True
            check_optional(config[key], ["endpoint"], str_to_bool, sublevel,
                           constants.DEFAULT_GEOM_SWEEP_RANGE_ENDPOINT)
            # Get the values from this range and append it to the
            # values array
            this_range = np.linspace(config[key]["start"],
                                     config[key]["end"],
                                     num=config[key]["number"],
                                     endpoint=config[key]["endpoint"])
            values.extend(this_range.tolist())
            # Remove this key for simplified later processing
            del config[key]
        # Now process the list
        # Note we check to see if it startswith list as then the user can
        # do multiple lists with list_1, list_2, etc
        elif key.lower().startswith("list"):
            # Then we only require values
            check_lists(config[key], ["values"], float, sublevel, min_num=1,
                        required=True)
            # Ok, we have our values, append to the running list
            values.extend(config[key]["values"])
            del config[key]

    # Now store the values as an array in the input config info
    # Note that we do not remove duplicate values as that is a simple burden
    # to place on the user, and not changing the ordering and indices of each
    # case will make it easier for them to parse the output.
    config["values"] = values


def _validate_ops_geom_search(config, top_config, level, group_names):
    check_in_set(config, ["group_name"], group_names, level, default=None)

    # Now the required floats
    keys = ["k_target", "target_interval"]
    check_required(config, keys, float, level)
    check_greater_than("k_target", config["k_target"], 0.)
    check_greater_than("target_interval", config["target_interval"], 0.)
    # The required list of floats
    check_lists(config, ["bracket_interval"], float, level, 2, 2, True)
    # Optional parameters
    check_optional(config, ["initial_guess"], float, level,
                   config["bracket_interval"][1])
    check_greater_than("initial_guess", config["initial_guess"],
                       config["bracket_interval"][0], equality=True)
    check_less_than("initial_guess", config["initial_guess"],
                    config["bracket_interval"][1], equality=True)
    check_optional(config, ["min_active_batches"], int, level,
                   constants.DEFAULT_GEOM_SEARCH_MIN_ACTIVE_BATCHES)
    check_greater_than("min_active_batches", config["min_active_batches"], 3)
    check_optional(config, ["max_iterations"], int, level,
                   constants.DEFAULT_GEOM_SEARCH_MAX_ITERATIONS)
    check_greater_than("max_iterations", config["max_iterations"], 0)
    check_optional(config, ["uncertainty_fraction"], float, level,
                   constants.DEFAULT_GEOM_SEARCH_UNCERTAINTY_FRACTION)


def _validate_msr(config, num_time_steps):
    level = "[msr]"

    # Check the params and sections
    for key in config:
        sublevel = level + "[[{}]]".format(key)
        subcfg = config[key]
        if key.lower().startswith("system"):
            # we have two required strings to check
            check_required(subcfg, ["name", "flow_start"], str, sublevel)

            # And the flowrate
            check_required(subcfg, ["flowrate"], float, sublevel)

            # Now there can be a feed block
            subkey = "feed"
            if subkey in subcfg:
                sscfg = subcfg[subkey]
                sslevel = sublevel + "[[[{}]]]".format(subkey)
                # Check the metadata info first
                check_required(sscfg, ["feed_rate_units", "vector_units"],
                               str, sslevel)
                check_in_set(sscfg, ["feed_rate_units"],
                             ["kg/sec", "kg/day", "atoms/sec", "kg/s", "kg/d",
                              "atoms/s"], sslevel, None)
                check_in_set(sscfg, ["vector_units"],
                             ["ao", "wo", "a/o", "a\\o", r"a\o",
                              "w/o", "w\\o", r"w\o"], sslevel, None)

                # Note that we allow negative values so that a feed
                # can be used as a constant removal
                check_lists(sscfg, ["feed_rate"], float, sslevel)
                check_lists(sscfg, ["feed_material"], str, sslevel)
                check_lists(sscfg, ["feed_mixture"], str, sslevel)
                check_lists(sscfg, ["feed_material"], str, sslevel)
                check_lists(sscfg, ["density"], float, sslevel)

                # duration, power/fluxes and feed rate must be the same length
                if len(sscfg["feed_rate"]) != num_time_steps and len(sscfg["feed_rate"]) > 1:
                    msg = "The total number of durations in the deplete " + \
                        "operations must be equal to number of feed_rate " + \
                        "values."
                    logger.log("error", msg)

                # feed_rate, feed_material, feed_mixture must be same length
                length = len(sscfg["feed_rate"])
                if any(len(lst) != length
                       for lst in [sscfg["feed_material"],
                                   sscfg["feed_mixture"], sscfg["density"]]):
                    msg = "feed_rate, feed_material, feed_mixture, and " + \
                        "density must be lists of the same length"
                    logger.log("error", msg)

                if len(sscfg["feed_rate"]) == 1:
                    msg = "Only 1 feed rate value provided, CONSTANT feed " +\
                        "rate assumed. "
                    logger.log("info", msg)

                # Do any unit conversion so we dont have to deal with it later
                if sscfg["feed_rate_units"] in ["kg/day", "kg/d"]:
                    # Put all mass-based units to kg/sec, including conversion
                    sscfg["feed_rate_units"] = "kg/sec"
                    sscfg["feed_rate"] = \
                        [v / 86400. for v in sscfg["feed_rate"]]
                elif sscfg["feed_rate_units"] == "kg/s":
                    # Just get to one representation, kg/sec
                    sscfg["feed_rate_units"] = "kg/sec"
                elif sscfg["feed_rate_units"] == "atoms/s":
                    sscfg["feed_rate_units"] = "atoms/sec"

                # Do same for vector_units
                if sscfg["vector_units"] in ["a/o", "a\\o", r"a\o"]:
                    sscfg["vector_units"] = "ao"
                elif sscfg["vector_units"] in ["w/o", "w\\o", r"w\o"]:
                    sscfg["vector_units"] = "wo"

                for subsubkey in sscfg:
                    if subsubkey in ["feed_rate", "feed_material",
                                     "feed_mixture", "feed_rate_units",
                                     "density", "density_units",
                                     "vector_units"]:
                        continue
                    elif not subsubkey.lower().startswith("material"):
                        msg = "Only [[[material_#]]] blocks " + \
                            "are allowed in a [msr][[system]][[[feed]]] block!"
                        logger.log("error", msg)

                    ssscfg = sscfg[subsubkey]
                    ssslevel = sublevel + "[[[[{}]]]]".format(subkey)

                    # Then we must have "names" (list of str) and
                    # "vector" (list of float) and that they are the same size
                    check_lists(ssscfg, ["names"], str, ssslevel)
                    check_lists(ssscfg, ["vector"], float, ssslevel,
                                min_num=len(ssscfg["names"]),
                                max_num=len(ssscfg["names"]))

                    # Now lets make sure the "names" are actual element or
                    # isotope names
                    check_with_function(ssscfg, ["names"], is_isotope,
                                        ssslevel)

                    # make list of material names
                    if 'material_names' in locals():
                        material_names.append(subsubkey.strip('material_'))
                    else:
                        material_names = [subsubkey.strip('material_')]

                # Now use feed_material and feed_mixture to organize isotope
                # amount and rearrange so we have a dictionary relating names
                # to the vector
                if "feed_vector" not in sscfg:
                    sscfg["feed_vector"] = []
                # loop through feed_material and feed_mixture for each
                # depletion step
                for i in range(len(sscfg['feed_material'])):
                    materials = sscfg['feed_material'][i].strip('()')
                    materials = materials.split()
                    mixture = sscfg['feed_mixture'][i].strip('()')
                    mixture = mixture.split()
                    # if feed_material and feed_mixture are not same length
                    # raise error
                    if len(materials) != len(mixture):
                        msg = "{} is not the same length as {}"
                        msg = msg.format(sscfg['feed_material'][i],
                                         sscfg['feed_mixture'][i])
                        logger.log("error", msg)
                    temp_dict = {}
                    # loop through list of feed_material for one depletion step
                    for j in range(len(materials)):
                        # check that all materials used in feed_material are
                        # in material_names list
                        if materials[j] not in material_names:
                            msg = "{} does not have a [[[[material_#]]]] block"
                            msg = msg.format(materials[j])
                            logger.log("error", msg)
                        material_dict = sscfg['material_' + materials[j]]
                        # add isotopes from each material based on feed_mixture
                        for n, v in zip(material_dict["names"],
                                        material_dict["vector"]):
                            if n in temp_dict:
                                temp_dict[n] += \
                                    v / sum(material_dict["vector"]) * \
                                    float(mixture[j])
                            else:
                                temp_dict[n] = \
                                    v / sum(material_dict["vector"]) * \
                                    float(mixture[j])
                    sscfg["feed_vector"].append(temp_dict)
                del sscfg["feed_material"]
                del sscfg["feed_mixture"]
            else:
                # Pass an empty feed dict
                subcfg["feed"] = {}

            # Now we can check the component data
            for subkey in subcfg:
                if subkey in ["feed", "name", "flow_start", "flowrate"]:
                    continue
                elif not subkey.lower().startswith("component"):
                    msg = "Only [[[feed]]] and [[[component_#]]] blocks " + \
                        "are allowed in a [msr][[system]] block!"
                    logger.log("error", msg)

                sscfg = subcfg[subkey]
                sslevel = sublevel + "[[[{}]]]".format(subkey)

                # Now process the component
                check_required(sscfg, ["name"], str, sslevel)
                check_in_set(sscfg, ["type"], ["generic", "in-core", "tank"],
                             sslevel, "generic")
                check_required(sscfg, ["volume"], float, sslevel)
                if sscfg["type"] == "in-core":
                    # then mat_name is required
                    check_required(sscfg, ["mat_name"], str, sslevel)
                    check_optional(sscfg, ["density"], float, sslevel, None)
                else:
                    check_optional(sscfg, ["mat_name"], str, sslevel, None)
                    check_required(sscfg, ["density"], float, sslevel)
                # Get the downstream information
                check_lists(sscfg, ["downstream_components"], str, sslevel,
                            min_num=1, required=True)
                check_lists(sscfg, ["downstream_mass_fractions"], float,
                            sslevel,
                            min_num=len(sscfg["downstream_components"]),
                            max_num=len(sscfg["downstream_components"]),
                            min_val=0., required=True)
                # normalize the mass fractions
                sscfg["downstream_mass_fractions"] = \
                    np.array(sscfg["downstream_mass_fractions"])
                sscfg["downstream_mass_fractions"] /= \
                    np.sum(sscfg["downstream_mass_fractions"])
                sscfg["downstream_mass_fractions"] = \
                    sscfg["downstream_mass_fractions"].tolist()

                # Time for the optional removal info
                check_lists(sscfg, ["removal_names"], str, sslevel,
                            required=False)

                # Note that we allow negative values so that the removal can
                # represent a proportional addition
                sscfg["removal_vector"] = OrderedDict()
                if "removal_names" in sscfg:
                    name_list = sscfg["removal_names"]
                    check_lists(sscfg, ["removal_rates"], float, sslevel,
                                min_num=len(name_list), max_num=len(name_list),
                                required=False)

                    # Now lets make sure the "names" are actual element or
                    # isotope names
                    check_with_function(sscfg, ["removal_names"], is_isotope,
                                        sslevel)

                    # Now rearrange so we have a dictionary relating names to
                    # the rates
                    if sscfg["removal_names"] is not None:
                        for n, v in zip(name_list, sscfg["removal_rates"]):
                            sscfg["removal_vector"][n] = v
                    del sscfg["removal_names"]
                    del sscfg["removal_rates"]

        elif key.lower() == "solve_method":
            # Check those required to be in a set
            keys = ["solve_method"]
            req_vals = constants.MSR_SOLN_METHODS
            default = constants.MSR_DEFAULT_SOLN_METHOD
            check_in_set(config, keys, req_vals, level, default)
        elif key.lower() == "flux_smoothing_method":
            check_in_set(config, ["flux_smoothing_method"],
                         ["histogram", "average"], level, "average")
        else:
            msg = "Input Block {} Will Be Ignored".format(sublevel)
            logger.log("error", msg)

########################################################################


def check_lists(cfg, keys, req_type, level, min_num=None, max_num=None,
                required=True, set_None=False, min_val=None):
    for param in keys:
        if param in cfg:
            # Force it to be a list
            if not isinstance(cfg[param], list):
                cfg[param] = [cfg[param]]

            # Set the type
            for i in range(len(cfg[param])):
                cfg[param][i] = req_type(cfg[param][i])
                if min_val is not None:
                    if cfg[param][i] < min_val:
                        raise ValueError(_min_val_errormsg(param, level,
                                                           min_val))

            # Check the length
            if min_num is not None:
                if len(cfg[param]) < min_num:
                    raise ValueError(_min_len_errormsg(param, level, min_num))
            if max_num is not None:
                if len(cfg[param]) > max_num:
                    raise ValueError(_max_len_errormsg(param, level, max_num))
        else:
            if required:
                raise ValueError(_reqd_errormsg(param, level))
            elif set_None:
                cfg[param] = None


def check_required(cfg, keys, req_type, level):
    for param in keys:
        if param not in cfg:
            raise ValueError(_reqd_errormsg(param, level))
        cfg[param] = req_type(cfg[param])


def check_optional(cfg, keys, req_type, level, default):
    for param in keys:
        if param not in cfg:
            cfg[param] = default
        else:
            cfg[param] = req_type(cfg[param])


def check_in_set(cfg, keys, req_vals, level, default, cast_to=None):
    for param in keys:
        if param not in cfg:
            cfg[param] = default
        else:
            if cast_to is not None:
                cfg[param] = cast_to(cfg[param])
            check_value(param, cfg[param], req_vals)


def check_num_and_range(cfg, keys, req_type, level, default,
                        min_val=None, max_val=None):
    # This must convert the type, check that it is in the range, and set
    # the default
    for param in keys:
        if param not in cfg:
            cfg[param] = default
        else:
            val = req_type(cfg[param])
            if min_val:
                check_greater_than(param, val, min_val, equality=True)
            if max_val:
                check_less_than(param, val, max_val, equality=True)
            cfg[param] = val


def check_with_function(cfg, keys, function, level):
    # This applies a function that returns True or False and raises
    # an error if the function does not return True on a value-by-value
    # basis.
    for param in keys:
        if param in cfg:
            if isinstance(cfg[param], list):
                # Then check all elements of list
                for test in cfg[param]:
                    if not function(test):
                        raise ValueError(_func_errormsg(param, level))
            else:
                if not function(cfg[param]):
                    raise ValueError(_func_errormsg(param, level))


def _reqd_errormsg(name, level):
    return "{} is required at the {} level of the input file!".format(name,
                                                                      level)


def _min_len_errormsg(name, level, min_num):
    msg = "{} at the {} level of the input file is ".format(name, level) + \
        "less than the minimum length of {}".format(min_num)
    return msg


def _max_len_errormsg(name, level, max_num):
    msg = "{} at the {} level of the input file is ".format(name, level) + \
        "less than the maximum length of {}".format(max_num)
    return msg


def _min_val_errormsg(name, level, min_num):
    msg = "{} at the {} level of the input file is ".format(name, level) + \
        "less than the minimum value of {}".format(min_num)
    return msg


def _func_errormsg(name, level):
    return "{} did not pass the check at the".format(name) + \
        " {} level of the input file!".format(level)


def _integral_errormsg(name, level):
    return "{} at the {} level of the input file ".format(name, level) + \
        "must not be an integral value!"


def str_to_bool(s):
    # Convert s to a form without spaces and as all lower case
    test = s.strip().lower()
    if test in ["true", "yes", "on", "1"]:
        return True
    elif test in ["false", "no", "off", "0"]:
        return False
    else:
        raise ValueError("{} cannot be converted to a boolean!".format(s))
