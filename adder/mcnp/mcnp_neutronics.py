from collections import OrderedDict
import shlex
import subprocess
import os
import glob
import copy
import re

import numpy as np

from adder.isotope import update_isotope_depleting_status
from adder.material import Material
from adder.neutronics import Neutronics
from adder.constants import *
from adder.type_checker import *
from adder.utils import get_transform_args

from . import input_methods
from . import input_utils
from . import initialize_data
from . import output_methods
from .cell import Cell
from .constants import MAX_LINE_LEN, TALLY_MAX_ID, MATL_MAX_ID, ROOT_UNIV, \
    NEWLINE_COMMENT, MCNP_OUT_SUFFIX, MCNP_OUT_NAMES, MCNP_FF_SUFFIX, CI_995, \
    CI_95
from .coord_transform import CoordTransform
from .surface import Surface
from .universe import Universe
from .sim_settings import SimSettings

_MCNP_XSDIR_FILENAME = "xsdir"
_NUM_VOL_NPS = int(1e6)


class McnpNeutronics(Neutronics):
    """The class which contains the data and methods needed for
    interfacing with the neutronics solver.

    This class should be extended for each and every neutronics solver
    for which support is added to Adder.

    Parameters
    ----------
    mpi_cmd : str
        The command used to run MPI (blank str if none)
    exec_cmd : str
        The command needed to run the neutronics solver
    base_input_filename : str
        Path and filename of the initial neutronics solver input file
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    use_depletion_library_xs : bool
        Whether or not to use the depletion library's cross sections
    reactivity_threshold : float
        The threshold to apply when determining whether or not to
        incorporate an isotope in the neutronics model
    reactivity_threshold_initial : bool
        Whether or not to apply the reactivity_threshold to the initial
        inventory

    Attributes
    ----------
    solver : str
        A shorthand for the solver interfaced with by this instance
    mpi_cmd : str
        The command used to run MPI (blank str if none)
    exec_cmd : str
        The command needed to run the neutronics solver, including any
        MPI commands
    base_input_filename : str
        Path and filename of the initial neutronics solver input file
    num_threads : int
        Number of shared memory threads to utilize
    num_procs : int
        Number of distributed memory instances to utilize
    use_depletion_library_xs : bool
        Whether or not to use the depletion library's cross sections
    base_input : None or OrderedDict
        The contents of the base input neutronics solver file.
    inputs : Iterable of OrderedDict
        The input files produced at each time; the dimension of the
        iterable is the time index, and the dictionary contains the
        contentsof the neutronics input file.
    neutronics_isotopes : OrderedDict
        Allowed isotopes with neutronics solver identifiers
        (e.g., 92235.70c) as keys, values are the atomic-weight-ratios.
    reactivity_threshold : float
        The threshold to apply when determining whether or not to
        incorporate an isotope in the neutronics model
    reactivity_threshold_initial : bool
        Whether or not to apply the reactivity_threshold to the initial
        inventory
    max_user_tally_id : int
        The maximum tally id of a user tally, to see if there are any
        conflicts
    universes : dict
        The Universe objects keyed by the universe id
    cells : dict
        The parsed cells (Cell objects) keyed by the cell id
    surfaces : dict
        The parsed surfaces (Surface objects) keyed by the surface id
    coord_transforms : dict
        The model's coordinate transforms (CoordTransform objects) keyed
        by the transform's id
    sim_settings : SimSettings
        The simulation settings object
    """

    VALID_SHUFFLE_TYPES = ("material", "universe")
    VALID_TRANSFORM_TYPES = ("universe", "cell", "surface")

    @property
    def solver(self):
        return "mcnp"

    @property
    def max_user_tally_id(self):
        return self._max_user_tally_id

    @max_user_tally_id.setter
    def max_user_tally_id(self, max_user_tally_id):
        check_type("max_user_tally_id", max_user_tally_id, int)
        check_greater_than("max_user_tally_id", max_user_tally_id, 0,
                           equality=True)
        check_less_than("max_user_tally_id", max_user_tally_id, TALLY_MAX_ID,
                        equality=True)
        self._max_user_tally_id = max_user_tally_id

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells):
        if cells is not None:
            check_type("cells", cells, dict)
            for key, value in cells.items():
                check_type("cells key", key, int)
                check_type("cells value", value, Cell)
        self._cells = cells

    @property
    def coord_transforms(self):
        return self._coord_transforms

    @coord_transforms.setter
    def coord_transforms(self, coord_transforms):
        if coord_transforms is not None:
            check_type("coord_transforms", coord_transforms, dict)
            for key, value in coord_transforms.items():
                check_type("coord_transforms key", key, int)
                check_type("coord_transforms value", value, CoordTransform)
        self._coord_transforms = coord_transforms

    def universe_by_name(self, name):
        for u in self.universes.values():
            if u.name == name:
                return u
        # If we get here, we didnt find it
        self.log("error", "Universe {} not found".format(name))

    def init_model(self, materials, universe_info):
        """This method builds our universes, assigns names, and assigns
        the cells and materials within it. This then initializes the
        statuses of these objects.

        Parameters
        ----------
        materials : Iterable of Material
            A constructed Material object for each material in the
            model; since no is_depleting (or isotope is_depleting)
            information is available at this stage, this must be
            overwritten upstream.
        universe_info : OrderedDict
            The key is the universe id and the value is the
            user-specified name

        """
        # For a shortcut later on, create a dictionary of materials by id
        mat_by_id = {}
        for mat in materials:
            mat_by_id[mat.id] = mat

        # We have materials and cells defined, but we need to assign
        # the materials to the cells. After that we create universes,
        # assign cells to the universes, and then assign universes to
        # fill of the cells (note that cells dont contain
        # their parent universes, only parent universe ids to break a
        # cyclic reference)

        # We can interweave these operations, but for clarity we will
        # keep separate as much as possible
        # First assign materials to the cells
        for cell in self.cells.values():
            if cell.material_id != 0:
                mat = mat_by_id[cell.material_id]
                if cell.volume is not None:
                    # Then the model contains our volume, assign it to
                    # the material
                    mat.volume = cell.volume
                cell.material = mat

        # Now create universes, including assigning names
        self.universes = OrderedDict()
        for cell in self.cells.values():
            u_id = cell.universe_id
            if u_id not in self.universes:
                if u_id in universe_info:
                    name = universe_info[u_id]["name"]
                else:
                    name = str(u_id)

                self.universes[u_id] = Universe(name, u_id)

        # Now add the cells to the universes
        for cell in self.cells.values():
            u_id = cell.universe_id
            self.universes[u_id].add_cells([cell])

        # Finally, add universes as fills of the cell
        for cell in self.cells.values():
            cell.assign_fill(self.universes)

        # Pre-cache the member status of the root universe
        self.universes[ROOT_UNIV].update_nested_children()

        # Now lets go through and set the statuses so that if
        # the univ/mat is not in the root universe, then it has a
        # status of supply
        root_univ_ids = [ROOT_UNIV] + \
            self.universes[ROOT_UNIV].nested_universe_ids
        for univ_id, univ in self.universes.items():
            if univ_id in root_univ_ids:
                univ.status = IN_CORE
            else:
                univ.status = SUPPLY

        self._rxn_rate_tally_map = None
        self._isos_and_rxn_keys = None   # This would need to be set back to 1
        # if somehow the isotopes in the depletion library change after
        # the first MCNP input is written w/ rxn rate tallies

    def read_input(self, library_file, num_neutron_groups, user_mats_info,
                   user_univ_info, shuffled_mats, shuffled_univs, depl_libs):
        """Return parsed information about an MCNP input file, including
        all the information needed to initialize adder.Material objects.

        This method parses the given file to identify separate input
        blocks, and gather pertinent cell and material information. All
        comments will be removed and line continuations also removed.

        Note the input file format is: an ordered dictionary of card
        blocks (Lists of str) where the keys are: "message", "title",
        "cell", "surface", "material", "tally", "output", and "other".

        Parameters
        ----------
        library_file : str
            The filename and path to the xsdir file
        num_neutron_groups : int
            The number of energy groups
        user_mats_info : OrderedDict
            The keys are the ids in the neutronics solver and
            the value is an OrderedDict of the name, depleting boolean,
            ex_core_status, non_depleting_isotopes list, and
            use_default_depletion_library flag.
        user_univ_info : OrderedDict
            The keys are the universe ids in the neutronics solver and
            the value is an OrderedDict of the name.
        shuffled_mats : set
            The set of material names that are shuffled
        shuffled_univs : set
            The set of universe names that are shuffled
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        Returns
        -------
        materials : Iterable of adder.Material
            A Material object for each material in the model;
            since no is_depleting (or isotope is_depleting) information is
            available at this stage, this must be overwritten upstream.
        """

        filename = self.base_input_filename

        # Get the information from the input file
        messages, title, cell_block, surface_cards, data = \
            input_methods.get_blocks(filename)
        parsed_transforms, data = input_methods.parse_transforms(data)
        parsed_surfaces = input_methods.parse_surfaces(surface_cards,
                                                       parsed_transforms)
        parsed_cells = input_methods.parse_cells(cell_block, parsed_transforms)
        m_data, m_default_nlib, mt_data, material_cards, other_cards = \
            input_methods.parse_materials(data)
        tally_data, other_cards, max_tally_id, multiplier_mat_ids = \
            input_methods.get_tallies(other_cards)
        output_data, other_cards = input_methods.get_output(other_cards)
        user_sim_settings, other_cards = \
            input_methods.get_sim_settings(other_cards)

        # Handle xsdir information
        # See if there is an xsdir in the messages block
        message_xsdir = None
        for msg_line in messages:
            if "xsdir=" in msg_line:
                message_xsdir = msg_line.split("xsdir=")[1].split()[0]

        if library_file is None:
            # Either use message xsdir or the environment default
            if message_xsdir is not None:
                self.xsdir_file = message_xsdir
            else:
                self.xsdir_file = os.environ["DATAPATH"] + "/" + \
                    _MCNP_XSDIR_FILENAME
        else:
            if message_xsdir is not None:
                # Then we need to tell the user if the message xsdir
                # conflicts
                msg = "The MCNP input's MESSAGE block contains an xsdir " + \
                    "file; this will be replaced with the " + \
                    "'neutronics_library_file' provided in the ADDER input."
                self.log("INFO_FILE", msg)
            self.xsdir_file = library_file

        # Now put the xsdir message in the message block
        if messages:
            # Then it has data, find and place
            found_it = False
            for i in range(len(messages)):
                msg_line = messages[i]
                if "xsdir=" in msg_line:
                    xloc = msg_line.find("xsdir=") + len("xsdir=")
                    sloc = msg_line.find(" ", xloc)
                    messages[i] = msg_line[:xloc] + self.xsdir_file
                    if sloc != -1:
                        # Then there is a space, so add on the remainder
                        messages[i] += msg_line[sloc:]
                    found_it = True
                    break
            if not found_it:
                # Then add to the end of the messages
                messages[-1] += " xsdir={}".format(self.xsdir_file)
        else:
            # Does not have anything, add
            messages.append("message: xsdir={}".format(self.xsdir_file))

        # Get the information from our xsdir file
        xs_updates, awtab_updates = input_methods.get_xs_awtab(other_cards)
        self.neutronics_isotopes = \
            initialize_data.parse_xsdir(self.xsdir_file, xs_updates,
                                        awtab_updates)

        # Put all of our input data into an input dictionary
        input_cards = OrderedDict()
        input_cards["message"] = messages
        input_cards["title"] = title
        input_cards["material"] = material_cards
        input_cards["tally"] = tally_data
        input_cards["output"] = output_data
        input_cards["other"] = other_cards
        self.base_input = input_cards
        self.max_user_tally_id = max_tally_id
        self.multiplier_mat_ids = multiplier_mat_ids
        self.cells = parsed_cells
        self.surfaces = parsed_surfaces
        self.coord_transforms = parsed_transforms
        self.sim_settings = user_sim_settings

        # Pre-determine which materials are owned by which cell(s) and univs
        # so we do not need to do this for each material. Additionally, generate
        # a list of cells attached to complement operators.
        cell_ids_by_mat = {}
        univ_ids_by_mat = {}
        complement_operator_cells = []
        for cell in self.cells.values():

            # MCNP defines the complement of another cells as #n where n is any
            # cell number.
            for complement_operator in re.findall('#[0-9]+', cell.surfaces):
                complement_operator_cells.append(int(complement_operator[1:]))

            if cell.material_id in cell_ids_by_mat:
                cell_ids_by_mat[cell.material_id].append(cell.id)
                univ_ids_by_mat[cell.material_id].append(cell.universe_id)
            else:
                cell_ids_by_mat[cell.material_id] = [cell.id]
                univ_ids_by_mat[cell.material_id] = [cell.universe_id]

        # Initialize the material information
        # In doing so, we have to read the cell data to figure out:
        # 1) if the material is in-core or ex-core to set status
        # 2) take the density from the cell and attach to material
        materials = []
        for m, data in m_data.items():
            mat_name = str(m)
            mat_id = m

            # Find the cell(s) that owns this material
            if mat_id in cell_ids_by_mat:
                cell_ids = cell_ids_by_mat[mat_id]
            else:
                cell_ids = []
            cell_densities = [self.cells[c_id].density for c_id in cell_ids]

            if len(cell_ids) > 0:
                # We will initialize just the first one
                density = cell_densities[0]
                isotopic_data, new_cell_density = \
                    input_methods.convert_density(data, density)

                # But update all other cell densities after this conversion
                for cell_id in cell_ids:
                    self.cells[cell_id].density *= new_cell_density / density

                # Set to in-core, and we will have to let universe
                # initialization assign correctly
                status = IN_CORE
            else:
                # Then this cell is not owned and so is not present
                # in the model
                # Set the status as supply
                status = SUPPLY
                # Then, set the density to 1.0 since MCNP has no
                # knowledge of the density of materials not assigned
                # to a cell
                isotopic_data, new_cell_density = \
                    input_methods.convert_density(data, 1.0)

            # Build our material object input
            # While doing this we will verify that the isotope names are not
            # duplicates and we will raise an error if they are
            isotope_args = []
            iso_fractions = []
            iso_names = set()
            for i in range(len(isotopic_data)):
                iso_name, fraction, nlib = isotopic_data[i]
                if iso_name in iso_names:
                    msg = f"Material id: {mat_id} " \
                        f"contains {iso_name} multiple times. " \
                        "Only one entry is supported."
                    self.log("ERROR", msg)
                else:
                    iso_names.add(iso_name)
                isotope_args.append((iso_name, nlib, True))
                iso_fractions.append(fraction)
            if m in mt_data:
                thermal_libs = mt_data[m]
            else:
                thermal_libs = []

            if len(cell_ids) > 0:
                density = self.cells[cell_ids[0]].density
                volume = self.cells[cell_ids[0]].volume
            else:
                density = 1.
                volume = None

            depleting = True
            # Get the user-set attributes which can be used with the
            # Material.__init__ function
            if mat_id in user_mats_info:
                mat_name = user_mats_info[mat_id]["name"]
                depleting = user_mats_info[mat_id]["depleting"]
                if user_mats_info[mat_id]["density"] is not None:
                    density = user_mats_info[mat_id]["density"]

            # The material name for now is the same as the id
            material = Material(mat_name, mat_id, density, isotope_args,
                                iso_fractions, depleting, m_default_nlib[m],
                                num_neutron_groups, thermal_libs, status)
            material.volume = volume

            # Raise an error if any log message was found
            self.update_logs(material.logs)
            material.clear_logs()

            # Since this is our initial input parsing, raise an error if the
            # material has isotopes that are not in the neutronics library
            for i in range(material.num_isotopes):
                iso = material.isotopes[i]
                zaid = input_methods._zam_to_mcnp(
                    iso.Z, iso.A, iso.M, iso.xs_library)
                if zaid not in self.neutronics_isotopes:
                    msg = f"Material {material.name} (id: {material.id}) " \
                        f"contains {zaid} which is not present in the xsdir " \
                        "file"
                    self.log("ERROR", msg)

            # If we are using the depletion library xs, then just set
            # the material to represent that now
            if self.use_depletion_library_xs:
                material.is_default_depletion_library = True

            # Now take the user-set attributes and set the material obj
            if mat_id in user_mats_info:
                # If we are not using the default depletion lib, then
                # allow the users option to pass
                if not self.use_depletion_library_xs:
                    key = "use_default_depletion_library"
                    material.is_default_depletion_library = \
                        user_mats_info[mat_id][key]

                key = "volume"
                if key in user_mats_info[mat_id]:
                    if user_mats_info[mat_id][key] is not None:
                        material.volume = user_mats_info[mat_id][key]

                key = "non_depleting_isotopes"
                if len(user_mats_info[mat_id][key]) > 0:
                    # Then we have specific values
                    for i in range(material.num_isotopes):
                        orig_iso = material.isotopes[i]
                        if orig_iso.name in user_mats_info[mat_id][key]:
                            new_iso = update_isotope_depleting_status(orig_iso,
                                                                      False)
                            material.isotopes[i] = new_iso

                # Now apply the reactivity_threshold_initial information
                key = "apply_reactivity_threshold_to_initial_inventory"
                if key in user_mats_info[mat_id] and \
                    user_mats_info[mat_id][key] is not None:
                    mat_reactivity_threshold_initial = \
                        user_mats_info[mat_id][key]
                else:
                    mat_reactivity_threshold_initial = \
                        self.reactivity_threshold_initial
            else:
                mat_reactivity_threshold_initial = \
                        self.reactivity_threshold_initial

            material.establish_initial_isotopes(
                mat_reactivity_threshold_initial)

            materials.append(material)

        # Before we duplicate all the materials, we must assign the
        # isotope_is_depleting status and make sure we have updated the base
        # library correspondingly
        for mat in materials:
            if mat.is_depleting:
                mat.update_isotope_is_depleting(depl_libs[BASE_LIB])
                self.update_logs(mat.logs)
                mat.clear_logs()

        # Now create the duplicate materials for the materials present
        # in multiple locations (if depleted or shuffled)
        self.log("info", "Cloning Depleting, Shuffled, or "
                    "Cross-Universe Materials", 4)
        Nmat = len(materials)
        for m in range(Nmat):
            mat = materials[m]
            mat_id = mat.id
            if mat_id in cell_ids_by_mat and len(cell_ids_by_mat[mat_id]) > 1:
                if mat.is_depleting or mat.name in shuffled_mats or \
                    len(univ_ids_by_mat[mat_id]) > 1:

                    for c in cell_ids_by_mat[mat_id][1:]:
                        # Then we have multiple instances of a shuffled and/or
                        # depleting material, we should make a copy to allow
                        # for the unique compositions as the depletion develops
                        new_mat = mat.clone(depl_libs,
                                            mat.name + "_reg_{}".format(c))
                        msg = "Material {} cloned as material {}".format(
                            mat.name, new_mat.name)
                        self.log("info_file", msg)
                        new_mat.density = self.cells[c].density
                        if self.cells[c].volume:
                            new_mat.volume = self.cells[c].volume
                        self.cells[c].material_id = new_mat.id
                        materials.append(new_mat)

        # Now reassign integer IDs in cells and universes to actual objects.
        self.init_model(materials, user_univ_info)

        # Print warning to user if there is a cell definition that uses the
        # complement operator in a shuffled universe.
        for univ in user_univ_info:
            if self.universes[univ].name in shuffled_univs:

                self.universes[univ].update_nested_children()
                univ_cells = self.universes[univ].nested_cells.keys()

                warning_cell_list = []
                for cell in complement_operator_cells:
                    if cell in univ_cells:
                        warning_cell_list.append(cell)

                if warning_cell_list:
                    msg = "ADDER does not modify complement operators on " \
                          "cells during shuffling. Cells {} are members of a " \
                          "shuffled universe (universe {}) and are attached " \
                          "to a complement operator in the definition of " \
                          "another cell. The user must confirm that any " \
                          "shuffling of this universe still results in a " \
                          "correct region definition with the original " \
                          "complement.".format(warning_cell_list, univ)
                    self.log("warning", msg)

        self.complement_operator_cells = complement_operator_cells

        return materials

    def _write_continue(self, filename):
        """Creates a continue-run input for the particular run case

        Parameters
        ----------
        filename : str
            The file to write to, without the suffix
        """

        # Get the current version of the input
        if len(self.inputs) == 0:
            input_cards = self.base_input
        else:
            input_cards = self.inputs[-1]

        # Add the message card
        messages = input_cards["message"]
        file_string = ""
        if messages:
            for message in messages:
                file_string += input_utils.card_format(message)
            file_string += "\nCONTINUE\n"

        # Now just add in the modified kcode card
        file_string += input_utils.card_format(self.sim_settings.kcode_str)
        file_string += "\n"

        with open(filename, "w") as file:
            file.write(file_string)

    def write_input(self, filename, label, materials, depl_libs, store_input,
                    deplete_tallies, user_tallies, user_output,
                    update_iso_status=True):
        """Updates the input for the particular run case

        Parameters
        ----------
        filename : str
            The file to write to, without the suffix
        label : str
            Label to append to the title
        materials: Iterable of adder.Materials
            New material information to include
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        store_input : bool
            Whether or not to store the input file in :attrib:`inputs`
        deplete_tallies : bool
            Whether or not to write the tallies needed for depletion
        user_tallies : bool
            Whether or not to write the user tallies
        user_output : bool
            Whether or not to write the user's output control cards
        update_iso_status : bool, optional
            Whether or not to update the status of the isotopes_in_neutronics
            parameter of each material; this is not needed for iterative
            analyses such as critical searches and therefore need not be
            performed again. Defaults to updating the status
        """
        # Get the current version of the input
        if len(self.inputs) == 0:
            input_cards = self.base_input
        else:
            input_cards = self.inputs[-1]

        messages = input_cards["message"]
        title = self.base_input["title"] + " " + label
        if len(title) > MAX_LINE_LEN:
            title = self.base_input["title"][0:MAX_LINE_LEN - (len(label) + 6)]\
                    + "[...] " + label

        model_mats = self.universes[ROOT_UNIV].nested_materials
        if update_iso_status:
            self.log("info_file",
                    "Evaluating Which Isotopes to Include in MCNP Model", 10)
            for mat_id, mat in model_mats.items():
                mat = model_mats[mat_id]
                # Update the isotopes in neutronics status
                if mat.is_depleting:
                    # For depleting materials, we expect a build-up of trace
                    # isotopes hence we should filter these out.
                    # Note we only do this for depleting isotopes as we assume
                    # that the user chose their constituents of non-depleting
                    # isotopes carefully.
                    lib = depl_libs[mat.depl_lib_name]
                    mat.determine_important_isotopes(lib,
                        self.reactivity_threshold)
                    # Next only allow isotopes that we have MCNP libs for
                    for i in range(mat.num_isotopes):
                        iso = mat.isotopes[i]
                        if mat.isotopes_in_neutronics[i]:
                            zaid = input_methods._zam_to_mcnp(
                                iso.Z, iso.A, iso.M, iso.xs_library)
                            if zaid not in self.neutronics_isotopes:
                                mat.isotopes_in_neutronics[i] = False

        self.log("info_file", "Preparing Tally Cards", 10)
        tallies = []
        # Add in the tallies that ADDER wants
        if deplete_tallies:
            # We are going to need the flux for the depletion, and
            # if needed, for normalizing the reaction rates, so lets get
            # that.
            tallies.extend(
                input_methods.create_material_flux_tallies(self.universes,
                    depl_libs))

            if not self.use_depletion_library_xs:
                # Now, if we are updating the depletion library, we
                # also need to populate the tallies for the reaction
                # rates of interest. Determine what these are, and add
                # their tally cards.
                rr_tallies, self._rxn_rate_tally_map, \
                    self._isos_and_rxn_keys = \
                        input_methods.create_rxn_rate_tallies(
                            self.universes, self.neutronics_isotopes,
                            self._isos_and_rxn_keys, depl_libs)
                # Store the tallies, and the unique cell sets for usage when
                # getting results
                tallies.extend(rr_tallies)

        # And put in the ones the user requested
        # (these have already been verified to not have conflicting IDs with
        #  the ADDER IDs)
        if user_tallies:
            # We have to use the tally card from the base input to get
            # the user tallies, and not what ADDER has done to it since
            tallies.extend(self.base_input["tally"])

        # Set the output cards while incorporating user output when needed
        if user_output:
            # We have to use the output card from the base input to get
            # the user output, and not what ADDER has done to it since
            output = input_methods.create_output(self.base_input["output"])
        else:
            output = input_methods.create_output()

        others = input_cards["other"]

        # Get the materials in card form
        if user_tallies:
            multiplier_mat_ids = self.multiplier_mat_ids
        else:
            multiplier_mat_ids = set()

        self.log("info_file", "Preparing Material Cards", 10)
        mat_cards = \
            input_methods.update_materials(self.universes,
                                           self.neutronics_isotopes,
                                           materials, multiplier_mat_ids,
                                           self.reactivity_threshold)

        # Now that we know the number of isotopes and the number of
        # material cards we can identify possible ID collisions
        if not self.use_depletion_library_xs and deplete_tallies:
            # The tallies array contains lines for each new pseudo-mat
            # but also one for the flux tally, so take that off
            num_mat_for_rxn_tallies = len(tallies) - 1
            if user_tallies:
                # Then we also should take this off.
                num_mat_for_rxn_tallies -= len(self.base_input["tally"])

            if len(mat_cards) + num_mat_for_rxn_tallies > MATL_MAX_ID:
                msg = "The number of materials in the model and the " + \
                    "maximum number of pseudo-materials for reaction rate " + \
                    "tallies exceeds the MCNP Material Limit of " + \
                    "{};\n\t".format(MATL_MAX_ID) + \
                    "Either reduce materials or set the " + \
                    "`use_depletion_library_xs` to True in the input."
                self.log("error", msg)

            # Also, now that we know the number of tallies, we can identify
            # possible tally ID collisions
            if user_tallies:
                min_rr_t_id = min(self._rxn_rate_tally_map.keys()) + 10
                if self.max_user_tally_id >= min_rr_t_id:
                    msg = "The user's tally IDs in the model conflict with the " + \
                        "IDs used for reaction rate tallies. Please modify user " + \
                        "tallies so the tally ID is less than " + \
                        "{}".format(min_rr_t_id)
                    self.log("error", msg)

        # Now we can write the input
        self.log("info_file", "Writing MCNP Input to Disk", 10)
        input_methods.create_input_file(filename, messages, title,
                                        self.universes, self.surfaces,
                                        mat_cards, self.coord_transforms,
                                        tallies, output, self.sim_settings,
                                        others)

        new_input = OrderedDict()
        new_input["message"] = messages
        new_input["title"] = title
        new_input["material"] = mat_cards
        new_input["tally"] = tallies
        new_input["output"] = output
        new_input["other"] = others

        if store_input:
            # TODO: Why do I do this?
            self.inputs.append(new_input)

    def _exec_solver(self, inp_name, out_name, print_log_msg=True,
                     keep_runtpe=False, is_continue=False, fast_forward=False):
        """Performs the computation.

        Parameters
        ----------
        inp_name : str
            The input file name
        out_name : str
            The output file name without the appendix
        print_log_msg : bool, optional
            Whether or not the status of this run should be printed on
            the log; defaults to printing the status
        keep_runtpe : bool, optional
            Whether or not to keep the RUNTPE file. This is primarily only
            used if a continue-run is expected next
        is_continue : bool, optional
            Whether or not this run itself is a continue-run. If so, the
            run-tape file is assumed to be named "runtpe".
        fast_forward : bool
            Whether or not to use existing output files of the correct names
            (True) or re-calculate.
        """

        if self.num_procs > 1 and self.mpi_cmd:
            run_cmd = "{} -np {} {}".format(self.mpi_cmd, self.num_procs,
                                            self.exec_cmd)
            run_cmd += " INP={} NAME={} MCTAL={}m".format(
                inp_name, out_name, out_name)
        else:
            run_cmd = "{} INP={} NAME={} MCTAL={}m".format(
                self.exec_cmd, inp_name, out_name, out_name)

        if is_continue:
            run_cmd = run_cmd + " C RUNTPE=runtpe"

        if self.num_threads > 1:
            run_cmd = run_cmd + " TASKS {}".format(self.num_threads)

        # Look to see if this run can be "fast_forwarded" by looking
        # for the MCNP outputs
        if fast_forward and all([os.path.exists(out_name + out_type)
                                 for out_type in MCNP_FF_SUFFIX]):
            # Then we just return here as our job is done
            self.log("info", "Skipping MCNP Calc on {}".format(out_name), 10)
            return

        # Clear the current directory of output files in case of an errant
        # execution in an earlier run
        for out_type in MCNP_OUT_SUFFIX:
            try:
                os.remove(out_name + out_type)
            except FileNotFoundError:
                pass
        for outfile in MCNP_OUT_NAMES:
            if outfile == "runtpe" and is_continue:
                # Keep the runtpe file by not deleting this
                pass
            else:
                try:
                    os.remove(outfile)
                except FileNotFoundError:
                    pass

        if print_log_msg:
            if is_continue:
                self.log("info", "Continuing MCNP on {}".format(out_name), 10)
            else:
                self.log("info", "Executing MCNP on {}".format(out_name), 10)
        status = subprocess.run(shlex.split(run_cmd), check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, shell=False)

        # Look for fatal error in the the output stream (probably the
        # fastest way to capture an error)
        error_index = status.stdout.find("fatal error.")
        if error_index != -1:
            msg = "MCNP Encountered an Error when running {}".format(inp_name)
            self.log("error", msg)

        # The runtpe is huge, delete it if we can
        if not keep_runtpe:
            try:
                os.remove(out_name + "r")
            except FileNotFoundError:
                pass
            try:
                os.remove(out_name + "runtpe")
            except FileNotFoundError:
                pass

        # Now lets move the other output files to out_name + _ + names
        for outfile in MCNP_OUT_NAMES:
            if os.path.exists(outfile):
                os.rename(outfile, out_name + "_" + outfile)

    def _read_results(self, filename, materials, depl_libs, deplete_tallies):
        """Gets the flux and volume of each material after the current
        neutronics solve. The material object is *not* updated; that is
        saved for the main adder code to handle since it is solver agnostic.

        Parameters
        ----------
        filename : str
            The filename that was used when executing MCNP
        materials : Iterable of adder.Material
            The materials, providing the IDs of their constituent cells
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        deplete_tallies : bool
            Whether or not depletion tallies are included in the model

        Returns
        -------
        keff : float
            Calculated k-eigenvalue of the problem.
        keff_uncertainty : float
            Estimated k-eigenvalue uncertainty
        flux : OrderedDict of np.ndarray
            Dictionary where the key is the material id and the value is the
            group-wise tallied flux mean.
        nu : float
            average number of neutrons produced per fission.
        volume : OrderedDict
            The material id is the key and the value is volume in cm^3
        """

        self.log("info_file", "Parsing MCNP Output", 10)

        # Initialize the MCTAL and get keff
        mctal, keff, keff_stddev = output_methods._init_mctal(filename + "m")

        # Parse the tables we need from the standard output file
        volumes_by_cell = None
        nu = 1.
        with open(filename + "o", "r") as file:
            line = file.readline()
            while line:
                table_num = output_methods._get_table_num(line)
                if table_num == 60:
                    volumes_by_cell = output_methods._get_table60_volumes(file)
                elif table_num == 130:
                    nu = output_methods._get_table130_nu(file, keff)
                # Advance to the next line
                line = file.readline()

        msg = "MCNP k_eff={:.6f}+/-{:.6f}".format(keff, keff_stddev)
        self.log("info", msg, 10)

        model_cells = self.universes[ROOT_UNIV].nested_cells
        if not deplete_tallies:
            # Then we just need to initialize our flux storage
            flux = OrderedDict()
        else:
            # Get the fluxes on a per-cell basis
            self.log("info_file", "Obtaining Fluxes from MCTAL", 10)
            flux = output_methods._get_material_wise_flux(mctal,
                                                          self.universes)

            if not self.use_depletion_library_xs:
                # Then this list is not empty, and thus there are rxn rates
                self.log("info_file", "Obtaining Rxn Rates from MCTAL", 10)
                output_methods._get_material_wise_rxn_rates(mctal, model_cells,
                    flux, self._rxn_rate_tally_map, depl_libs)
        # Clear the storage used by _rxn_rate_tally_map as it may be large and
        # we dont need that info anymore
        self._rxn_rate_tally_map = None

        # Convert volumes from being by cell to by material id
        volumes = OrderedDict()
        self.log("info_file", "Assigning Volumes", 10)
        if volumes_by_cell is not None:
            for cell in model_cells.values():
                if cell.material is not None:
                    mat_id = cell.material_id
                    volumes[mat_id] = volumes_by_cell[cell.id]

        self.log("info_file", "Completed MCNP Post-Processing", 10)

        return keff, keff_stddev, flux, nu, volumes

    def calc_volumes(self, materials, vol_data, target_unc, max_hist,
                     only_depleting):
        """Computes the volumes of the materials of interest, storing
        them on the Material objects

        Parameters
        ----------
        materials : Iterable of Material
            The problem materials, providing the IDs of their
            constituent cells
        vol_data : dict
            Parameters for the sampling volume
        target_unc : float
            The target uncertainty, in percent, for which the stochastic
            volume estimation will meet on every region's volume.
        max_hist : int
            The maximum number of histories to run in the stochastic
            volume estimation.
        only_depleting : bool
            Whether or not to iterate on volumes for only depleting
            materials; note this can significantly increase the runtime
            if set to False.

        Returns
        -------
        volume : OrderedDict
            The material id is the key and the value is volume in cm^3
        """

        # To do this we will use MCNP to mimic the stochastic volume
        # approach done in serpent and openmc (i.e., sample src points
        # uniformly in space, and then the volume of each cell is
        # related to the probability a src is born in that cell)
        # We will use MCNP's analytically determined volumes
        # preferentially over the stochastic volumes

        self.log("info", "Performing MCNP Stochastic Volume Computation", 6)

        # First determine which cells we should even care about the
        # volume of
        model_cells = self.universes[ROOT_UNIV].nested_cells
        if only_depleting:
            cells_to_obtain = []
            for c in model_cells.values():
                if c.material_id and c.material is not None:
                    if c.material.is_depleting:
                        cells_to_obtain.append(c.id)
        else:
            cells_to_obtain = [c.id for c in model_cells.values()
                               if c.material_id != 0]

        # Compute the sampling volume
        if vol_data["type"] == "box":
            deltas = np.subtract(vol_data["upper_right"],
                                 vol_data["lower_left"])
            V_tot = deltas[0] * deltas[1] * deltas[2]
        elif vol_data["type"] == "cylinder":
            V_tot = np.pi * vol_data["height"] * vol_data["radius"]**2

        # Get the current version of the input
        if len(self.inputs) == 0:
            input_cards = self.base_input
        else:
            input_cards = self.inputs[-1]

        messages = input_cards["message"][:]
        title = self.base_input["title"] + " VCalc"
        # Incorporate our outputs with only the bare minimum info
        output = ["prdmp 0 0 0 0", "print -60 -126"]
        # Get the other cards
        others = input_cards["other"][:]

        # Get the material cards
        multiplier_mat_ids = set()
        mat_cards = \
            input_methods.update_materials(self.universes,
                                           self.neutronics_isotopes,
                                           materials, multiplier_mat_ids,
                                           self.reactivity_threshold)

        # Build the sdef
        if vol_data["type"] == "box":
            sdef_cards = ["sdef X=d1 Y=d2 Z=d3"]
            si_str = "si{} {} {}"
            sp_str = "sp{} 0. 1."
            right = vol_data["upper_right"]
            left = vol_data["lower_left"]
            for i, (lo, hi) in enumerate(zip(left, right)):
                sdef_cards.append(si_str.format(i + 1, lo, hi))
                sdef_cards.append(sp_str.format(i + 1))
        elif vol_data["type"] == "cylinder":
            card = "sdef pos={} {} {} rad=d1 axs=0 0 1 ext=d2"
            bottom = vol_data["bottom"]
            card = card.format(bottom[0], bottom[1], bottom[2])
            sdef_cards = [card]
            ro = vol_data["radius"]
            sdef_cards.append("si1 H 0. {}".format(ro))
            sdef_cards.append("sp1 -21 1")
            sdef_cards.append("si2 H 0 {}".format(vol_data["height"]))
            sdef_cards.append("sp2 D 0 1")
        others.extend(sdef_cards)

        # Add in the VOID command
        if "void" not in others:
            others.append("void")

        # Remove cut, lost, and nps cards that might be already present in the
        # MCNP input
        for iother, other in reversed(list(enumerate(others))):
            if any([other.lower().startswith(x) for x in
                    ["cut", "lost", "nps"]]):
                others.pop(iother)

        # And add in a particle time cutoff that is exceedingly low so
        # we dont get any collisions to save compute time
        others.append("cut:n 2.E-14")

        # Make the run resilient to lost particles
        others.append("lost {} 1".format(_NUM_VOL_NPS))

        # Add in the number of histories
        others.append("nps {}".format(_NUM_VOL_NPS))

        # Create the seeds for our PRNG cards
        np.random.seed(17)
        seeds = np.arange(1, 10000001, 2, dtype=np.int64)
        np.random.shuffle(seeds)

        self.log("info", "Beginning Iteration 1", 8)

        # Perform first pass
        scores, tot_hits, volumes_by_cell = \
            _run_volume(messages, title, self, mat_cards, output,
                        others, seeds[0], True, cells_to_obtain)
        seed_idx = 1

        # Now we need to combine the results
        running_scores, running_hits, running_vols, running_errs = \
            _convert_cell_hits_to_volume(scores, tot_hits, volumes_by_cell,
                                         cells_to_obtain, V_tot, {}, 0,
                                         target_unc)

        # Edit the scoring efficiency to the log
        efficiency = np.sum(list(running_scores.values())) / running_hits
        msg = "Volume Sampling Efficiency is {:.1f}%"
        self.log("info", msg.format(100. * efficiency), indent=10)

        converged = False
        num_iterations = 1
        while not converged:
            # Test for convergence
            max_err = np.max([v for v in running_errs.values()])
            total_histories = _NUM_VOL_NPS * num_iterations
            converged = total_histories >= max_hist or max_err < target_unc

            # Only do the rest if we are not yet converged, setting us
            # up for the next loop
            if not converged:
                num_iterations += 1
                msg = "Iteration {}: Max. Err is {:.2f}%"
                self.log("info", msg.format(num_iterations, max_err), 8)
                # If we have not converged, run the next computation
                scores, tot_hits, _ = \
                    _run_volume(messages, title, self, mat_cards,
                                output, others, seeds[seed_idx], False,
                                cells_to_obtain)

                seed_idx += 1
                # Now we need to combine the results
                running_scores, running_hits, running_vols, running_errs = \
                    _convert_cell_hits_to_volume(scores, tot_hits,
                                                 volumes_by_cell,
                                                 cells_to_obtain, V_tot,
                                                 running_scores, running_hits,
                                                 target_unc)

        # Write the log message
        msg = "Completed; {:,} Histories Simulated".format(total_histories)
        msg += " for Max. Uncertainty of {:.4f}%".format(max_err)
        self.log("info", msg, 8)

        # Convert volumes from being by cell to by material id
        volumes = OrderedDict()
        for cell in model_cells.values():
            if cell.material is not None:
                if cell.material.is_depleting:
                    mat_id = cell.material_id
                    volumes[mat_id] = running_vols[cell.id]

        return volumes

    def shuffle(self, start_name, to_names, shuffle_type, materials, depl_libs):
        """Shuffles the object named start_name to the locations listed
        in to_names. This will include handling the cases where the
        start_name object is in storage, supply, or in-core.

        Parameters
        ----------
        start_name : str
            The name of the item (type shuffle_type) to move
        to_names : Iterable of str
            The names of the objects (of type shuffle_type) to be
            displaced. The 0th entry is the location for start_name to
            be moved to. The 1st entry is for the item displaced by that
            first move, so on and so forth. The last item displaced will
            head to storage.
        shuffle_type : str
            The type of shuffle to perform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        materials : List of Material
            The materials to work with
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        """

        # Use the base class' type checking
        super().shuffle(start_name, to_names, shuffle_type, materials,
                        depl_libs)

        if shuffle_type == "material":
            self._move_materials(start_name, to_names, materials, depl_libs)
        elif shuffle_type == "universe":
            self._move_universes(start_name, to_names, materials, depl_libs)
        else:
            raise NotImplementedError

        # Update the cache the nested members of the root universe
        self.universes[ROOT_UNIV].update_nested_children()

    def _move_materials(self, mat_name, to_mats, materials, depl_libs):
        """Moves a material in to the model from storage or supply as
        well as the materials displaced by that move.

        If the starting material is in the core its' location is simply
        moved to the cell. If the starting material is in storage, then
        it gets assigned to the cell. Finally, if the material is a
        supply material (e.g., fresh fuel from the factory for later
        batches), then it will be cloned (leaving the original material
        as-is), given a new ID and its name will be updated per the
        rules in the documentation for Material.clone().

        Parameters
        ----------
        mat_name : str
            Material to move
        to_mats : Iterable of str
            Names of Materials to be displaced; the 0th entry is the
            location for mat_name to move to, the 1st for the material
            displaced by the 0th move to go to, and so on and so forth.
            The last material displaced will go to storage.
        materials : List of adder.Material
            The materials to work with.
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        """

        # For convenience and speed, create easy lookups into the
        # materials by name; we have to do this with every move as the
        # move can create new materials (with clones of storage) so we
        # dont pre-determine this
        mat_by_name = {}
        for i, mat in enumerate(materials):
            mat_by_name[mat.name] = mat

        # Now if we are not starting in-core, and we have a supply mat'l
        # as our start, then we need to add a copy of it
        # (and also update our quick access dictionaries with it)
        if mat_by_name[mat_name].status == SUPPLY:
            orig_mat = mat_by_name[mat_name]
            start_mat = orig_mat.clone(depl_libs)
            msg = "Material {} cloned as material {}".format(orig_mat.name,
                                                             start_mat.name)
            self.log("info_file", msg)
            start_mat.status = STORAGE
            # Also save this in our materials array
            materials.append(start_mat)
            mat_by_name[start_mat.name] = start_mat
        else:
            start_mat = mat_by_name[mat_name]

        # Verify no other materials are in supply
        for to_mat in to_mats:
            if mat_by_name[to_mat].status == SUPPLY:
                msg = "Supply Materials can only be the first entry of a " \
                    "shuffle (see Material {})".format(to_mat)
                self.log("error", msg)

        # mat_set now contains the names of the materials in this set
        mat_set = [start_mat.name] + to_mats

        # Now get the list of cells that each material we are interested
        # in are assigned
        model_cells = self.universes[ROOT_UNIV].nested_cells
        mat_cells = []
        for name in mat_set:
            mat = mat_by_name[name]
            cell_id = None
            # Identify which in-core cell owns this material
            # (This makes assumption user never needs to move
            # ex-core to ex-core)
            # if none, then we will have None still
            for cell in model_cells.values():
                if cell.material_id == mat.id:
                    cell_id = cell.id
            mat_cells.append(cell_id)

        # Make sure all volumes are consistent and warn user if not.
        volumes = []
        if mat_by_name[mat_name].status == IN_CORE:
            # Then we add in the in-core volume
            volumes.append(mat_by_name[mat_name].volume)
        for to_mat in to_mats:
            volumes.append(mat_by_name[to_mat].volume)

        # Now we can compare; We can assume there is only one volume per
        # material based on the check in the block above. First, we will check
        # that the volumes of the materials being shuffled are all set.
        if None in volumes:
            msg = "At least one of the in-core materials being shuffled does " \
                  "not have a volume set!"
            self.log("warning", msg)
        elif not np.allclose(volumes[0], volumes[1:], rtol=VOLUME_PRECISION):
            msg = "Material volumes within this shuffle set are not " + \
                "within {}% of each other!".format(VOLUME_PRECISION * 100.)
            self.log("warning", msg)

        # Now we can step through and perform the actual move
        for i in range(len(mat_set)):
            if i == len(mat_set) - 1:
                # Then the target is actually the 1st location
                j = 0
            else:
                # Otherwise the target is the next location
                j = i + 1
            mat = mat_by_name[mat_set[i]]

            # Bypass any move positions that actually are ex-core
            if mat_cells[j] is not None:
                target_cell = self.cells[mat_cells[j]]
                mat.volume = target_cell.volume
                target_cell.material = mat
                mat.status = IN_CORE
            else:
                mat.status = STORAGE

    def _move_universes(self, univ_name, to_univ_names, materials, depl_libs):
        """Perform a shuffle by exchanging the universe that are
        are assigned to a cell in the model.

        Parameters
        ----------
        univ_name : str
            Name of universe to start the move
        to_univ_names : Iterable of str
            Name of universes to be displaced; the 0th entry is
            the location for univ_id to move to, the 1st for the
            universe displaced by the 0th move to go to, and so on and
            so forth. The last displaced will effectively go to storage.
        materials : List of adder.Material
            The materials to work with.
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        """

        # Convert the universe names to the identifiers
        univ_id = self.universe_by_name(univ_name).id
        to_univs = [self.universe_by_name(n).id for n in to_univ_names]

        # Make sure all universes exist somewhere in the model
        # and that they do not have an invalid value
        univs_set = [univ_id] + to_univs
        for univ in univs_set:
            if univ not in self.universes:
                msg = "Universe {} does not exist".format(univ) + \
                    " and thus cannot be shuffled!"
                self.log("error", msg)

            if univ == ROOT_UNIV:
                msg = "Cannot shuffle the root universe!"
                self.log("error", msg)

        # If we need to clone the universe, do it
        start_univ = self.universes[univ_id]
        if start_univ.status == SUPPLY:
            start_univ = self.clone_universe(univ_id, depl_libs,
                materials, new_status=IN_CORE)

        start_id = start_univ.id

        # univ_set now contains the universe ids in this movement set
        univ_set = [start_id] + to_univs

        # Verify no other universes are in supply
        for t, to_univ in enumerate(to_univs):
            if self.universes[to_univ].status == SUPPLY:
                msg = "Supply Universes can only be the first entry of a " \
                    "shuffle (see Universe {})".format(to_univ_names[t])
                self.log("error", msg)

        # Find the cells which own these universes (i.e., one level up)
        root_cells = self.universes[ROOT_UNIV].nested_cells
        univ_cells = []
        for u_id in univ_set:
            cell_ids = []
            # Look at all cells in the root universe to find out which
            # owns the universe whose id is u_id
            for cell in root_cells.values():
                if cell.fill is not None:
                    if u_id in cell.fill_ids:
                        cell_ids.append(cell.id)
            # We want to make sure there aren't multiple cells owning
            # this universe as that would be non-mass-conserving.
            if len(cell_ids) > 1:
                msg = "Cannot move a universe which is present in " \
                    "multiple cells!"
                self.log("error", msg)
            elif len(cell_ids) == 0:
                univ_cells.append(None)
            else:
                univ_cells.append(cell_ids[0])

        # Get the indices of the positions in the fill array to move
        # Note that if the fill is a single universe (vice array), then
        # the index of interest will be None
        locs = []
        for i in range(len(univ_set)):
            cell_id = univ_cells[i]
            if cell_id is not None:
                cell = self.cells[cell_id]
                fill_ids = cell.fill_ids
                index = np.where(fill_ids == univ_set[i])
                locs.append((index[0][0], index[1][0], index[2][0]))
            else:
                locs.append(None)

        # Now we can step through and perform the actual move
        for i in range(len(univ_set)):
            if i == len(univ_set) - 1:
                # Then the target is actually the 1st location
                j = 0
            else:
                # Otherwise the target is the next location
                j = i + 1

            # Bypass any move positions that actually are ex-core
            cell_id = univ_cells[j]
            if cell_id is not None:
                target_cell = self.cells[cell_id]
                if target_cell.fill_type == "single":
                    target_cell.fill[0][0][0] = self.universes[univ_set[i]]
                elif target_cell.fill_type == "array":
                    target_cell.fill[locs[j][0]][locs[j][1]][locs[j][2]] = \
                        self.universes[univ_set[i]]
            else:
                # Then this is going to storage
                self.universes[univ_set[i]].status = STORAGE

    def clone_universe(self, u_id, depl_libs, model_mats, new_name=None,
        new_status=None):
        """Create a copy of this Universe with a new unique ID and new
        IDs for all constituents, who are also cloned.

        Parameters
        ----------
        u_id : int
            The universe ID to clone
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        model_mats : List
            The materials in use in the model
        new_name : str or None, optional
            The name to apply to the clone; if None then [#] will be
            appended where # is the number of copies
        new_status : {IN_CORE, STORAGE, SUPPLY}
            The status to apply to the universe

        Returns
        -------
        clone : Universe
            The clone of this Universe

        """

        if u_id in self.universes:

            # The number of nested cells in the universe that is being cloned
            # will be checked against the number of nested cells in its clone to
            # determine if any cells have been split.
            self.universes[u_id].update_nested_children()
            num_cells_before_clone = len(self.universes[u_id].nested_cells)

            clone, logs = \
                self.universes[u_id].clone(
                    depl_libs, self.universes, self.cells, model_mats,
                    new_name=new_name)
            self.update_logs(logs)

            clone.update_nested_children()
            num_cells_after_clone = len(clone.nested_cells)

            if num_cells_after_clone > num_cells_before_clone:
                msg = "There are more cells in universe {} than the universe " \
                      "it was cloned from due to a lattice structure being " \
                      "split up. The volumes for cells in universe {} should " \
                      "be checked.".format(clone.id, clone.id)
                self.log("warning", msg)

            if new_status is not None:
                clone.status = new_status
            else:
                clone.status = self.universes[u_id].status
        else:
            msg = "{} is not a valid universe id!".format(u_id)
            self.log("error", msg)

        return clone

    def geom_search(self, transform_type, axis, names, angle_units, k_target,
                    bracket_interval, target_interval, uncertainty_fraction,
                    initial_guess, min_active_batches, max_iterations,
                    materials, case_idx, operation_idx, depl_libs):
        """Performs a search of geometric transformations to identify that
        which yields the target k-eigenvalue within the specified interval.

        Parameters
        ----------
        transform_type : str
            The type of object to transform; this depends on the actual
            neutronics solver type; this could be "material" or
            "universe", for example.
        axis : constants.VALID_GEOM_SWEEP_AXES
            The axis of interest to sweep
        names : List of str
            The names of the components of type `type_` to apply the
            perturbation to
        angle_units : {"degrees", "radians"}
            The units of the above angles
        k_target : float
            The target k-eigenvalue to search for
        bracket_interval : Iterable of float
            The lower and upper end of the ranges to search.
        target_interval : float
            The range around the k_target for which is considered success
        uncertainty_fraction : float
            This parameter sets the stochastic uncertainty to target when the
            initial computation of an iteration reveals that a viable solution
            may exist.
        initial_guess : float
            The starting point to search
        min_active_batches : int
            The starting number of batches to run to ensure enough keff samples
            so that it follows the law of large numbers. This is the number of
            batches run when determining if a case has a chance of being a
            suitable solution. This only applies to Monte Carlo solvers.
        max_iterations : int
            The total number of search iterations to perform before terminating
            the search.
        materials : List of Material
            The model materials
        case_idx : int
            The case index, for printing
        operation_idx : int
            The operation index, for printing
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model

        Returns
        -------
        converged : bool
            A flag denoting whether or not the solution was converged
        iterations : int
            The number of iterations required
        val_new : float
            The latest control group displacement
        keff_new : float
            The obtained value of k-effective
        keff_std_new : float
            The obtained k-effective (1-sigma) uncertainty

        """

        # Helper function to execute each case
        def _eval(val, axis, names, angle_units, transform_type,
                  materials, case_idx, operation_idx, this, min_sim_settings,
                  k_target, target_interval, iteration, depl_libs):
            msg = "Iteration {}: Position {:.3f}".format(iteration, val)
            this.log("info", msg, 8)
            # Perform the transform
            yaw, pitch, roll, displacement = get_transform_args(val, axis)
            matrix = None
            this.transform(names, yaw, pitch, roll, angle_units, matrix,
                           displacement, transform_type,
                           transform_in_place=False)

            # Swap out the sim settings data for the minimum case
            orig_sim_settings = this.sim_settings
            this.sim_settings = min_sim_settings

            # Execute the neutronics solver
            fname = "crit_search_model"
            step_label = "Op {} Case {} Search Position {:.3f}".format(
                case_idx, operation_idx, val)

            # Re-run the executor, but set ourselves up for a continue run
            inp_name = fname + ".inp"
            if iteration == 0:
                update_iso_status = True
            else:
                update_iso_status = False
            this.write_input(inp_name, step_label, materials, depl_libs,
                store_input, deplete_tallies, user_tallies, user_output,
                update_iso_status)
            this._exec_solver(inp_name, fname, keep_runtpe=True)
            keff, keff_stddev, _, _, _ = \
                this._read_results(fname, materials, depl_libs,
                    deplete_tallies)

            # Cleanup the files
            for out_type in MCNP_OUT_SUFFIX:
                if out_type != "r":
                    try:
                        os.remove(fname + out_type)
                    except FileNotFoundError:
                        pass
            for outfile in MCNP_OUT_NAMES:
                try:
                    os.remove(outfile)
                except FileNotFoundError:
                    pass

            # Now we determine if do an early-reject or not.
            # This uses a 99.5% CI-based rejection: if our k +/- 99.5%CI is
            # outside of the target and its range, then we reject this case
            target_lo = k_target - target_interval
            target_hi = k_target + target_interval
            keff_995hi = keff + CI_995 * keff_stddev
            keff_995lo = keff - CI_995 * keff_stddev
            is_below = keff_995hi < target_lo
            is_above = keff_995lo > target_hi

            # If it is inside the range, then that means it is not below and
            # not above
            if not is_below and not is_above:
                # Then we have a keeper
                # Run more histories. How many should we run? Well, do it so
                # that our 95%CI is about uncertainty_fraction * the target interval
                new_active_batches = \
                    (min_sim_settings.batches - min_sim_settings.inactive) * \
                    (CI_95 * keff_stddev /
                     (target_interval * uncertainty_fraction))**2

                new_batches = this.sim_settings.inactive + \
                    round(new_active_batches)

                # Now re-run with this batch estimate
                this.sim_settings.batches = new_batches
                this._write_continue(inp_name)
                os.rename(fname + "r", "runtpe")
                this._exec_solver(inp_name, fname, is_continue=True)
                os.remove(fname + "_runtpe")

                keff, keff_stddev, _, _, _ = \
                    this._read_results(fname, materials, depl_libs,
                        deplete_tallies)

                # Cleanup all the files
                for out_type in MCNP_OUT_SUFFIX:
                    try:
                        os.remove(fname + out_type)
                    except FileNotFoundError:
                        pass
                for outfile in MCNP_OUT_NAMES:
                    try:
                        os.remove(outfile)
                    except FileNotFoundError:
                        pass

            # Put the sim_settings back to the original
            this.sim_settings = orig_sim_settings

            # Now put the surface back where it started from so we can
            # compute the difference correctly next time
            yaw, pitch, roll, displacement = get_transform_args(-val, axis)
            this.transform(names, yaw, pitch, roll, angle_units, matrix,
                           displacement, transform_type,
                           transform_in_place=False)

            return keff, keff_stddev

        # Save constants used for each case
        store_input = False
        user_tallies = True
        user_output = True
        deplete_tallies = False
        iterations = 0

        # Make a copy of the simulation settings that uses the minimum number
        # of histories
        min_sim_settings = copy.deepcopy(self.sim_settings)
        min_sim_settings.batches = min_active_batches + \
            min_sim_settings.inactive

        # First we need to evaluate the lower bracket point
        val_old = bracket_interval[0]
        keff_old, _ = \
            _eval(val_old, axis, names, angle_units, transform_type, materials,
                  case_idx, operation_idx, self, min_sim_settings, k_target,
                  target_interval, iterations, depl_libs)
        iterations += 1
        # Set iterations to non-zero so that _eval doesnt update isos_in_neuts
        # statuses

        # And the initial guess
        val_new = initial_guess
        keff_new, keff_std_new = \
            _eval(val_new, axis, names, angle_units, transform_type, materials,
                  case_idx, operation_idx, self, min_sim_settings, k_target,
                  target_interval, iterations, depl_libs)
        iterations = 0

        # Now we can start our loop
        converged = False
        while True:
            iterations += 1
            # Check for convergence
            target_lo = k_target - target_interval
            target_hi = k_target + target_interval
            keff_95lo = keff_new - CI_95 * keff_std_new
            keff_95hi = keff_new + CI_95 * keff_std_new

            if target_lo <= keff_95lo and keff_95hi <= target_hi:
                converged = True
                break
            elif iterations > max_iterations:
                break
            elif keff_new == keff_old or val_new == val_old:
                # Then the search has no effect on keff, or
                # the likely value is out of bounds
                converged = True
                break

            # If we made it here then we need to
            # Compute the differential rod worth
            dk_dz = (keff_new - keff_old) / (val_new - val_old)
            # And use that to compute the next guess
            val_next = (k_target - keff_new) / dk_dz + val_new

            # Make sure val_next is in our bracket range
            if val_next < bracket_interval[0]:
                val_next = bracket_interval[0]
            elif val_next > bracket_interval[1]:
                val_next = bracket_interval[1]

            # Move our new values to beold
            val_old = val_new
            keff_old = keff_new
            # Now update results at val_next to be at keff_2
            val_new = val_next
            keff_new, keff_std_new = \
                _eval(val_new, axis, names, angle_units, transform_type,
                      materials, case_idx, operation_idx, self,
                      min_sim_settings, k_target, target_interval, iterations,
                      depl_libs)

        return converged, iterations, val_new, keff_new, keff_std_new

    def transform(self, names, yaw, pitch, roll, angle_units, matrix,
                  displacement, transform_type, transform_in_place=False):
        """Transforms the universe named `name` according to the angles in
        yaw, pitch, and roll and the translation in displacement.

        Parameters
        ----------
        names : str
            The names of the item to transform.
        yaw : None or float
            The angle to rotate about the z-axis. If None, then the matrix is
            used instead.
        pitch : None or float
            The angle to rotate about the y-axis. If None, then the matrix is
            used instead.
        roll : None or float
            The angle to rotate about the x-axis. If None, then the matrix is
            used instead.
        angle_units : {"degrees", "radians"}
            The units of the above angles; ignored if a matrix is provided
        matrix : None or np.ndarray
            The rotation matrix, provided in a 3x3 matrix. If None, then yaw,
            pitch, and roll are used instead.
        displacement : Iterable of float
            The displacement vector to translate the object
        transform_type : str
            The type of object to transform; this depends on the actual
            neutronics solver type; this could be "cell" or
            "universe", for example.
        transform_in_place : bool, optional
            If the existing transform should be updated (True) or a new
            transform created. Defaults to False.
        """

        # Use the base class' type checking
        super().transform(names, yaw, pitch, roll, angle_units, matrix,
                          displacement, transform_type, transform_in_place)

        # Now check the transform_type value
        check_value("transform_type", transform_type,
                    self.VALID_TRANSFORM_TYPES)

        # Now set the rotation angles and/or matrix to what they need to be
        # for Transfor object
        if matrix is None:
            ypr = [yaw, pitch, roll]
        else:
            ypr = None

        if transform_type == "universe":
            self._transform_universes(names, ypr, angle_units, matrix,
                                      displacement, transform_in_place)
        elif transform_type == "cell":
            self._transform_cells(names, ypr, angle_units, matrix,
                                  displacement, transform_in_place)
        elif transform_type == "surface":
            self._transform_surfaces(names, ypr, angle_units, matrix,
                                     displacement, transform_in_place)

        # Now we have to merge and clean all of our transforms so the IDs
        # in the model does not grow with each transform operation
        CoordTransform.merge_and_clean(self.coord_transforms,
                                       self.cells, self.surfaces)


    def _transform_universes(self, names, ypr, angle_units, matrix,
                             displacement, transform_in_place):
        """Transforms the universe named `name` according to the angles in
        yaw, pitch, and roll (or the provided matrix) and the translation in
        displacement.

        Note that this will operate on all instances of the universe,
        even if that is not desired.
        """

        # Create the transform to potentially be used by all
        transform = CoordTransform(None, displacement, ypr, matrix,
                                   in_degrees=angle_units == "degrees")
        used_transform = False

        # Convert the universe names to the identifiers
        univ_ids = [self.universe_by_name(n).id for n in names]

        # Now go find where each universe is used as a fill, and
        # incorporate this transform.
        for u_id in univ_ids:
            if u_id == ROOT_UNIV:
                msg = "Cannot transform root universe"
                self.log("error", msg)

            # Find the cells which own this universe (i.e., one level up)
            # Look at all cells in the root universe to find out which
            # owns the universe whose id is u_id
            for cell in self.cells.values():
                if cell.fill is not None:
                    fill_ids = cell.fill_ids
                    if u_id in fill_ids:
                        # Get the indices of the positions in the fill
                        # array to move Note that if the fill is a
                        # single universe (vice array), then the index
                        # of interest will be None
                        index = np.where(fill_ids == u_id)

                        # Do for each of the indices in index
                        for n_idx in range(len(index[0])):
                            idx = (index[0][n_idx], index[1][n_idx],
                                   index[2][n_idx])
                            # See if there is a transform here already
                            tf = cell.fill_transforms[idx[0]][idx[1]][idx[2]]
                            if tf is not None:
                                # A transform exists, so we need to apply
                                # this one to it to create a new transform
                                tf = tf.combine(transform,
                                                in_place=transform_in_place)
                                # Apply the new transform
                                # and save it globally
                                cell.fill_transforms[idx[0]][idx[1]][idx[2]] = tf
                                self.coord_transforms[tf.id] = tf
                            else:
                                cell.fill_transforms[idx[0]][idx[1]][idx[2]] = transform
                                used_transform = True

        if used_transform:
            # This way only if we use our base transform do we need
            # to actually keep it, saving one of our 999 ids.
            self.coord_transforms[transform.id] = transform

    def _transform_cells(self, names, ypr, angle_units, matrix, displacement,
                         transform_in_place):
        """Transforms the cells named `name` according to the angles in
        yaw, pitch, and roll (or the provided matrix) and the translation in
        displacement.
        """

        # Convert names to ids
        ids = [input_utils.num_format(name, 'int') for name in names if
               name.isdigit()]

        # Create the transform to potentially be used by all
        transform = CoordTransform(None, displacement, ypr, matrix,
                                   in_degrees=angle_units == "degrees")
        used_transform = False
        for c_id in ids:
            cell = self.cells[c_id]

            # See if there is a transform already here
            tf = cell.coord_transform
            if tf is not None:
                # A transform exists, so we need to apply the new
                # transform to it, creating a new transform
                tf = tf.combine(transform, in_place=transform_in_place)

                # Apply the new transform to the cell and save it
                # globally
                cell.coord_transform = tf
                self.coord_transforms[tf.id] = tf
            else:
                cell.coord_transform = transform
                used_transform = True

        # Now if we used the new transform, we must store it so it
        # gets written to the model
        if used_transform:
            # This way only if we use our base transform do we need
            # to actually keep it, saving one of our 999 ids.
            self.coord_transforms[transform.id] = transform

    def _transform_surfaces(self, names, ypr, angle_units, matrix,
                            displacement, transform_in_place):
        """Transforms the surfaces named `name` according to the angles in
        yaw, pitch, and roll (or the provided matrix) and the translation in
        displacement.
        """

        # Convert names to ids
        ids = [input_utils.num_format(name, 'int') for name in names if
               name.isdigit()]

        # Create the transform to potentially be used by all
        transform = CoordTransform(None, displacement, ypr, matrix,
                                   in_degrees=angle_units == "degrees")
        used_transform = False
        for surf_id in ids:
            surf = self.surfaces[surf_id]

            # See if there is a transform already here
            tf = surf.coord_transform
            if tf is not None:
                # A transform exists, so we need to apply the new
                # transform to it, creating a new transform
                tf = tf.combine(transform, in_place=transform_in_place)
                # Apply the new transform to the cell and save it
                # globally
                surf.coord_transform = tf
                self.coord_transforms[tf.id] = tf
            else:
                surf.coord_transform = transform
                used_transform = True

        # Now if we used the new transform, we must store it so it
        # gets written to the model
        if used_transform:
            # This way only if we use our base transform do we need
            # to actually keep it, saving one of our 999 ids.
            self.coord_transforms[transform.id] = transform


def _run_volume(messages, title, this, mat_cards, output, others,
                seed, first_pass, cells_to_obtain):

    other_cards = others + ["rand gen=2 seed={}".format(seed)]

    # Get the list of files that exist before we execute
    dir_contents = glob.glob("*")

    # Now we can write the input
    inp_name = "volume.inp"
    input_methods.create_input_file(inp_name, messages, title,
                                    this.universes, this.surfaces, mat_cards,
                                    this.coord_transforms, [], output,
                                    this.sim_settings, other_cards,
                                    is_volume_calc=True)

    # Now execute MCNP
    out_name = "v_calc"
    this._exec_solver(inp_name, out_name, print_log_msg=False)

    # if this is our first time, then we should get table 60 data
    volumes_by_cell = None
    if first_pass:
        with open(out_name + "o", "r") as file:
            line = file.readline()
            while line:
                table_num = output_methods._get_table_num(line)
                if table_num == 60:
                    volumes_by_cell = \
                        output_methods._get_table60_volumes(file)
                    break
                # Advance to the next line
                line = file.readline()

        # Filter volumes_by_cell to be only those we care about
        c_keys = list(volumes_by_cell.keys())
        for c in c_keys:
            if c not in cells_to_obtain:
                del volumes_by_cell[c]

    # Now get the table 126 data; here we read from the end as the
    # lost particles could be hundreds of thousands of lines
    ending_lines = []
    for line in output_methods.reverse_readline(out_name + "o"):
        table_num = output_methods._get_table_num(line)
        if table_num == 126:
            break
        ending_lines.append(line.strip())
    t126_to_end_lines = ending_lines[::-1]

    # Get table 126 data
    cell_scores, cell_total_scores = \
        output_methods._get_table126_scores(t126_to_end_lines)

    # Now remove any files that are new
    new_dir_contents = glob.glob("*")
    new_files = list(set(new_dir_contents) - set(dir_contents))
    # Clean up the new files so the directory is as it was when we started
    for file in new_files:
        os.remove(file)

    return cell_scores, cell_total_scores, volumes_by_cell


def _convert_cell_hits_to_volume(scores, tot_hits, volumes_by_cell,
                                 cells_to_obtain, V_tot, running_scores,
                                 running_hits, target_unc):
    # First incorporate the latest gen
    for c in scores:
        if c in cells_to_obtain:
            if c in running_scores:
                running_scores[c] += scores[c]
            else:
                running_scores[c] = scores[c]
    running_hits += tot_hits

    # Convert to a volume and uncertainty
    vols = OrderedDict()
    vols_unc = OrderedDict()
    for c_id in running_scores.keys():
        vols[c_id] = running_scores[c_id] * V_tot / running_hits
        vols_unc[c_id] = np.sqrt(vols[c_id] *
                                 (V_tot - vols[c_id]) / running_hits)

    # Now convert relative errors to actual relative errors and use
    # analytic volumes
    for c in volumes_by_cell.keys():
        # If a cell is in our list of ones to care about,
        # continue analyzing
        if c in cells_to_obtain:
            # Only consider those we dont have analytic values for
            if volumes_by_cell[c] == 0.:
                if c not in vols:
                    # Then we didnt get any hits in the cell,
                    # so the relative uncertainty should be
                    # set larger than the goal uncertainty
                    vols[c] = 0.
                    vols_unc[c] = 2. * target_unc
                else:
                    # Compute the rel err
                    if vols[c] == 0:
                        vols_unc[c] = 2. * target_unc
                    else:
                        vols_unc[c] *= 100. / vols[c]
            else:
                # Then we have an analytic solution, set unc. to 0
                vols[c] = volumes_by_cell[c]
                vols_unc[c] = 0.
        else:
            # Set the uncertainty to 0 so we dont wait for it to
            # converge
            vols_unc[c] = 0.

    return running_scores, running_hits, vols, vols_unc
