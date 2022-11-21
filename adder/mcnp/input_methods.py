import re
import warnings
from collections import OrderedDict

import numpy as np

import adder.data
from adder.constants import AVOGADRO, IN_CORE, BASE_LIB
from .constants import *
from .cell import Cell
from .coord_transform import CoordTransform
from .input_utils import *
from .sim_settings import SimSettings
from .surface import Surface

# Note on lacking features:
# This tool does not support:
# - The READ command [Sec 3.1]
# - Vertical format is NOT supported


def get_tallies(data_block):
    """Obtain cards of tallies present in the input.

    No processing or extraction of information from these tally cards is
    done; this method simply finds tallies and pulls them out of
    data_block.

    Parameters
    ----------
    data_block : List of str
        The cards within the data block

    Returns
    -------
    List of str:
        Tally cards
    List of str:
        Remaining cards
    max_tally_id : int
        The maximum tally id of a user tally, to see if there are any
        conflicts
    multiplier_mat_ids : Set of int
        The ids of the materials which are relied on for tally
        multipliers
    """

    tally_cards = []
    other_cards = []
    multiplier_mat_ids = set()
    max_tally_id = 0
    active_tmesh_block = False
    for line in data_block:
        found_it = False
        # See if this line starts with any of our tally card identifiers
        if not active_tmesh_block:
            # See if this is the beginning of a TMESH block
            if line.lower().startswith(TMESH_CARDS[0]):
                # This means we have the start of this tally type
                # Also store it
                found_it = True
                tally_cards.append(line)
                # But we also should denote that this is an
                # active tmesh block so that we store all others
                # until we find the end of the tmesh block
                active_tmesh_block = True

            if not found_it:
                for key in TALLY_CARDS:
                    # The isdigit portion is needed because sdef
                    # and sd* are possible, one is a tally, one isnt
                    if line.lower().startswith(key) and \
                        line[len(key)].isdigit():

                        # Then it is a tally
                        # Check if the tally ID is too large
                        # and if not, store the tally

                        # To get the number, isolate the tally type
                        # and tally id
                        tally_dat = line.split()[0].split(":")[0]
                        # Ring detector tally types have an x, y, or z before
                        # the ":"; so to get the tally id we need to discard
                        # the x, y, or z
                        if tally_dat[0].lower() == "f" and \
                            tally_dat[-1].lower() in  ["x", "y", "z"]:
                            # Remove the x/y/z
                            tally_dat = tally_dat[:-1]
                        tally_id = num_format(tally_dat[len(key):], 'int')
                        if max_tally_id < tally_id:
                            max_tally_id = tally_id
                        if tally_id >= ADDER_TALLY_ID:
                            msg = "Tally {} has an ID ".format(tally_id)
                            msg += "that will conflict with ADDER IDs; " + \
                                   "all tally ids greater than "
                            msg += "{}".format(ADDER_TALLY_ID - 1) + \
                                   "must be reduced "
                            raise ValueError(msg)
                        tally_cards.append(line)
                        found_it = True

                        # Before we quit, lets analyze any FM cards to
                        # see what multiplier material ids are used
                        if key == "fm":
                            multiplier_mat_ids.update(_analyze_fm_card(line))
                        break

            # Now if we got here and found_it is still false, then we
            # can put it in the other cards bin
            if not found_it:
                other_cards.append(line)
        else:
            # If we are here then we are in the midst of an active
            # tmesh block, and so we store all cards within the block
            # as tallies and have to check to see when the block ends
            tally_cards.append(line)
            if line.lower().startswith(TMESH_CARDS[1]):
                active_tmesh_block = False

    return tally_cards, other_cards, max_tally_id, multiplier_mat_ids


def get_output(data_block):
    """Obtain cards of the output control present in the input.

    Specifically, these are the "PRINT", "TALNP", "PRDMP", "PTRAC",
    "MPLOT", "HISTP", and "DBCN" cards. No processing or extraction of
    information from these cards is done; this method simply finds the
    cards and pulls them out of data_block.

    Parameters
    ----------
    data_block : List of str
        The cards within the data block

    Returns
    -------
    output_cards : List of str
        Output control cards
    other_cards : List of str
        Remaining cards
    """

    output_cards = []
    other_cards = []

    # This is called after _get_tallies so we do not need to look for
    # TMESH
    for line in data_block:
        found_it = False
        # See if this line starts with any of our output card types
        for key in OUTPUT_CARDS:
            if line.lower().startswith(key):
                # Then it is an output control card, store it as such
                output_cards.append(line)
                found_it = True
                break

        # Now if we got here and found_it is still false, then we
        # can put it in the other cards bin
        if not found_it:
            other_cards.append(line)

    return output_cards, other_cards


def get_sim_settings(data_block):
    """Extra input data needed for the simulation settings to populate the
    SimSettings object.

    Specifically, this is the "kcode" card.

    Parameters
    ----------
    data_block : List of str
        The cards within the data block

    Returns
    -------
    sim_settings : SimSettings
        Simulation Settings object
    other_cards : List of str
        Remaining cards
    """

    sim_settings_cards = []
    other_cards = []

    for line in data_block:
        found_it = False
        # See if this line starts with any of our output card types
        for key in SIM_SETTING_CARDS:
            if line.lower().startswith(key + " "):
                # Then it is a simulation setting card, store it as such
                if key == 'kcode':
                    line = " ".join(expand_jumps(line.split()))
                sim_settings_cards.append(line)
                found_it = True
                break

        # Now if we got here and found_it is still false, then we
        # can put it in the other cards bin
        if not found_it:
            other_cards.append(line)

    # Ok now initialize the simulation settings object
    sim_settings = SimSettings.from_cards(sim_settings_cards)

    return sim_settings, other_cards


def parse_materials(data_block):
    """Given the data block, this returns the materials defined within.

    Parameters
    ----------
    data_block : List of str
        The data block without comment lines or inline comments

    Returns
    -------
    m_data : OrderedDict
        The dictionary will have the material ids as keys and the
        values are Lists of 3-tuples of (name, fraction, nlib)
    m_default_nlib : OrderedDict()
        The default neutron cross section library for each material
    mt_cards : OrderedDict
        The dictionary will have the material ids as keys and the
        values are Lists of lower-case S(a,b) identifiers
        (e.g., "lwtr.10t")
    mat_cards : Iterable of str
        The unparsed material cards.
    other_cards : Iterable of str
        The unparsed non-material cards

    """

    # Go through and pull out material (m#) and material-thermal (mt#)
    # cards
    m_cards = OrderedDict()
    mt_cards = OrderedDict()
    other_cards = []
    mat_cards = []

    for card in data_block:
        # Split apart the card type (and id for m*, mt*, s*, ...)
        # from the card data
        split_card = card.split(maxsplit=1)
        if len(split_card) < 2:
            # Then we have one of those rare simple-statement cards
            # (i.e., TMESH); just move on its not a material anyways
            other_cards.append(card)
            continue
        else:
            card_type_and_id, card_data = split_card[:]

        # Determine if this is a material (or MT) card or not
        test_card = card_type_and_id.lower()
        if test_card[0] == "m" and test_card not in NOT_MAT_CARDS:
            # Now differentiate between MT and M and store in the right
            # place
            if test_card[1] == "t":
                # Then this is MT
                card_id = num_format(card_type_and_id[2:], 'int')
                if card_id in mt_cards:
                    msg = "". join(["Card ", card_type_and_id,
                                    " defined multiple times; ",
                                    "only the last entry will be used."])
                    warnings.warn(msg)
                mt_cards[card_id] = card_data.strip().split()
                mat_cards.append(card)
            else:
                card_id = num_format(card_type_and_id[1:], 'int')
                if card_id in m_cards:
                    msg = "". join(["Card ", card_type_and_id,
                                    " defined multiple times; ",
                                    "only the last entry will be used."])
                    warnings.warn(msg)
                m_cards[card_id] = card_data.lstrip()
                mat_cards.append(card)

        else:
            # Then we dont want it now but will want to separate it out
            other_cards.append(card)

    # First see if there is an m0 card with the default libraries
    default_nlib = ""
    default_pnlib = ""
    if MATL_VOID in m_cards:
        # Then we have one, get the parameters
        # zaid_fracs should be empty for an m0 card
        zaid_fracs, keywords = split_data_and_keywords(m_cards[0], MAT_KW)

        if len(zaid_fracs) > 0:
            # Raise a warning since this is trivial for us to find and
            # warn
            msg = "The m0 card contains ZAIDs and atom fractions, this" + \
                " is invalid MCNP input!"
            warnings.warn(msg)

        # Lets grab the default libraries we need
        if "nlib" in keywords:
            default_nlib = keywords["nlib"]
            # Remove the leading decimal if it exists
            if default_nlib[1] == ".":
                default_nlib = default_nlib[1:]
        if "pnlib" in keywords:
            default_pnlib = keywords["pnlib"]
            # Remove the leading decimal if it exists
            if default_pnlib[1] == ".":
                default_pnlib = default_pnlib[1:]

    # Now extract the ZAID, fractions, nlib, pnlib) for each isotope
    # within a material card
    zaids = OrderedDict()
    fractions = OrderedDict()
    m_nlibs = OrderedDict()
    m_pnlibs = OrderedDict()
    m_data = OrderedDict()
    for m, data in m_cards.items():
        if m == MATL_VOID:
            # We already have this, so just skip it
            continue

        # First set the nlib and pnlib to be the defaults from m0
        # (if no m0, then we just assign "", which is a great
        # initialized state)
        m_nlibs[m] = default_nlib
        m_pnlibs[m] = default_pnlib

        # Table 3-32 defines material cards as having some number of
        # zaid and fraction pairs, followed by the keywords
        # So first lets separate the data (zaid/fracs) and keywords
        zaid_fracs, keywords = split_data_and_keywords(data, MAT_KW)

        zaids[m] = []
        fractions[m] = []
        split_zaid_fracs = zaid_fracs.split()
        for i in range(0, len(split_zaid_fracs), 2):
            zaids[m].append(split_zaid_fracs[i])
            fractions[m].append(num_format(split_zaid_fracs[i + 1], 'float'))

        # Normalize the fractions and make sure they are all the
        # same sign
        # Easier to work with numpy arrays, so do that
        fractions[m] = np.array(fractions[m])
        if np.all(fractions[m] >= 0.):
            is_positive = True
        elif np.all(fractions[m] <= 0.):
            is_positive = False
        else:
            raise ValueError("Atomic fractions in material {} ".format(m) +
                             "must be the same sign")

        fractions[m] /= np.sum(fractions[m])
        if not is_positive:
            fractions[m] *= -1.

        # And get our nlib and pnlib information
        if "nlib" in keywords:
            m_nlibs[m] = keywords["nlib"]

        if "pnlib" in keywords:
            m_pnlibs[m] = keywords["pnlib"]

        # Now convert this material's processed info into what we
        # need to return
        m_data[m] = []
        for i, zaid in enumerate(zaids[m]):
            # See if there is a library specified
            index_separator = zaid.find(".")
            if index_separator != -1:
                # Then there is one. Get it and find the type of lib
                lib_id = zaid[index_separator + 1:]
                lib_class = lib_id[-1]

                # Assign this lib_id to the right xs class (neut, pn)
                # and use the default for the other.
                if lib_class in ["c", "d", "m"]:
                    # c == C.E., d == discrete neutron, m = neutron MGXS
                    # then this is for neutrons
                    nlib = lib_id
                    pnlib = m_pnlibs[m]
                elif lib_class in ["u", "g"]:
                    # this is photoneutron library
                    nlib = m_nlibs[m]
                    pnlib = lib_id
                else:
                    # The lib_id is not of interest, use defaults
                    nlib = m_nlibs[m]
                    pnlib = m_pnlibs[m]
            else:
                # then we are using defaults
                nlib = m_nlibs[m]
                pnlib = m_pnlibs[m]

            # Now we want to figure out the isotope name
            if index_separator == -1:
                zaid_num = num_format(zaid, 'int')
            else:
                zaid_num = num_format(zaid[:index_separator], 'int')
            name, element, Z, A, M = \
                adder.data.get_metadata(zaid_num, metastable_scheme="mcnp")

            # Raise an error if we dont know the library type
            if nlib == "":
                msg = "ZAID {} in Material {}".format(zaid_num, m) + \
                    " does not have a neutron cross section library specified!"
                raise ValueError(msg)

            # Now put the data in the preferred format
            # FUTURE TODO: Include pnlib here
            m_data[m].append((name, fractions[m][i], nlib))

    # The reader will notice we do nothing with pnlib despite processing
    # it; This is to minimize what we have to do later for pnlibs, but
    # for now ADDER does not handle it.

    return m_data, m_nlibs, mt_cards, mat_cards, other_cards


def parse_cells(cell_block, transforms):
    """Given the block of cell cards, this returns the instantiated
    Cell objects.

    Coordinate transforms objects are used in place of their ids or
    TRCL cards in the cell block

    This method assumes 1) multi-line cards have been pre-processed
    and placed on a single line, and 2) that there are no "LIKE"
    commands, either because they are restricted, or they have been
    parsed out.

    The cell id is checked to be between 1 and 99,999,999. The material
    id is between 0 and 99,999,999 where 0 denotes void, and positive
    denotes material ids. A positive density denotes units of
    atoms/b-cm, negative density is units of g/cm^3.

    Parameters
    ----------
    cell_block : List of str
        The cell lines from the input file
    transforms : dict
        The parsed transforms (CoordTransform objects) keyed by their id

    Returns
    -------
    cells : dict
        The parsed cells (Cell objects) keyed by the cell id
    """

    # Per Section 3.2, Form 1 of a cell card is defined as:
    # "<cell id> <material id> <density> <geom> <...params...>" where
    # <density> is not provided if the cell is a void
    not_like_cells = []

    # First do all the cells that do not contain a "like-but" construct
    like_card_nums = []
    cell_ids_in_order = []
    for card_num, cell_card in enumerate(cell_block):
        if "like" not in cell_card.lower():
            not_like_cells.append(Cell.from_not_like_line(cell_card,
                                                          transforms))
        else:
            # Save the line for later
            like_card_nums.append(card_num)
        # Save the ID in order of presentation
        cell_ids_in_order.append(num_format(cell_card.split(maxsplit=1)[0],
                                            'int'))

    # Now do the cells with "LIKE-BUT"
    like_cells = []
    for card_num in like_card_nums:
        like_cells.append(Cell.from_like_line(cell_block[card_num],
                                              not_like_cells,
                                              transforms))

    # Combine our cells
    parsed_cell_list = not_like_cells + like_cells

    # And put the cells back in the order they were in the original
    # input
    parsed_cell_ids = [c.id for c in parsed_cell_list]
    parsed_cell_in_order = []
    for c_id in cell_ids_in_order:
        # Find where the cell is in parsed_cell_list
        i = parsed_cell_ids.index(c_id)
        # Now put it in the original order
        parsed_cell_in_order.append(parsed_cell_list[i])
    parsed_cell_list = parsed_cell_in_order

    parsed_cells = OrderedDict()
    # Get the index of a cell in parsed_cells, given its cell id
    for cell in parsed_cell_list:
        parsed_cells[cell.id] = cell

    return parsed_cells


def parse_surfaces(surface_cards, transforms):
    """Given the data block, this returns the instantiated
    CoordTransform objects. These objects are those specifically
    identified by TR# or *TR# cards. Those within cell definitions
    will be handled separately.

    This method assumes 1) multi-line cards have been pre-processed
    and placed on a single line, and 2) that there are no "LIKE"
    commands, either because they are restricted, or they have been
    parsed out.

    Parameters
    ----------
    surface_cards : List of str
        The surface lines from the input file
    transforms : dict
        The parsed transforms (CoordTransform objects) keyed by their id

    Returns
    -------
    parsed_surfaces : dict of Surface
        The parsed surfaces keyed by their id

    """

    parsed_surfaces = {}

    for card in surface_cards:
        sfc = Surface.from_card(card, transforms)
        parsed_surfaces[sfc.id] = sfc

    return parsed_surfaces


def parse_transforms(data_block):
    """Given the data block, this returns the instantiated
    CoordTransform objects. These objects are those specifically
    identified by TR# or *TR# cards. Those within cell definitions
    will be handled separately.

    This method assumes 1) multi-line cards have been pre-processed
    and placed on a single line, and 2) that there are no "LIKE"
    commands, either because they are restricted, or they have been
    parsed out.

    Parameters
    ----------
    data_block : List of str
        The cell lines from the input file

    Returns
    -------
    transforms : dict
        The parsed transforms (CoordTransform objects) keyed by their id
    other_cards : Iterable of str
        The unparsed cards from data_block

    """

    # First step through and pull out our transform cards
    transform_cards = []
    other_cards = []

    for card in data_block:
        # First make this card lowercase
        card_lower = card.lower()
        # Now see if we have a valid option (tr, *tr)
        # Since TROPT is an irrelevant option, we will check and discard
        # that first
        if card_lower.startswith("tropt"):
            # Keep the original version so downstream can do what it
            # wants
            other_cards.append(card)
        elif card_lower.startswith("*tr") or card_lower.startswith("tr"):
            # Then we have a match! Keep the lower case version
            transform_cards.append(card_lower)
        else:
            # Not interested.
            other_cards.append(card)

    # Now we can start processing our transform cards
    transforms = dict()
    # First create a null transform as this is a feasible option
    transforms[0] = CoordTransform(0)
    for card in transform_cards:
        # Split out the keyword from the data
        keyword, card_data = card.split(maxsplit=1)
        # Expand the jumps
        card_data = " ".join(expand_jumps(card_data.split()))
        # Now we can initialize the object
        tr = CoordTransform.from_card(keyword, card_data)

        # And store by id
        transforms[tr.id] = tr

    return transforms, other_cards


def _add_data_cards_to_cells(cell_block, data_block):
    """In MCNP, you can add certain cell parameters as their own cards
    in the data block. This pre-processor takes that data block info
    and puts it as a keyword in the corresponding cell card.

    Parameters
    ----------
    cell_block : Iterable of str
        One line for each cell card
    data_block : Iterable of str
        One line for each data card

    Returns
    -------
    new_cell_block : Iterable of str
        The updated cell block
    new_data_block : Iterable of str
        Data block with the 'spent' cards discarded
    """

    new_cell_block = cell_block[:]
    new_data_block = []

    # Find any data blocks that should be put on the cell
    for line in data_block:
        lower_line = line.lower()
        case_found = False
        for case in CARD_TO_CELL:
            if case == "cosy" and lower_line.startswith("cosyp"):
                # Then we need to skip this one
                check = False
            elif lower_line.startswith(case):
                check = True
            else:
                check = False
            if check:
                # Get the individual values but discard the first since
                # it is just the card name
                line_data = lower_line.split()
                card_name = line_data[0]
                values = line_data[1:]

                # The only remaining way we expect case and card_name to not
                # match is the case is one with a particle designator. If so,
                # the card_name should end with the particle designator.
                # Make sure of that
                if case == "u" and not card_name.startswith("unc"):
                    if case[-1] != ":" and case != card_name:
                        msg = "The data card {} contains ".format(card_name) + \
                            "additional information beyond the expected " + \
                            "{}!".format(case)
                        raise ValueError(msg)

                # Now just make sure the particle designator is valid
                # We will do this in a 'forwards-compatible' way
                # (i.e., allowing for particle types with more than 1
                # character as an identifier)
                if case[-1] == ":":
                    # There could be multipe particle designators, split it
                    # with a ","
                    p_types = card_name[len(case):].split(",")
                    for p_type in p_types:
                        if p_type not in PARTICLE_TYPES:
                            msg = "Invalid particle type on data card {}!"
                            raise ValueError(msg.format(card_name))

                # We have a special case to check for; if doing a volume
                # card, and the parameter is "NO", then the user is
                # telling MCNP not to calculate volume
                if card_name == "vol" and values[0].lower() == "no":
                    # Just skip over that part
                    values = values[1:]

                # Remove repeated jumps
                values = expand_jumps(values)
                if len(values) != len(cell_block):
                    raise ValueError("Number of cells does not match "
                                     "parameters in the "
                                     "{} card!".format(card_name))
                else:
                    for i in range(len(cell_block)):
                        # Skip jump values (which have previously had
                        # multiples removed, i.e. 3J converted to J J J)
                        if values[i] != "j":
                            new_cell_block[i] += \
                                " {}={}".format(card_name, values[i])
                case_found = True
                break
        if not case_found:
            new_data_block.append(line)

    return new_cell_block, new_data_block


def get_xs_awtab(block):
    """Extracts the updated atomic weight data from AWTAB and possible
    isotopes appended to an xsdir file with the XSn cards. Since XSn can
    include atomic weights, AWTAB will have priority over XSn data.
    """
    # Our updates to the xsdir listing will be in updates
    xs_updates = OrderedDict()
    awtab_updates = OrderedDict()

    for card in block:
        lower_split_card = card.lower().split(maxsplit=1)
        if lower_split_card[0].startswith("xs"):
            # Then we have an xs card
            # We then have a list of (zaid.abx, awr) for each isotope to be
            # added
            vals = lower_split_card[1].split(maxsplit=2)
            zaid = vals[0]
            awr = num_format(vals[1], 'float')
            # This method will make it so later XSn cards overrule
            # earlier ones
            xs_updates[zaid] = awr
        if lower_split_card[0].startswith("awtab"):
            # Then we have an xs card
            # We then have a list of (zaid.abx, awr) for each isotope to be
            # added
            vals = lower_split_card[1].split()
            for i in range(0, len(vals), 2):
                zaid = vals[i]
                awr = num_format(vals[i + 1], 'float')
                awtab_updates[zaid] = awr

    return xs_updates, awtab_updates


def get_blocks(filename):
    """Given the input file, this method removes comments and splits up
    the model in to the message block, title card, cell card block,
    surface card block, and the data block.

    Parameters
    ----------
    filename : str
        The name of the file to process.

    Returns
    -------
    messages : list of str
        The message block (blank if not present), with each entry in the
        list being a line in this block.
    title : str
        The title card block (one line as required by MCNP, Sec 2.5).
    cells : list of str
        The cell card block, with each entry in the list being a line in
        this block.
    surfaces : list of str
        The surface card block, with each entry in the list being a line
        in this block.
    data : list of str
        The data card block, with each entry in the list being a line in
        this block.
    """

    # Open the file and look for each block, do not yet remove comments
    # as that would confuse the search for blank line delimeters
    with open(filename, "r") as file:
        # We are goin to read the first line and see if it has a message
        # card
        messages = []
        first_line = file.readline().strip()
        if first_line[:len(MESSAGE_KEY)].lower() == MESSAGE_KEY:
            # Then we have a message, store this line and keep storing
            # until we find a blank
            messages.append(first_line.lower())
            for line in file:
                if line.strip():
                    # Then we have a non-blank line, remove any comments
                    # and store it if it comes back with relevant info
                    line = strip_comments(line)
                    if line != "":
                        messages.append(line.lower())
                else:
                    # Then we have a blank and have reached the end of
                    # the message block, dont store the blank but exit
                    # the loop.
                    break
            # Now we know the next line is the title, so lets get it
            title = file.readline().rstrip()
        else:
            # Then this is the title line, so set the title with it
            title = first_line

        # Ok so far we have the message and title, now we want to get
        # the cells, surface, and data blocks
        cells = []
        surfaces = []
        data = []
        num_blanks = 0
        temporary_block = []
        for line in file:
            if line.strip():
                # Then we have a non-blank line, remove any comments
                # and store it if it comes back with relevant info
                line = strip_comments(line)

                if line != "":
                    temporary_block.append(line)
            else:
                # it is a blank
                num_blanks += 1
                # Then we have a blank and have reached the end of
                # the current block, lets store it where it goes and
                # move on
                # The number of blanks so far tells us what block
                # we just read
                if num_blanks == 1:
                    cells = temporary_block[:]
                    temporary_block = []
                elif num_blanks == 2:
                    surfaces = temporary_block[:]
                    temporary_block = []
                if num_blanks > 2:
                    break
        # Our last block, the data block, will not end with a blank line
        # but instead the end of the file, so that means when we exit
        # the above loop, the temporary_block is the data block
        data = temporary_block[:]

        # Now combine multiple lines into one for each block
        messages = combine_lines(messages)
        cells = combine_lines(cells)
        surfaces = combine_lines(surfaces)
        data = combine_lines(data)

        # Remove "repeat" shortcuts
        messages = remove_repeat_shortcut(messages)
        cells = remove_repeat_shortcut(cells)
        surfaces = remove_repeat_shortcut(surfaces)
        data = remove_repeat_shortcut(data)

        # And at the very end, take data cards which can also be cell
        # arguments and move them to the cell cards so the cell cards
        # are described as fully as possible
        cells, data = _add_data_cards_to_cells(cells, data)

        return messages, title, cells, surfaces, data


def convert_density(m_data, density):
    """Converts the density to units of atom/b-cm and all nuclides in
    the material to be fractions of that density.

    Parameters
    ----------
    m_data : List of 3-tuples
        Lists of 3-tuples of (isotope_name, fraction, nlib)
    density : float
        Density of the material in question. A positive density denotes
        units of atoms/b-cm, negative density is units of g/cm^3.

    Returns
    -------
    revised_m_data : OrderedDict
        The dictionary will have the material ids as keys and the
        values are Lists of 3-tuples of (isotope_name, fraction, nlib).
    revised_density : float
        Density of the material in question in units of atom/b-cm.

    """

    # The following is adapted from the
    # openmc.Material.get_nuclide_atom_densities method

    # Get the nuclide names in GND format to easily gather
    # atomic weight data
    nucs = []
    nuc_densities = []
    nuc_density_types = []
    numerator = 0.
    denominator = 0.
    zero_frac_nuc_data = []

    for i in range(len(m_data)):
        nuc, fraction = m_data[i][0:2]
        if fraction > 0.:
            nucs.append(nuc)
            nuc_density_types.append("ao")
            nuc_densities.append(fraction)
            numerator += fraction * adder.data.atomic_mass(nuc)
            denominator += fraction
        elif fraction < 0.:
            nucs.append(nuc)
            nuc_density_types.append("wo")
            nuc_densities.append(-fraction)
            numerator += -fraction
            denominator += -fraction / adder.data.atomic_mass(nuc)
        else:
            zero_frac_nuc_data.append((nuc, m_data[i][-1]))

    average_molar_mass = numerator / denominator

    nucs = np.array(nucs)
    nuc_densities = np.array(nuc_densities)
    nuc_density_types = np.array(nuc_density_types)

    percent_in_atom = np.all(nuc_density_types == 'ao')
    sum_percent = 0.

    # Convert the weight amounts to atomic amounts
    if not percent_in_atom:
        for n, nuc in enumerate(nucs):
            nuc_densities[n] *= average_molar_mass / \
                adder.data.atomic_mass(nuc)

    # Now that we have the atomic amounts, finish calculating densities
    sum_percent = np.sum(nuc_densities)
    nuc_densities = nuc_densities / sum_percent

    # Convert the mass density to an atom density
    if density < 0.:
        revised_density = -density / average_molar_mass * 1.E-24 * AVOGADRO
    else:
        revised_density = density
    nuc_densities = revised_density * nuc_densities

    # Now we must convert the nuclide densities into fractions
    nuc_fractions = nuc_densities / revised_density

    # Finally create a new m_data for returning
    revised_m_data = []
    for n, nuc in enumerate(nucs):
        revised_m_data.append((nuc, nuc_fractions[n], m_data[n][-1]))
    # And add in the zero fraction isotopes in case they have meaning in
    # the future [as an example of meaning, this may be a way to have the user
    # input specific isotopes they wish to have new depletion library xs
    # gathered]
    for nuc, m_data_val in zero_frac_nuc_data:
        revised_m_data.append((nuc, 0.0, m_data_val))

    return revised_m_data, revised_density


def _zam_to_mcnp(Z, A, M, nlib):
    if Z == 95 and A == 242:
        if M == 1:
            zaid = "95242"
        elif M == 0:
            zaid = "95642"
        return zaid + "." + nlib
    zzzaaa = 1000 * Z
    if M > 0:
        zzzaaa += (A + 300) + (M * 100)
    else:
        zzzaaa += A
    if nlib != "":
        zaid = str(zzzaaa) + "." + nlib
    else:
        zaid = str(zzzaaa)
    return zaid


def create_input_file(filename, messages, title, universes, surfaces,
                      materials, transforms, tallies, outputs, sim_settings,
                      others, is_volume_calc=False):
    """Writes the input cards to an MCNP file as named by filename

    Parameters
    ----------
    filename : str
        The file to write to.
    messages : Iterable of str
        The messages portion of the input, one card per item
    title : str
        Title of the input
    universes : dict of Universes
        The universes in the entire model
    surfaces : dict of Surfaces
        The surfaces portion of the input, one card per item
    materials : Iterable of str
        The materials portion of the input, one card per item
    transforms : dict of CoordTransform
        The coordinate transforms in the model
    tallies : Iterable of str
        The tally portion of the input, one card per item
    outputs : Iterable of str
        The output control portion of the input, one card per item
    sim_settings : SimSettings
        The simulation settings to write
    others : Iterable of str
        The remaining portion of the input, one card per item
    is_volume_calc : bool, optional
        If this is for a volume calculation, we need 0 importance cells
        to have a non-zerp importance so that our randomly chosen
        samples, if outside the geometry, do not cause MCNP to quit.
        We also want to not print the volume so that we  can see if MCNP
        computes an analytic value to compare with. Defaults to False

    Returns
    -------
    file_string : str
        The contents of the file
    """

    model_univs = set(universes[ROOT_UNIV].nested_universe_ids)
    model_cells = universes[ROOT_UNIV].nested_cells

    # Create a long string to just dump
    file_string = ""

    # Process messages
    if messages:
        for message in messages:
            file_string += card_format(message)
        file_string += "\n"

    # Process title
    file_string += title + "\n"

    # Print a table with the universe names and ids
    if len(model_univs) > 0:
        file_string += "{} ADDER UNIVERSE IDENTIFICATION ".format(NEWLINE_COMMENT)
        file_string += "TABLE\n"
        file_string += "{} {:^11s} | {:<13s}\n".format(NEWLINE_COMMENT,
                                                       "Universe ID",
                                                       "Universe Name")
        file_string += "{} ".format(NEWLINE_COMMENT) + "-" * (12 + 14) + "\n"
        for u_id in model_univs:
            u = universes[u_id]
            file_string += "{} {:^11} | {:<13s}\n".format(NEWLINE_COMMENT,
                                                          u_id, u.name)

    # Process cells
    transforms_to_print = {}
    # Print the cells in order of cell id
    for cell_id in sorted(model_cells.keys()):
        file_string += card_format(model_cells[cell_id].to_str(is_volume_calc))
        cell_transforms = model_cells[cell_id].get_transforms()
        for transform in cell_transforms:
            transforms_to_print[transform.id] = transform
    file_string += "\n"

    # Process surfaces
    for surface in surfaces.values():
        file_string += card_format(str(surface))
        if surface.coord_transform is not None:
            transforms_to_print[surface.coord_transform.id] = \
                surface.coord_transform
    file_string += "\n"

    # Process data cards
    for material in materials:
        file_string += card_format(material)

    for t_id in sorted(transforms_to_print.keys()):
        file_string += card_format(str(transforms_to_print[t_id]))
        # Update global transform information as a backstop
        if t_id not in transforms:
            transforms[t_id] = transforms_to_print[t_id]

    for output in outputs:
        file_string += card_format(output)

    if not is_volume_calc:
        for card in sim_settings.all_cards:
            file_string += card_format(card)

    for other in others:
        file_string += card_format(other)

    if tallies:
        for tally in tallies:
            file_string += card_format(tally)
    file_string += "\n"

    with open(filename, "w") as file:
        file.write(file_string)

    return file_string


def update_materials(universes, allowed_isotopes, materials,
                     multiplier_mat_ids, reactivity_threshold):
    """Creates revised material cards for MCNP input

    Parameters
    ----------
    universes: OrderedDict of Universes
        Universe information to retrieve materials from.
    allowed_isotopes : OrderedDict
        The keys are the MCNP nuclide and library (e.g., 92235.71c) and
        the values are the atomic weight ratios for that nuclide in that
        library.
    materials : List of Material
        All the materials in the model, provided for multiplier mats
    multiplier_mat_ids : Set of int
        The material ids that are needed for tallies; these will always
        be printed.
    reactivity_threshold : float
        The threshold to apply when determining whether or not to
        incorporate an isotope in the neutronics model

    Returns
    -------
    cards : Iterable of str
        Each material and material-thermal card to write to the input

    """

    cards = []

    # For user-readability, we would like the materials printed in order
    # of increasing mat id so sort in this way; this is why we have the
    # sorted command
    model_mats = universes[ROOT_UNIV].nested_materials
    for mat_id in sorted(model_mats.keys()):
        mat = model_mats[mat_id]
        # Create the material card
        # Append with a comment containing the adder name
        data = "{} ADDER Material Name: {}".format(NEWLINE_COMMENT, mat.name)
        cards.append(data)
        data = "m{}".format(mat_id)
        zaids_to_write = []
        fracs_to_write = []
        for iso, frac, to_print in zip(mat.isotopes, mat.atom_fractions,
                                       mat.isotopes_in_neutronics):
            zaid = _zam_to_mcnp(iso.Z, iso.A, iso.M, iso.xs_library)
            if to_print and frac > 0.:
                zaids_to_write.append(zaid)
                fracs_to_write.append(frac)

        # Normalize fraction
        fracs_to_write = np.array(fracs_to_write)
        fracs_to_write /= np.sum(fracs_to_write)

        # And now we can write it
        for z, f in zip(zaids_to_write, fracs_to_write):
            data += " {} {:8E}".format(z, f)
        cards.append(data)

        if mat.thermal_xs_libraries:
            # Create the material-thermal card
            thermal_data = "mt{}".format(mat_id)
            for thermal_xs_library in mat.thermal_xs_libraries:
                thermal_data += " " + thermal_xs_library
            cards.append(thermal_data)

    # Now include the materials not printed above
    multiplier_mats = multiplier_mat_ids - set(model_mats.keys())
    if multiplier_mats:
        mat_by_id = dict()
        for i in range(len(materials)):
            # Ugh this is wasteful.
            mat_by_id[materials[i].id] = materials[i]

    for mat_id in sorted(multiplier_mats):
        mat = mat_by_id[mat_id]
        # Create the material card
        # Append with a comment containing the adder name
        data = "{} Multiplier Material Name: {}".format(NEWLINE_COMMENT,
                                                        mat.name)
        cards.append(data)
        data = "m{}".format(mat.id)
        zaids_to_write = []
        fracs_to_write = []
        for iso, frac, to_print in zip(mat.isotopes, mat.atom_fractions,
                                       mat.isotopes_in_neutronics):
            zaid = _zam_to_mcnp(iso.Z, iso.A, iso.M, iso.xs_library)
            if zaid in allowed_isotopes and to_print and frac > 0.:
                zaids_to_write.append(zaid)
                fracs_to_write.append(frac)

        # Normalize fraction
        fracs_to_write = np.array(fracs_to_write)
        fracs_to_write /= np.sum(fracs_to_write)

        # And now we can write it
        for z, f in zip(zaids_to_write, fracs_to_write):
            data += " {} {:8E}".format(z, f)
        cards.append(data)

        if mat.thermal_xs_libraries:
            # Create the material-thermal card
            thermal_data = "mt{}".format(mat.id)
            for thermal_xs_library in mat.thermal_xs_libraries:
                thermal_data += " " + thermal_xs_library
            cards.append(thermal_data)

    return cards


def create_material_flux_tallies(universes, depl_libs):
    """Creates MCNP flux tallies to get the flux for each material

    Parameters
    ----------
    universes: OrderedDict of Universes
        Universe information to retrieve materials from.
    depl_libs : OrderedDict of DepletionLibrary
        The depletion libraries in use in the model

    Returns
    -------
    cards : Iterable of str
        Each tally card to write to the input.

    """

    cards = []
    t_id = ADDER_TALLY_ID + 4
    f4 = "F{}:N".format(t_id)

    # Add in the cell id for every cell with a depleting material
    model_cells = universes[ROOT_UNIV].nested_cells
    Ebounds = None
    for cell_id in sorted(model_cells.keys()):
        cell = model_cells[cell_id]
        if cell.material is not None:
            mat = cell.material
            if mat.is_depleting:
                # Get the energy bounds while we are here
                if Ebounds is None:
                    Ebounds = \
                        depl_libs[mat.depl_lib_name].neutron_group_structure
                # Then we want to add a tally for this cell
                f4 += " {}".format(cell_id)
    cards.append(f4)

    # Add in the energy group structure
    e4 = "E{}".format(t_id)
    for energy in Ebounds[1:]:
        e4 += " {}".format(energy)
    cards.append(e4)
    cards.append("FC{} Flux tally for depletion".format(t_id))

    return cards


def create_rxn_rate_tallies(universes, allowed_isotopes, isos_and_rxn_keys,
    depl_libs):
    """Establishes the tallies we need to obtain rxn rate information for
    each isotope in each material that isotope exists.

    Parameters
    ----------
    universes: OrderedDict of Universes
        Universe information to retrieve materials from
    allowed_isotopes : OrderedDict
        The keys are the MCNP nuclide and library (e.g., 92235.71c) and
        the values are the atomic weight ratios for that nuclide in that
        library.
    isos_and_rxn_keys : OrderedDict or None
        The isotopes in the library and their corresponding reaction keys and
        MCNP tally strings for obtaining them.
    depl_libs : OrderedDict of DepletionLibrary
        The depletion libraries in use in the model

    Returns
    -------
    tally_cards : Iterable of str
        Each tally card to write to the input.
    tally_map : dict
        Dict providing the tally id as a key, and the value is a tuple of:
        (list of cell ids, and a list of
        (iso_name, [rxn_type, multiplier bin_ids]))
    isos_and_rxn_keys : OrderedDict or None
        The isotopes in the library and their corresponding reaction keys and
        MCNP tally strings for obtaining them.
    """

    # This method will create a list of MCNP tallies, tally multipliers,
    # and the pseudo-materials with the isotopes for the tally mutipliers.
    # These tallies will be grouped so there is one tally for each set of
    # isotopes that exists in the same cells. This way we do not over-use our
    # available # of tallies (9,999) while also not over-tallying info (by
    # using a single tally with every isotope, rxn rate, and cell)

    model_cells = universes[ROOT_UNIV].nested_cells

    # In the following loop we will determine the reactions we need to
    # tally from MCNP as well as gather the list of *all* isotopes for
    # which data will need to be obtained.
    # When doing so, we implicitly assume that all material's depletion
    # libraries have the same xs channels. This is fine for now, but, if the
    # code is expanded in the future to allow differing chains/libs in
    # different regions, then this will be incorrect
    # TODO, we should only need to do this once.
    if isos_and_rxn_keys is None:
        # key is iso name, value is list of the rxn keys that we
        isos_and_rxn_keys = OrderedDict()

    # Get the base depletion lib
    depl_lib = depl_libs[BASE_LIB]
    for i, iso_name in enumerate(depl_lib.isotopes):
        xs_lib = depl_lib.isotopes[iso_name].neutron_xs

        # If there is no xs_lib, then there is no xs in the chain, so nothing
        # to get from MCNP, skip
        if xs_lib is None:
            continue
        rxn_keys = []
        rxn_strs = []
        for key in xs_lib.keys():
            if key in MCNP_RXN_TALLY_IDS:
                rxn_keys.append(key)
                rxn_strs.append(
                    "(1. {} " + "{})".format(MCNP_RXN_TALLY_IDS[key]))
        if rxn_keys:
            isos_and_rxn_keys[iso_name] = (rxn_keys, rxn_strs)

    # To create the pseudo-mats, we need to factor in the isotopic libraries,
    # and that we can only assume comes from the materials.
    # So we will go through and 1) create pseudo material dictionaries keyed
    # by the mcnp zaid name (w/ lib attached), 2) get the maximum material id,
    # 3) keep track of what cells each zaid is in
    max_mat_id = 0
    pmats_by_zaid = {}
    cells_by_zaid = {}
    for cell_id in sorted(model_cells.keys()):
        # Get the corresponding material, if there is one
        cell = model_cells[cell_id]
        mat = cell.material
        if mat is None:
            continue

        # Update the maximum material id before we filter on depleting or not
        if mat.id > max_mat_id:
            max_mat_id = mat.id

        # Now only act on those materials we need to deplete
        if mat.is_depleting and not mat.is_default_depletion_library:
            for i, iso in enumerate(mat.isotopes):
                # Skip any isotopes that are not depleting, not in the neut
                # model, and not in the library
                if not iso.is_depleting or not mat.isotopes_in_neutronics[i]:
                    continue

                if iso.name not in isos_and_rxn_keys:
                    # Then its not in the depletion lib and/or has no neut xs
                    continue

                # Skip any isotopes that are not allowed isotopes
                zaid = _zam_to_mcnp(iso.Z, iso.A, iso.M, iso.xs_library)
                if zaid not in allowed_isotopes:
                    continue

                # Ok, now we can store the pseudo mat key if we havent yet
                if zaid not in pmats_by_zaid:
                    # (note we leave the mat id blank for now until we have
                    # counted them all)
                    pmats_by_zaid[zaid] = "m{} " + "{} 1.".format(zaid)
                # Update the listing of which zaids are in what cells
                if zaid not in cells_by_zaid:
                    cells_by_zaid[zaid] = [cell_id]
                else:
                    cells_by_zaid[zaid].append(cell_id)

    # Now that we know all the actual zaids and all their locations, lets
    # filter down to sets of zaids that share common cells (and keep those cells)
    # We will also be updating the pmats card with the actual pseudo-mat now
    # that we have the maximum used material id
    unique_cells = {}
    zaid_cell_sets = {}
    pmat_card_ids_by_zaid = {}
    tally_cards = []
    m = max_mat_id + 1
    for i, (zaid, cell_set) in enumerate(cells_by_zaid.items()):
        # Convert to tuple so it can be hashed
        zaid_cell_sets[zaid] = tuple(cell_set)
        # Check for uniqueness of the set
        if zaid_cell_sets[zaid] in unique_cells:
            # We have it already, so add the zaid to the list
            unique_cells[zaid_cell_sets[zaid]].append(zaid)
        else:
            # We dont have it, so create a new entry
            unique_cells[zaid_cell_sets[zaid]] = [zaid]

        # Update the pseudo-mats cards with the mat id
        pmat_card_ids_by_zaid[zaid] = m + i
        tally_cards.append(pmats_by_zaid[zaid].format(m + i))

    # Now we have the pseudo-mat cards, the rxn bin cards, and the cell filter
    # sets, we can now create our tallies
    # While doing that we store a tally map that has the tally id as a key,
    # and the values are a tuple of the (cell ids, and a list of the
    # [(zaid, [(rxn_type, multiplier bin_id)])]
    tally_map = {}
    # Get the pmat cards
    tally_cards = [pmats_by_zaid[zaid].format(m)
        for zaid, m in pmat_card_ids_by_zaid.items()]
    t_id = ADDER_TALLY_ID + 4
    # First get a blank card for e group structure
    e_grid_str = ""
    for energy in depl_lib.neutron_group_structure[1:]:
        e_grid_str += " {:.8E}".format(energy)
    # Now go on to making the cards and map
    for cell_set, zaid_set in unique_cells.items():
        t_id -= 10 # Take away 10 so we always have a tally id that ends in 4

        # Create the tally card with the cell bins
        f4 = "F{}:N".format(t_id)
        for cell_id in cell_set:
            f4 += " {}".format(cell_id)
        tally_cards.append(f4)

        # Create the energy grid
        tally_cards.append("E{}".format(t_id) + e_grid_str)

        # Now store the tally multiplier bins for each zaid
        fm4 = "FM{}:N".format(t_id)
        tally_map_mult_bins = []
        bin_id = 0
        for zaid in zaid_set:
            # Going to need to convert the zaid to an isotope name
            iso_name = adder.data.iso_name_from_zaid(zaid)
            # Now we can get the tally bins themselves
            bin_ids = []
            rxn_types, rxn_bins = isos_and_rxn_keys[iso_name]
            for rxn_type, rxn_bin in zip(rxn_types, rxn_bins):
                bin_set = rxn_bin.format(pmat_card_ids_by_zaid[zaid])
                fm4 += " " + bin_set
                bin_ids.append((rxn_type, bin_id))
                bin_id += 1
            tally_map_mult_bins.append((iso_name, bin_ids))
        tally_map[t_id] = (cell_set, tally_map_mult_bins)
        tally_cards.append(fm4)

    # So now we have the pseudo mat cards printed, the tally energy cards
    # printed, and the tally cards. We also have the map of where to find
    # the tally data as well.

    # At this stage we are set to return our results
    return tally_cards, tally_map, isos_and_rxn_keys


def create_output(user_output=None):
    """Creates the output control cards that we rely on to get
    information from MCNP

    Parameters
    ----------
    user_output : None or str, optional
        Either None (indicating there is no user output to include), or
        a string of the user output that should be incorporated with
        ADDER's output needs.

    Returns
    -------
    cards : Iterable of str
        The output cards to write to the input.

    """

    cards = []

    if user_output is None:
        # Set our required tables
        card = "PRINT "
        for table_num in PRINT_TABLES:
            card += " {}".format(table_num)
        cards.append(card)

        # Ensure there is a print and dump cycle card to get an mctal file
        cards.append("PRDMP 0 0 1")
    else:
        # Find the user's PRINT command and PRDMP command
        split_print_card = [""]
        split_prdmp_card = [""]
        other_cards = []
        for card in user_output:
            stripped_card = card.split()
            if stripped_card[0].lower() == "print":
                split_print_card = stripped_card
            elif stripped_card[0].lower() == "prdmp":
                split_prdmp_card = stripped_card
            else:
                other_cards.append(card)

        # Now pull the user's options
        if split_print_card != [""]:
            user_print_options = [num_format(i, 'int') for i in
                                  split_print_card[1:]]
        else:
            user_print_options = []

        if split_prdmp_card != [""]:
            user_prdmp_options = [i if i == 'j' else num_format(i, 'int') for i
                                  in expand_jumps(split_prdmp_card[1:])]
        else:
            user_prdmp_options = []

        # Create a set of Adder and user print options together
        all_print_options = set(user_print_options + PRINT_TABLES)

        # The only prdmp option we care about is the MCTAL option; so
        # force it (the third entry) to be a 1
        all_prdmp_options = user_prdmp_options[:]
        # Make sure we have at least enough prdmp options
        for i in range(len(user_prdmp_options), 3):
            all_prdmp_options.append(0)

        # Now ensure the third position is a 1
        all_prdmp_options[2] = 1

        # Set our tables
        card = "PRINT "
        for table_num in sorted(all_print_options):
            card += " {}".format(table_num)
        cards.append(card)

        card = "PRDMP "
        for option in all_prdmp_options:
            card += " {}".format(option)
        cards.append(card)

        # And the remainder
        cards.extend(other_cards)

    return cards


def _analyze_fm_card(line):
    """Analyzes an FM card to extract the material ids used for the
    multiplication"""
    mat_ids = set()

    # FM card can have one or multiple (multiplier sets) and
    # (attenuator sets). If there is only one of these sets, then the
    # parentheses are not required

    # To clean up the analysis, we will (1) ensure the card is lowercase
    # (2) remove any T or C present at the end, (3) strip off the fm#
    data = line.lower().rstrip(" tc").split(maxsplit=1)[1]

    # Now aggregate into bin sets, which are denoted by (...)
    # Note that if no parentheses are present, there is only one bin set
    if data.startswith("("):
        # Then there is at least one "parenthesed" bin set
        bin_sets = re.findall(r'\(([^)]+)', data)
        # The above seems to leave left parentheses; remove
        for i in range(len(bin_sets)):
            if bin_sets[i].startswith("("):
                bin_sets[i].lstrip("(")

    else:
        # Only one bin set, just store the whole data as one bin set
        bin_sets = [data]

    # Now we need to extract the material id from each bin set
    for i in range(len(bin_sets)):
        # Pull off the first two values from the bin data
        bin_data = bin_sets[i].split(maxsplit=2)
        # The only type of bin set that does not have a material id is
        # the special multiplier set: "c k". This also happens to be the
        # only bin set type that has < 3 values and so we can use that
        # to discriminate (this is also the reason we use maxsplit=2)
        if len(bin_data) > 2:
            # Then this has a material, store its id
            mat_ids.add(num_format(bin_data[1], 'int'))

    return mat_ids
