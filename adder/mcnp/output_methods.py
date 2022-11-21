import os
from collections import OrderedDict

import numpy as np

import mcnptools

from .constants import *
from .input_utils import num_format
from adder.type_checker import *


def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def _init_mctal(mctal_filename):
    """Initializes the MCTAL file and gets keff.

    Parameters
    ----------
    mctal_filename : str
        Path/filename of the MCTAL file

    Returns
    -------
    mctal : mcnptools.Mctal
        The MCTAL file interface object
    keff : float
        Calculated k-eigenvalue of the problem.
    keff_stddev : float
        Estimated k-eigenvalue standard deviation.
    """

    check_type("mctal_filename", mctal_filename, str)
    if os.path.exists(mctal_filename):
        mctal = mcnptools.Mctal(mctal_filename)
    else:
        msg = "MCTAL filename {} does not exist".format(mctal_filename)
        raise FileNotFoundError(msg)

    kcode = mctal.GetKcode()
    keff = kcode.GetValue(mcnptools.MctalKcode.AVG_COMBINED_KEFF)
    keff_stddev = \
        kcode.GetValue(mcnptools.MctalKcode.AVG_COMBINED_KEFF_STD)

    return mctal, keff, keff_stddev


def _extract_results(mctal, tally_ids):
    """Gets tally information for the tallies defined in tally_ids
    from the provided MCTAL file.

    Parameters
    ----------
    mctal : mcnptools.Mctal
        The MCTAL file interface object
    tally_ids : Iterable of str
        List of the tally ids to extract the information from.

    Returns
    -------
    tallies : Iterable of Mctal
        The requested tallies
    """

    tallies = [mctal.GetTally(i) for i in tally_ids]

    return tallies


def _get_table_num(line):
    """Determines the table we are in if this is a header, otherwise
    returns None"""

    table = None

    # We are going to split the line from the right and then we should
    # expect the split parts to be ["print", "table", TABLENUM]
    split_line = line.rsplit(maxsplit=3)
    if len(split_line) >= 3:
        # Then we have the 3 words we are looking for
        # that means this could be a hit
        word1, word2, word3 = split_line[-3:]
        if word1.lower() == "print" and word2.lower() == "table":
            # Then we have a match!
            # The last one should be a digit make sure
            if word3.isdigit():
                table = num_format(word3, 'int')
    return table


def _get_table130_nu(file, keff):
    """Finds and reads the Table 130 data from an MCNP output file.
    The file object will be left at the next non-blank line after this
    table.

    Parameters
    ----------
    file : File
        The output file with the current position being right after the
        the start of table 60.
    keff : float
        The calculated effective neutron multiplication factor.

    Returns
    -------
    nu : float
        The number of neutrons produced per fission
    """

    line = file.readline()
    src_weight = None
    fission_loss_weight = None
    while line:
        # there will be a line that starts with "cell number" and ends
        # with "total", that is the one we want and that tells us it is
        # time to start getting data
        split_line = line.split()
        if len(split_line) >= 3:
            if ((split_line[0] == "cell" and split_line[1] == "number" and
                 split_line[-1] == "total")):
                # Then we have entered into the region with the total
                # Step through lines until we get the source line
                for line in file:
                    if not line.isspace():
                        split_line = line.split()
                        if split_line[0] == "source":
                            src_weight = num_format(split_line[-1], 'float')
                            break

                # the next piece of data we need is the loss to fission
                for line in file:
                    if not line.isspace():
                        if line.strip().startswith("loss to fission"):
                            split_line = line.split()
                            fission_loss_weight = -num_format(split_line[-1],
                                                              'float')
                            break

                # Now lets get to the final total line, there should be
                # two
                num_total = 0
                while num_total < 2:
                    line = file.readline()
                    if not line.isspace():
                        if line.strip().startswith("total"):
                            num_total += 1

        # Advance to the next line
        line = file.readline()

    # Ok now we just need to compute nu
    if src_weight is None and fission_loss_weight is None:
        raise ValueError("Did not find source and loss to fission weights!")
    nu = keff * src_weight / fission_loss_weight
    return nu


def _get_table60_volumes(file):
    """Finds and reads the Table 60 data from an MCNP output file.
    The file object will be left at the next non-blank line after this
    table.

    Parameters
    ----------
    file : File
        The output file with the current position being right after the
        the start of table 60.

    Returns
    -------
    volumes : OrderedDict
        The cell number is the key and the value is the volume in cm^3.
    """

    # We enter here right after the "print table 60" line of the output
    # we should expect a blank, two lines of table labels, and then
    # another blank and a line with the total volume and mass and then
    # another blank line.

    # The line we want will start with the number 1
    cell_lines = []
    for line in file:
        if not line.isspace():
            split_line = line.split(maxsplit=8)
            if split_line[0].isdigit():
                # Got a cell
                cell_lines.append(split_line)
            elif split_line[0].lower() == "total":
                # Then the next line is the end so break here and then
                # immediately read the line
                break
    # discard that last blank line
    file.readline()

    # Ok now we can operate on the cell data
    volumes = OrderedDict()
    for split_line in cell_lines:
        cell_id, mat_id, a_dens, g_dens, vol, mass, pieces, imp = \
            split_line[1:]
        volumes[num_format(cell_id, 'int')] = num_format(vol, 'float')

    return volumes


def _get_table126_scores(lines):
    """Finds and reads the table 126 data from an MCNP output file.
    The file object will be left at the next non-blank line after this
    table.

    Parameters
    ----------
    lines : Iterable of str
        The lines from the output file with the current position being
        right after the the start of table 126.

    Returns
    -------
    hit_fraction : OrderedDict
        The cell number is the key and the value is the fraction of hits
        in that cell.
    """

    # We enter here right after the "print table 126" line of the output
    # we should expect a blank, three lines of table labels, and then
    # another blank and then our table of interest, ending in a line
    # that starts with "total".

    # The line we want will start with the number 1
    # Discard the blank, 3 headers, and blank
    lines = lines[1:]

    # Now get the table data
    cell_ids = []
    cell_hits = []
    for line in lines:
        if not line.isspace():
            split_line = line.split()
            if split_line[0].isdigit():
                # Got a cell
                cell_ids.append(num_format(split_line[1], 'int'))
                cell_hits.append(num_format(split_line[3], 'int'))
            elif split_line[0].lower() == "total":
                # Then the next line is the end so break here
                num_hits = num_format(split_line[2], 'int')
                break

    # Now we can create the return dictionary while normalizing the data
    hits_by_cell = {k: v for k, v in zip(cell_ids, cell_hits)}

    return hits_by_cell, num_hits


def _get_material_wise_flux(mctal, universes):
    """Gets flux from the material-wise flux tallies created in the
    write_input.create_material_wise_flux_tallies() method.

    Parameters
    ----------
    mctal : mcnptools.Mctal
        The MCTAL file interface object
    universes: OrderedDict of Universes
        Universe information to retrieve materials from.

    Returns
    -------
    keff : float
        Calculated k-eigenvalue of the problem.
    keff_uncertainty : float
        Estimated k-eigenvalue uncertainty.
    flux : OrderedDict of np.ndarray
        Dictionary where the key is the material id and the value is the
        group-wise tallied flux mean.
    """

    tally_ids = [ADDER_TALLY_ID + 4]
    tally = _extract_results(mctal, tally_ids)[0]

    # Get the attributes of our results
    bins = tally.GetFBins()
    energy_struct = tally.GetEBins()
    num_groups = len(energy_struct)

    # Now create our results arrays with this information
    mctal_flux = np.zeros((len(bins), num_groups))

    # Get the results from the tally fluctuation-chart
    tfc = tally.TFC
    for b in range(len(bins)):
        for e in range(len(energy_struct)):
            mctal_flux[b, e] = \
                tally.GetValue(b, tfc, tfc, tfc, tfc, tfc, e, tfc)

    # Now we need to convert from cell bin ordering to mat_id ordering
    mat_ids = []
    model_cells = universes[ROOT_UNIV].nested_cells
    for cell_id in sorted(model_cells.keys()):
        cell = model_cells[cell_id]
        if cell.material is not None:
            if cell.material.is_depleting:
                mat_ids.append(cell.material_id)

    # The above is now the ordering that we had
    flux = OrderedDict.fromkeys(mat_ids)
    for i, mat_id in enumerate(mat_ids):
        flux[mat_id] = mctal_flux[i, :]

    return flux


def _get_material_wise_rxn_rates(mctal, nested_cells, flux, tally_map, depl_libs):
    """Gets the reaction rate tallies from the MCTAL file and converts to
    multigroup xs

    Parameters
    ----------
    mctal : mcnptools.Mctal
        The MCTAL file interface object
    nested_cells : OrderedDict
        All nested cells contained in this universe; indexed by cell id, and
        a value of the Cell
    flux : OrderedDict of np.ndarray
        Dictionary where the key is the material id and the value is the
        group-wise tallied flux mean.
    tally_map : dict
        Dict providing the tally id as a key, and the value is a tuple of:
        (list of cell ids, and a list of
        (iso_name, [rxn_type, multiplier bin_ids]))
    depl_libs : OrderedDict of DepletionLibrary
        The depletion libraries in use in the model

    """

    for t_id, tmap_vals in tally_map.items():
        # Extract relevant tally map info
        cell_set, bin_map = tmap_vals

        # Get the tally info
        tally = _extract_results(mctal, [t_id])[0]
        cell_bins = tally.GetFBins()
        tfc = tally.TFC

        # Now process each cell
        for cell_id in cell_set:
            # Get some shorthand objects
            cell = nested_cells[cell_id]
            mat = cell.material   # Assume mat is depleting and is a real mat
            depl_lib = depl_libs[mat.depl_lib_name]
            cell_flux = flux[mat.id]
            cell_index = cell_bins.index(num_format(cell_id, 'float'))

            # Now process each bin of the tally
            for iso_name, bins in bin_map:
                iso_xs = depl_lib.isotopes[iso_name].neutron_xs
                for rxn_type, rxn_bin_index in bins:
                    xs = np.zeros(depl_lib.num_neutron_groups)
                    for g in range(depl_lib.num_neutron_groups):
                        xs[g] = tally.GetValue(cell_index, tfc, tfc, tfc,
                                               rxn_bin_index, tfc, g, tfc)
                    # Divide by the flux to yield xs from rxn rate
                    xs /= cell_flux

                    # Incorporate into the library
                    # We will assume the reaction type already
                    # exists and thus no checking is necessary
                    # Get the reaction tuple
                    _, targets_, yields_, q_value = \
                        iso_xs._products[rxn_type]
                    # Now re-build the tuple with the new yield
                    iso_xs._products[rxn_type] = \
                        (xs, targets_, yields_, q_value)
