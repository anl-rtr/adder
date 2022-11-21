from collections import OrderedDict
from .input_utils import num_format


def parse_xsdir(filename, xs_updates, awtab_updates):
    """Parses the model's xsdir file to provide a list of allowed
    isotope/library combinations and their atomic weight ratios.

    Parameters
    ----------
    filename : str
        The xsdir file
    xs_updates : dict of str: float
        The dictionary of the zaid.abx as keys and awr as values coming from
        the XSn card of the input. These should be updating what is in the
        xsdir file.
    awtab_updates: dict of str: float
        The dictionary of the zaid.ab as keys and awr as values coming from
        the AWTAB cards of the input. These should be updating all instances of
        zaid.ab in the xsdir listing

    Returns
    -------
    allowed_isotopes : OrderedDict
        The keys are the MCNP nuclide and library (e.g., 92235.71c) and
        the values are the atomic weight ratios for that nuclide in that
        library.

    """

    allowed_isotopes = OrderedDict()

    # Find 'directory' section
    lines = open(filename, 'r').readlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == 'directory':
            break
    else:
        raise IOError("Could not find 'directory' section in MCNP xsdir file")

    # Handle continuation lines indicated by '+' at end of line
    lines = lines[index + 1:]
    continue_lines = [i for i, line in enumerate(lines)
                      if line.strip().endswith('+')]
    for i in reversed(continue_lines):
        lines[i] += lines[i].strip()[:-1] + lines.pop(i + 1)

    # Create list of ACE libraries
    for line in lines:
        words = line.split()
        if len(words) < 3:
            continue
        elif words[0][0] == '#':
            # Ignore comments
            continue

        # The first word is the zaid.lib, the second is the
        # atomic-weight-ratio (AWR), and the remainder are unnecessary
        # for our purposes
        # So we can just store these results and move on
        allowed_isotopes[words[0]] = num_format(words[1], 'float')

    # Now include the xs_updates
    for zaid_abx, new_awr in xs_updates.items():
        # This will over-rule anything in the xsdir information if conflicts
        allowed_isotopes[zaid_abx] = new_awr

    # Finally, the awtab_updates
    for zaid_ab, new_awr in awtab_updates.items():
        # Find which keys in allowed_isotopes have the same zaid.ab
        for zaid_abx in allowed_isotopes.keys():
            if zaid_abx.startswith(zaid_ab):
                # Then we have a value to update
                allowed_isotopes[zaid_abx] = new_awr

    return allowed_isotopes
