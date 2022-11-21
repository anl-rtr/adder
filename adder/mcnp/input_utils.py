from collections import OrderedDict
import textwrap
import re

from .constants import *


def find_invalid_cards(line):
    # raises an error msg if an invalid input card is obtained
    stripped_line = line.strip()
    for error_card in ERROR_CARDS:
        if stripped_line.startswith(error_card):
            msg = "The {} card is not supported!".format(error_card)
            raise ValueError(msg)
    # Next we need to check for vertical input format delineation
    # The character to look for is the VERTICAL delineator, however, this can
    # conflict with the same character (#) that is a particle type designator.
    # So, to make sure we only get the VERTICAL delineator, we look for that
    # character with only spaces to the left
    if stripped_line.startswith(VERTICAL):
        msg = "The vertical input format card is not supported!"
        raise ValueError(msg)


def split_data_and_keywords(input_params, allowed_keywords):
    # Splitting the card into just data and just parameters is slightly
    # complex since the keywords could have "=" or spaces, and the
    # data for each keyword too can have spaces between entries.

    # Only work with lower-case card info
    params = input_params.lower()

    # So to have the simplest implementation of this, we will do this work
    # in four steps.
    # First, we find the indices of all the keywords
    # Second, we split apart the params and separate data from kw-values
    # Third, we split the keyword/value pairs in to our dictionary that is
    # to be returned

    # First, get the indices of all keywords, while doing, store all indices
    keys_and_indices = {}
    all_indices = []
    for key in allowed_keywords:
        if key.startswith("*"):
            # Regex doesnt like * as it conflicts with the its own *
            # replace with an escape
            regex_key = "\\" + key
        else:
            regex_key = key
        try:
            matches = [match.start() for match in re.finditer(regex_key, params)]
        except re.error:
            # Then we dont have this key, so just store None
            matches = None
        if matches:
            # Check
            if key in ["trcl", "fill"]:
                # trcl and fill can have an asterisk before. Dont get an index
                # for the key without an asterisk if the value has an asterisk
                to_pop = []
                for i, idx in enumerate(matches):
                    # Then look one value before to see if it is an asterisk
                    if idx >= 1 and params[idx - 1] == "*":
                        # Welp, lets not double count. Remove this.
                        to_pop.append(i)
            # The 'u' keyword conflicts with others, lets rule that case out
            elif key == "u":
                to_pop = []
                for i, idx in enumerate(matches):
                    # See if it has an n or : before the u (nonu or particle u)
                    if idx >= 1 and params[idx - 1] in ["n", ":"]:
                        # Remove this match then as the u was actually nonu or
                        # a particle designator
                        to_pop.append(i)
                    elif idx < len(params) - 1 and params[idx + 1] == "n":
                        # See if it has an n after the u (unc)
                        to_pop.append(i)
            else:
                to_pop = []
            # Now pop the entries (starting from the end so the indices
            # do not need updating)
            for i in reversed(to_pop):
                matches.pop(i)
            keys_and_indices[key] = matches
            all_indices.extend(matches)
    all_indices.sort()

    # Second split the params across all keyword positions and extract the
    # data part
    split_indices = [0] + all_indices + [len(params)]
    split_string = [params[i: j].strip()
                    for i, j in zip(split_indices, split_indices[1:])]
    data = split_string[0]
    if len(split_string) > 1:
        kw_info = split_string[1:]
    else:
        # Just return so we dont need size-blocks for this
        return data, {}

    # Third, split the key-value pairs and store in keywords
    keywords = {}
    for key_values in kw_info:
        # Now we either have an equal sign or spaces. The most likely is = so
        # try that first, and then if that fails move on to spaces
        splits = key_values.split("=")
        if len(splits) == 1:
            # Then we had no equal, we split on space(s)
            splits = key_values.split(maxsplit=1)
        key = splits[0]
        value = splits[1]
        keywords[key] = value
    return data, keywords


def remove_repeat_shortcut(block):
    """Given a line, this replaces the *repeat* shortcuts as defined in
    section 2.8.1 of the MCNP 6.2 users manual.

    This function is assumed to be called after strip_comments and
    combine_lines.

    Parameters
    ----------
    block : List of str
        The block where each entry in the list is a line from the file
        that has already has comments, line-returns, and line
        continuations removed.

    Returns
    -------
    new_block : str
        The block with repeat shortcuts removed.
    """

    # Since we dont really know when given a card name or its parameters
    # we will look for a "R" (or "r") immediately after a digit without
    # any spaces. A review of Chapter 7 indicates this should yield no
    # conflicts with non-repeated inputs

    new_block = []
    for line in block:
        # First split the line by spaces
        split_line = line.split()

        # First off, do not do this for material/mat-thermal cards;
        # it wastes time (you cant repeat alternating arrays of
        # zaids/fractions), but it is theoretically possible for a zaid
        # to have a library identifier that ends in "#r", so we should
        # just not bother.
        # check material; first make sure we have enough chars to check
        if len(split_line[0]) > 2:
            if split_line[0] == "m" and split_line[1].isdigit():
                # Then this is a material card, store and move on
                # to the next line
                new_block.append(line)
                continue
        # Check mt; first make sure we have enough chars to check
        if len(split_line[0]) > 3:
            if split_line[:2] == "mt" and split_line[2].isdigit():
                # Then this is an mt card, store and move on
                # to the next line
                new_block.append(line)
                continue

        # Now step through each entry in the line and, if it is a repeat
        # then replace with the repeated information
        # We start with the second entry since the 1st has to be the one
        # to repeat
        last_item = split_line[0]
        new_line = last_item
        for item in split_line[1:]:
            if item[-1] in ["r", "R"] and item[:-1].isdigit():
                # Then we repeat the last one item[:-1] times
                n_repeats = num_format(item[:-1], 'int')
                new_line += " " + " ".join([last_item] * n_repeats)
            else:
                new_line += " " + item
            last_item = item

        new_block.append(new_line)

    return new_block


def expand_jumps(values):
    """Given split values, convert cases like 3j into j j j

    Parameters
    ----------
    values : List of str
        The card parameters already split by spaces

    Returns
    -------
    new_values : List of str
        New values with repeated jumps replaced
    """

    new_values = []

    for value in values:
        # We either have #j or j
        if len(value) > 1:
            # Then we *may* have #j
            if value[:-1].isdigit() and value[-1] == "j":
                # This is a #j!
                num_jumps = num_format(value[:-1], 'int')
                for j in range(num_jumps):
                    new_values.append("j")
            else:
                # It is not
                new_values.append(value)
        else:
            # This could be a jump or not, but either way we dont need
            # to do anything, just copy it over
            new_values.append(value)

    return new_values


def strip_comments(line):
    """Given a line, this strips out all commented information.

    Specifically, if it finds a 'c' in the first CARD_TYPE_SIZE
    positions, then the whole line is a comment (and an empty string is
    returned). If a '$' is identified, all content, including the '$' is
    removed in the returned string.

    Parameters
    ----------
    line : str
        The input line to operate on.

    Returns
    -------
    str:
        The line stripped of all comments and trailing spaces.
    """

    # CARD_TYPE_SIZE characters, removing leading spaces, and making sure
    # the c is either a blank or end of line
    first_entry = line.lstrip().split(maxsplit=1)[0].lower()
    c_index = first_entry.find(NEWLINE_COMMENT)
    is_comment = False
    if c_index == 0:
        # Then we start with a c character
        if len(first_entry) > 1:
            # Then there is more, if this is a comment, the next character
            # is a space
            if first_entry[c_index + 1].isspace():
                # Comment!
                is_comment = True
            # If there is no more, then this is also a comment
        else:
            is_comment = True
    if is_comment:
        stripped_line = ""
    else:
        # Ok this is a line with real data, so lets remove all
        # information to the right of the comment character
        index = line.find(COMMENT)
        if index != -1:
            # -1 indicates COMMENT was not found, so we only do this
            # if we found it
            stripped_line = line[:index]
        else:
            stripped_line = line

        # Finally remove white space and newlines from the right side
        stripped_line = stripped_line.rstrip()

    return stripped_line


def combine_lines(block):
    """Combines data on multiple lines (via continuations or, blanks in
    first CARD_TYPE_SIZE spaces) onto a single line.
    This method also removes all leading spaces since after we extract
    line continuation information, the spaces provide no further
    information.

    Parameters
    ----------
    block : List of str
        The block where each entry in the list is a line from the file
        that has already has comments and line-returns removed

    Returns
    -------
    List of str:
        A new block where cards are combined onto a single line
        and thus into a single entry in the list.

    """

    reduced_block = []
    # For code readability this code will first combine lines where the
    # continuation was denoted with CARD_TYPE_SIZE blanks at the start
    # of the line
    for line in block:
        if line.startswith(" " * CARD_TYPE_SIZE):
            # This shouldn't be the first line, but if it is, then
            # reduced_block will be empty and this will raise an error
            # lets provide a more accurate error message just in case
            if len(reduced_block) == 0:
                raise ValueError("Line continuation detected at the "
                                 "beginning of a block!")
            else:
                # Ok, add it to the previous line with a blank between
                # this line and the last
                # There can be an ampersand on the line above, if so,
                # remove it
                reduced_block[-1] = \
                    " ".join([reduced_block[-1].rstrip(" " + CONTINUATION),
                              line.lstrip()])
        else:
            reduced_block.append(line.lstrip())

    # Now we can go through the once reduced block and treat the case of
    # line continuation characters
    reduced_block_again = []

    i = 0
    while i < len(reduced_block):
        this_line = reduced_block[i]
        if this_line.rstrip().endswith(CONTINUATION):
            # Then this line is a continuation, so we want to store
            # this line and the next line as one
            if i == len(reduced_block) - 1:
                # Then there is no next line so raise this as an error
                raise ValueError("Line continuation detected at the "
                                 "end of a block!")
            reduced_block_again.append(" ".join(
                [this_line[:this_line.index(CONTINUATION)],
                 reduced_block[i + 1].lstrip()]))
            i += 2
        else:
            # just store this line
            reduced_block_again.append(this_line.lstrip())
            i += 1

    # Now go through and check the correctness of each line
    for line in reduced_block_again:
        find_invalid_cards(line)

    return reduced_block_again


def card_format(card):
    # Use the standard textwrap package to have an initial indent of 0
    # and subsequent indents of 5 spaces
    # Remove multiple spaces and replace with just one
    shorter_card = ' '.join(card.split())
    new_card = textwrap.fill(shorter_card, initial_indent="",
                             subsequent_indent=" " * CARD_TYPE_SIZE,
                             width=MAX_LINE_LEN)
    new_card += "\n"

    return new_card


def num_format(value: str, num_type: str = None):
    """Converts strings from MCNP output to the desired numeric value.

    This function converts strings that feature exponential formats that are
    allowed by MCNP but not recognized by python, e.g., -1+1 representing -10.
    In addition, it allows integer in exponential form, e.g., 1E+1 returns a
    valid integer. The function raises a ValueError exception if value/num_type
    combination does not result in a valid numeric format as expected by ADDER.

    Parameters
    ----------
    value : str
        The string containing the number to parse 
    num_type : str, default: None
        Allowed values are 'int' and 'float'

    Returns
    -------
    parsed_value : int or float
        The value included in the value string converted to a number if
        consistent with the requested num_type
    """

    if num_type not in ['int', 'float']:
        raise ValueError(f'Unexpected number type "{num_type}" '
                         f'provided to function num_format')

    # First try to use the built-in functions,
    # otherwise try the fancier parsing approach
    if num_type == 'int':
        try:
            parsed_value = int(value)
            return parsed_value
        except ValueError:
            pass
    elif num_type == 'float':
        try:
            parsed_value = float(value)
            return parsed_value
        except ValueError:
            pass

    value = value.upper() 
    new_value = ''
    for ichar, char in enumerate(value):
        if char in ['+', '-'] and not ichar == 0 and \
                not value[ichar - 1] == 'E':
            new_value += 'E' + char
        else:
            new_value += char

    if num_type == 'int':
        if float(new_value) == int(float(new_value)): 
            parsed_value = int(float(new_value))
        else:
            raise ValueError(f'{new_value} does not represent an int')
    elif num_type == 'float':
        parsed_value = float(new_value)

    return parsed_value
