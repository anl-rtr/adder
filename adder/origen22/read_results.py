from collections import OrderedDict


def get_tape7(filename="TAPE7.OUT", vector=-1):
    """This method reads a TAPE 7 (i.e., a punch card) output created by
    ORIGEN2.2 with the transmuted inventories.

    Parameters
    ----------
    filename : str, optional
        The TAPE7 filename, defaults to ORIGEN's default: TAPE7.OUT.
    vector : int, optional
        The vector to return, defaults to the last vector

    Returns
    -------
    OrderedDict
        Dictionary where the key is the nuclide identifier and the value
        is the inventory in moles.
    """

    # While wasteful, it is simpler to find the read the file into a
    # list of strings (one for each line)
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Convert to a 2D list, where the first dimension is the output
    # vector index, the second dimension is the nuclide type
    # (activation, actinide, fission product) and the entry is a
    # dictionary where the key is the nuclide name (string) and the
    # value is the mass in grams
    vector_data = []
    data = [{}, {}, {}]
    for i, line in enumerate(lines):
        # Each line is formatted as: 2X,I2,4(1X,I6,2X,lPE10.4)
        tokens = line.split()
        nuc_type = int(tokens[0])
        if nuc_type != 0:
            # Then this is of the same vector, so add it to data,
            # converting types as we go and neglecting the 0s
            card = {}
            for i in range(1, 9, 2):
                if int(tokens[i]) != 0:
                    nuc = tokens[i]
                    card[nuc] = float(tokens[i + 1])
                else:
                    break
            data[nuc_type - 1].update(card)
        else:
            # Then this is the end, we will not save this line but we
            # will store our current vector and move on to the next
            vector_data.append(data[:])
            data = [{}, {}, {}]

    # At this point we want to reorder the inner list to instead be a
    # dictionary where the key is the nuclide and the value is a list of
    # masses for each vector
    # Get the set of nuclides from each of the dict keys in vector_data
    nuclide_set = set()
    for i in range(len(vector_data)):
        for j in range(len(vector_data[i])):
            nuclide_set |= vector_data[i][j].keys()

    results = OrderedDict()
    for nuclide in sorted(nuclide_set):
        nuclide_sum = 0.
        i = vector
        for j in range(len(vector_data[i])):
            if nuclide in vector_data[i][j]:
                nuclide_sum += vector_data[i][j][nuclide]

        # Only include the nuclide if it has a non-zero amount
        if nuclide_sum > 0.:
            results[nuclide] = nuclide_sum

    return results
