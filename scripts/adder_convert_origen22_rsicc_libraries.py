#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py

import adder

# The following contains the cross section library identifiers as
# provided on page 217 of the ORIGEN 2.2 RSICC package manual, CCC-371.
# The first entry is for activation products, second for actinides,
# third for FPs. The fourth is for NLIB(12) if applicable
DECAY_LIBS = {'DECAY':  [1, 2, 3, None]}
XS_LIBS = {"PWRU":      [204, 205, 206, 1],
           "PWRPUU":    [207, 208, 209, 2],
           "PWRPUPU":   [210, 211, 212, 3],
           "PWRDU3TH":  [213, 214, 215, 7],
           "PWRPUTH":   [216, 217, 218, 8],
           "PWRU50":    [219, 220, 221, 9],
           "PWRD5D35":  [222, 223, 224, 10],
           "PWRD5D33":  [225, 226, 227, 11],
           "PWRUS":     [601, 602, 603, 38],
           "PWRUE":     [604, 605, 606, 39],
           "BWRU":      [251, 252, 253, 4],
           "BWRPUU":    [254, 255, 256, 5],
           "BWRPUPU":   [257, 258, 259, 6],
           "BWRUS":     [651, 652, 653, 40],
           "BWRUS0":    [654, 655, 656, 41],
           "BWRUE":     [657, 658, 659, 42],
           "CANDUNAU":  [401, 402, 403, 21],
           "CANDUSEU":  [404, 405, 406, 22],
           "EMOPUUUC":  [301, 302, 303, 18],
           "EMOPUUUA":  [304, 305, 306, 19],
           "EMOPUUUR":  [307, 308, 309, 20],
           "AMOPUUUC":  [311, 312, 313, 12],
           "AMOPUUUA":  [314, 315, 316, 13],
           "AMOPUUUR":  [317, 318, 319, 14],
           "AMORUUUC":  [321, 322, 323, 15],
           "AMORUUUA":  [324, 325, 326, 16],
           "AMORUUUR":  [327, 328, 329, 17],
           "AMOPUUTC":  [331, 332, 333, 32],
           "AMOPUUTA":  [334, 335, 336, 33],
           "AMOPUUTR":  [337, 338, 339, 34],
           "AMOPTTTC":  [341, 342, 343, 29],
           "AMOPTTTA":  [344, 345, 346, 30],
           "AMOPTTTR":  [347, 348, 349, 31],
           "AMO0TTTC":  [351, 352, 353, 35],
           "AMO0TTTA":  [354, 355, 356, 36],
           "AMO0TTTR":  [357, 358, 359, 37],
           "AMO1TTTC":  [361, 362, 363, 23],
           "AMO1TTTA":  [364, 365, 366, 24],
           "AMO1TTTR":  [367, 368, 369, 25],
           "AMO2TTTC":  [371, 372, 373, 26],
           "AMO2TTTA":  [374, 375, 376, 27],
           "AMO2TTTR":  [377, 378, 379, 28],
           "FFTFC":     [381, 382, 383, 0],
           "THERMAL":   [201, 202, 203, 0],
           "CRBRC":     [501, 502, 503, 0],
           "CRBRA":     [504, 505, 506, 0],
           "CRBRR":     [507, 508, 509, 0],
           "CRBRI":     [510, 511, 512, 0]}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_library(path, name, h5_file):
    xs_filename = str(Path.joinpath(path, name + ".LIB"))
    decay_filename = str(Path.joinpath(path, "DECAY.LIB"))
    xs_ids = {"activation": XS_LIBS[name][0], "actinide": XS_LIBS[name][1],
              "fp": XS_LIBS[name][2]}
    decay_ids = {"activation": DECAY_LIBS["DECAY"][0],
                 "actinide": DECAY_LIBS["DECAY"][1],
                 "fp": DECAY_LIBS["DECAY"][2]}

    lib = adder.DepletionLibrary.from_origen(xs_filename, decay_filename,
                                             xs_ids, decay_ids, new_name=name,
                                             var_actinide_lib=XS_LIBS[name][3])

    lib.to_hdf5(h5_file)


description = """
This script can be used to create HDF5 depletion data libraries used by
ADDER with data sourced from the ORIGEN2.2 RSICC distribution.

"""


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=description,
    formatter_class=CustomFormatter
)
parser.add_argument('-d', '--destination', default='./origen_lib.h5',
                    help='Path and filename of HDF5 library file to write')
parser.add_argument('-r', '--rsicc',
                    help='This denotes the directory of the '
                    'RSICC-distributed libraries to process')
parser.add_argument('-o', '--overwrite', default=False, type=str2bool,
                    const=True, nargs='?',
                    help='Whether or not to overwrite the destination HDF5 '
                         'file if it is present')
args = parser.parse_args()

# Get and create the destination path if needed
dest_file = Path(args.destination)
dest_path = dest_file.parent
dest_path.mkdir(parents=True, exist_ok=True)

# Determine if we are overwriting or appending
overwrite = args.overwrite
if overwrite:
    mode = "w"
else:
    mode = "a"
# Create the h5 file we will store everything in
h5_file = h5py.File(dest_file, mode)

# Process rsicc option
if args.rsicc is not None:
    rsicc_path = Path(args.rsicc)
    # Make sure this directory exists
    if not rsicc_path.is_dir():
        msg = "Invalid RSICC path: {}, ".format(rsicc_path) + \
              "ensure this is a path to a directory, that directory exists" + \
              " and you have read access!"

        raise ValueError(msg)

    # If here, the user wants RSICC data, the directory works, and the
    # h5 file is setup.
    # Go ahead and get it
    for name in XS_LIBS:
        print("Processing ORIGEN2.2 Library {}".format(name))
        get_library(rsicc_path, name, h5_file)
