#!/usr/bin/env python3
import argparse
from pathlib import Path

import adder


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_library(xs_filename, decay_filename, lib_name,
                xs_ids, decay_ids, var_id, h5_filename):

    lib = adder.DepletionLibrary.from_origen(xs_filename, decay_filename,
                                             xs_ids, decay_ids,
                                             new_name=lib_name,
                                             var_actinide_lib=var_id)

    lib.to_hdf5(h5_filename)


description = """
This script can be used to create HDF5 depletion data libraries used by
ADDER with data sourced from custom ORIGEN2.2 libraries.

Example usage:
adder_convert_origen22_library.py ./PWRU.LIB ./DECAY.LIB 204 205 206 1 2 3 PWRU -d ./test.h5

"""


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=description,
    formatter_class=CustomFormatter
)
parser.add_argument('xslib_filename', action='store', type=str,
                    help='This denotes the filename of the ORIGEN2.2 ' +
                    'cross section and yield library')
parser.add_argument('decay_filename', action='store', type=str,
                    help='This denotes the filename of the ORIGEN2.2 ' +
                    'decay library')
parser.add_argument('xslib_ids', action='store', nargs=3, type=int,
                    help='This denotes the activation, actinide and ' +
                    'fission product library IDs in the cross section and ' +
                    'yield library. Three values are REQUIRED.')
parser.add_argument('decay_ids', action='store', nargs=3, type=int,
                    help='This denotes the activation, actinide and ' +
                    'fission product library IDs in the decay library. ' +
                    'Three values are REQUIRED.')
parser.add_argument('new_name', action='store', type=str,
                    help='This denotes the name of the library in the ' +
                    ' HDF5 file.')
parser.add_argument('-d', '--destination', default='./origen_lib.h5',
                    help='Path and filename of HDF5 object to write')
parser.add_argument('-o', '--overwrite', default=False, type=str2bool,
                    const=True, nargs='?',
                    help='Whether or not to overwrite the destination HDF5 '
                         'file if it is present')
args = parser.parse_args()

# Get and create the destination path if needed
dest_file = Path(args.destination)
dest_path = dest_file.parent
dest_path.mkdir(parents=True, exist_ok=True)

# Get the xslib_filename and decay_filename
if not Path(args.xslib_filename).exists():
    msg = "{} does not exist!".format(args.xslib_filename)
    raise ValueError(msg)
if not Path(args.decay_filename).exists():
    msg = "{} does not exist!".format(args.decay_filename)
    raise ValueError(msg)
xslib_filename = args.xslib_filename
decay_filename = args.decay_filename

# Get the xslib_ids
for lib_id in args.xslib_ids:
    if lib_id < 0:
        raise ValueError("All xslib_ids must be positive!")
for lib_id in args.decay_ids:
    if lib_id < 0:
        raise ValueError("All decay_ids must be positive!")
xslib_ids = {"activation": args.xslib_ids[0], "actinide": args.xslib_ids[1],
             "fp": args.xslib_ids[2]}
decay_ids = {"activation": args.decay_ids[0], "actinide": args.decay_ids[1],
             "fp": args.decay_ids[2]}

new_name = args.new_name

# Determine if we are overwriting or appending
overwrite = args.overwrite
if overwrite and dest_file.exists():
    # Then delete the file
    dest_file.unlink()

# Set var_id to its base value for now
var_id = 0

# Process the library
print("Processing ORIGEN2.2 Library {}".format(new_name))
get_library(xslib_filename, decay_filename, new_name, xslib_ids, decay_ids,
            var_id, str(dest_file))
