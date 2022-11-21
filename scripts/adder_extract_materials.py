#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict

import h5py
import numpy as np

import adder


def get_data(h5_file, mat_name):
    h5_in = h5py.File(h5_file, "r")

    # We need to process the case_#/operation_#/step_# trees and get the data
    # And figure out the right ordering
    case_ids = []
    op_ids = []
    step_ids = []
    grp_names = []

    # Get the sorted order of the cases
    case_ids = sorted([int(k[len("case_"):])
                        for k in h5_in.keys() if k.startswith("case_")])
    # Now iterate through in this sorted order
    for case_id in case_ids:
        case_grp = h5_in["case_" + str(case_id)]

        # Get the sorted order of the operations
        op_ids = sorted([int(k[len("operation_"):])
                         for k in case_grp.keys()
                         if k.startswith("operation_")])
        # Now iterate through in this sorted order
        for op_id in op_ids:
            op_grp = case_grp["operation_" + str(op_id)]
            # Get the sorted order of the steps
            step_ids = sorted([int(k[len("step_"):])
                               for k in op_grp.keys()
                               if k.startswith("step_")])

            # Now we can create the group names
            for step_id in step_ids:
                grp_name = "case_{}/operation_{}/step_{}".format(case_id,
                                                                 op_id,
                                                                 step_id)
                grp_names.append(grp_name)

    # Initialize data and progress through the order
    case_labels = []
    operation_labels = []
    step_idxs = []
    times = []
    keffs = []
    keffs_stddev = []
    powers = []
    fluxes_1grp = []
    Q_recs = []
    isotope_data = []
    isotope_set = set()
    for grp_name in grp_names:
        grp = h5_in[grp_name]
        materials_grp = grp['materials']
        if mat_name in materials_grp.keys():
            material = adder.Material.from_hdf5(materials_grp, mat_name)
            case_labels.append(grp.attrs["case_label"].decode())
            operation_labels.append(grp.attrs["operation_label"].decode())
            step_idxs.append(int(grp.attrs["step_idx"]))
            times.append(float(grp.attrs["time"]))
            keffs.append(grp.attrs["keff"])
            keffs_stddev.append(grp.attrs["keff_stddev"])
            powers.append(grp.attrs["power"])
            Q_recs.append(grp.attrs["Q_recoverable"])
            fluxes_1grp.append(np.sum(material.flux))

            isotope_info = defaultdict(lambda: 0.0)
            for i in range(len(material.isotopes)):
                isotope_info[material.isotopes[i].name] = \
                    material.number_densities[i]
            isotope_set.update(isotope_info.keys())
            isotope_data.append(isotope_info)

    results = OrderedDict()
    results['case_label'] = case_labels
    results['operation_label'] = operation_labels
    results['step_idx'] = step_idxs
    results['times'] = times
    results['powers'] = powers
    results['keffs'] = keffs
    results['keffs_stddev'] = keffs_stddev
    results['fluxes_1grp'] = fluxes_1grp
    results['Q_recs'] = Q_recs
    results['isotope_data'] = isotope_data
    results['isotope_set'] = sorted(list(isotope_set))

    return results


def make_csv(fname, results):
    # Print header
    with open(fname, "w") as file:
        # Write the header
        header = "# data \\ time [d]\n"
        file.write(header)

        # First print labels
        for key in ["case_label", "operation_label", "step_idx"]:
            labels = results[key]
            label_row = "{}, ".format(key)
            for l in labels:
                label_row += "{}, ".format(l)
            file.write(label_row + "\n")

        # Then the times
        row = "times, "
        values = results["times"]
        for v in values:
            row += "{:1.6E}, ".format(v)
        file.write(row + "\n")

        # Everything else aside from isotopics and what was already printed
        for entry in results.keys():
            if entry not in ["case_label", "operation_label", "step_idx",
                             "times", "isotope_set", "isotope_data"]:
                row = entry + ", "
                values = results[entry]
                for v in values:
                    row += "{:1.6E}, ".format(v)
                file.write(row + "\n")

        # And end with isotopics
        isotope_set = results["isotope_set"]
        isotope_data = results["isotope_data"]
        for iso in isotope_set:
            iso_row = "{}, ".format(iso)
            for t in range(len(results["times"])):
                iso_row += "{:1.6E}, ".format(isotope_data[t][iso])

            file.write(iso_row + "\n")


description = """
This script can be used to parse an ADDER HDF5 output file to create a
comma-separated-value file containing the time-valued isotopic
concentrations of a requested material. The rows will be the isotopes
and the columns will be the times.

Example usage:
adder_extract_materials.py in.h5 out.csv mat_name

"""


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=description,
    formatter_class=CustomFormatter
)
parser.add_argument('results_file', action='store', type=str,
                    help='The HDF5 file to read from')
parser.add_argument('csv_file', action='store', type=str,
                    help='The CSV file to create and write to')
parser.add_argument('material_name', action='store', type=str,
                    help='The material name to process')
args = parser.parse_args()

# Get the results file
if not Path(args.results_file).exists():
    msg = "{} does not exist!".format(args.results_file)
    raise ValueError(msg)
results_file = args.results_file

# Get and create the destination path if needed
csv_file = Path(args.csv_file)
csv_path = csv_file.parent
csv_path.mkdir(parents=True, exist_ok=True)

# Process the library
print("Processing ADDER Output: {}".format(results_file))

# Get the data
results = get_data(results_file, args.material_name)

# Write the data
make_csv(str(csv_file), results)
