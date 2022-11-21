#!/usr/bin/env python

import argparse

import adder

description = """
The Advanced Dimensional Depletion for Engineering of Reactors (ADDER)
software.

"""

def main():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input", help="ADDER input file")
    parser.add_argument("-n", "--no-deplete", action="store_true",
                        help="If included, the depletion commands will be "
                        "skipped; use this to interrogate fuel management "
                        "operations")
    parser.add_argument("-f", "--fast-forward", action="store_true",
                        help="If included, any existing neutronics output "
                        "files from a previous ADDER run in the working "
                        "directory will be used instead of re-computed. When "
                        "using this option, the user must be sure that the "
                        "neutronics output files are consistent with the "
                        "current case. Note this only skips the neutronics "
                        "calculations done during deplete and geom_sweep "
                        "operations.")
    args = parser.parse_args()

    input_file = args.input
    no_deplete = args.no_deplete
    fast_forward = args.fast_forward

    root_logger = adder.init_root_logger('adder')
    logger = adder.init_logger(__name__)
    msg = "Beginning ADDER Version {} Execution".format(adder.__version__)
    logger.info(msg)

    # Get and process input
    rx, ops = adder.get_input(input_file)

    # Now execute the operations
    rx.process_operations(ops, no_deplete, fast_forward)


if __name__ == "__main__":
    main()
