from collections import OrderedDict

from .constants import *
from adder.depletionlibrary import DepletionLibrary, origen_nucid_to_GND


def write_input(case_name, composition, duration, duration_units, num_substeps,
                irradiation_level, origen_isotope_types, num_nonstd_rxn,
                irrad_is_flux=True, verbose_output=False):
    """This method creates an ORIGEN 2.2 input file that burns the
    provided nuclides for a given time at a given flux or power.

    This method will use either constant flux or constant power
    irradiation mode, depending on the value of :param:`irrad_is_flux`.
    Since ORIGEN treats fission products, actinides, and activation
    isotopes separately, this script will use the information within
    :param:`decay_libs` and :param:`xs_libs` to first assign actinides,
    then activation isotopes and finally the fission products.

    Parameters
    ----------
    case_name : str
        Descriptive name for the case.
    composition : OrderedDict
        The material composition to model with the keys being the
        nuclide identifier in 10000*Z+10*A+IS format, and the values
        being the corresponding isotopic molar densities in units of
        mol/cm^3 (g-atom/cm^3 in ORIGEN parlance).
    duration : float
        The duration of the irradiation or decay interval
    duration_units : {"s", "m", "h", "d", or "y"}
        A string identifier for the time units.
    num_substeps : int
        The number of substeps to use within ORIGEN when performing the
        depletion
    irradiation_level : float
        Irradiation flux in units of neutrons/cm^2-sec or specific power
        in units of MW per cm^2.
    origen_isotope_types : dict
        The isotope types of each valid ORIGEN NUCID
    num_nonstd_rxn : int
        The number of non standard reaction channels to include
    irrad_is_flux : bool, optional
        Denotes whether :param:`irradiation_level` denotes a flux (True)
        or a power (False), defaults to True
    verbose_output : bool, optional
        Whether or not to include the OPTL/A/F and OUT cards in the
        ORIGEN file. Defaults to False (do not include).

    Returns
    ----------
    tape5 : str
        Properly-formatted input file ready to be written to tape5.inp
        for ORIGEN 2.2 execution.

    """

    # Build the LIB card
    # This card is defined in Sec 4.18 of the manual
    # We will:
    # - NOT print array A (NLIB(1) = -1)
    # - use the provided decay and xs/yield data in depletion_lib
    # - use the x/s libs according to the core type
    # - Use the tape defined in _XS_LIB_UNIT for input
    # - use tape 50 for substitutions
    # - specify the number of non-standard reactions
    # - use Th, U, Pu, Cm, and Cf for the actinides with direct fp yields
    #   according to Table 4.5
    LIB = "  LIB  {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        -1, ORIGEN_LIB_IDS[0], ORIGEN_LIB_IDS[1], ORIGEN_LIB_IDS[2],
        ORIGEN_LIB_IDS[3], ORIGEN_LIB_IDS[4], ORIGEN_LIB_IDS[5],
        XS_LIB_UNIT, SUBXS_LIB_UNIT, -num_nonstd_rxn, ACTINIDE_YIELD_SET,
        VAR_ACTINIDE_ID)

    # Build the PHO card if needed
    PHO = ""
    # When we want to include this, use the following:
    # # This card is defined in Sec 4.19 of the manual
    # # We will:
    # # - Use the tape defined in _PHOTON_LIB_UNIT for input
    # PHO = "  PHO  {} {} {} {}\n".format(
    #     activation_photon_lib.origen_id, actinide_photon_lib.origen_id,
    #     fp_photon_lib.origen_id, PHOTON_LIB_UNIT)

    # Build the INP card
    # This card is defined in Section 4.6 of the manual
    # We will
    # - Store compositions in vector 1
    # - Read nuclide compositions on unit 5 w/ mol/basis unit
    # - Dont read continuous nuclide feed or removal data
    # - But give the units for these feeds as seconds
    INP = "  INP  {} {} {} {} {} {}\n".format(1, 2, -1, -1, 1, 1)

    # Now we either add a decay card, flux-irradiation card, or a power-card
    # depending irradiation_level and irrad_is_flux
    DEC_IRF_IRP = ""
    from_vec = 1
    to_vec = 2
    # The first time through we want the start time to be 0
    # "1" is the flag to pass to ORIGEN to set this
    start_time = 2
    dt = duration / (num_substeps + 1.)
    for i in range(num_substeps + 1):
        if i > 0:
            # The 2nd time through we want the start time to be where we
            # just left off; "0" is the flag to pass to ORIGEN for this
            start_time = 0

            # Adjust our start and end vectors
            from_vec = to_vec
            if to_vec == MAX_NUM_VECTORS:
                to_vec = 2
            else:
                to_vec += 1

        if irradiation_level > 0.:
            if irrad_is_flux:
                # Then we will have an IRF card from Section 4.21
                DEC_IRF_IRP += "  IRF  "
            else:
                # Then we will have an IRP card from Section 4.22
                DEC_IRF_IRP += "  IRP  "
            # In either case we will:
            # - Get data from vector 1 and put in vector 2
            # - Set time to previous end
            DEC_IRF_IRP += "{:1.15E} {:1.15E} {} {} {} {}\n".format(
                (i + 1) * dt, irradiation_level, from_vec, to_vec,
                ORIGEN_TIME_UNITS[duration_units], start_time)
        elif irradiation_level == 0.:
            # Then we will have a DEC card from Section 4.23 of the manual
            # We will
            # - Get data from vector 1 and put in vector 2
            # - Set time to previous end
            DEC_IRF_IRP += "  DEC  {:1.15E} {} {} {} {}\n".format(
                (i + 1) * dt, from_vec, to_vec,
                ORIGEN_TIME_UNITS[duration_units], start_time)

    # Ensure that our final output vector is in vector 2
    out_vec = 2
    if to_vec != out_vec:
        MOV = "  MOV {} {} 0 1.0\n".format(to_vec, out_vec)
    else:
        MOV = ""

    if verbose_output:
        # Build or OPT* cards
        # These cards are defined in Section 4.25-4.27 of the manual
        # We will
        # - Only request the composition in grams for each nuclide
        OUT = "  OPTL 4*8 5 19*8\n  OPTA 4*8 5 19*8\n  OPTF 4*8 5 19*8\n"

        # Print our output
        # This is defined in Section 4.5 of the manual
        # We will
        # - print the first 2 vectors (start and end)
        # - not print loop, recycle info
        # - consider all vectors for inclusion in summary
        OPTs = "  OUT  {} {} {} {}\n".format(2, 1, -1, 0)
        OUT += OPTs
    else:
        OUT = ""

    # Setup our "punch-card" output for the vectors before
    # (vector 1) and after (2) irradiation
    PCH = "  PCH  1 1 1\n  PCH  {0} {0} {0}\n".format(out_vec)

    # Now we can end with our material information.
    # We need to divvy up the materials into their "classes"
    # of actinides, activation, and fps.
    activ_isos = []
    actin_isos = []
    fp_isos = []
    for nuc, density in composition.items():
        iso_name = origen_nucid_to_GND(nuc)
        for set_type, working_list in zip(["actinide", "activation", "fp"],
                                          [actin_isos, activ_isos, fp_isos]):
            iso_types = origen_isotope_types[set_type]
            if iso_name in iso_types:
                working_list.append((nuc, density))
                # Now stop checking the isotope sets because we only
                # want to have the isotope show up in one of the types
                break

    # Now we can define our materials in the input deck
    def _make_mat_cards(next_id, mat_set):
        mat_cards = ""
        if mat_set:
            fmt = " {:1} {:>6} {:1.15E} 0 0.\n"
            for _, (nuc, density) in enumerate(mat_set):
                mat_cards += fmt.format(next_id, nuc, density)
        return mat_cards

    # Our input is done being generated, lets build our tape5 deck so far
    tape5 = "".join(["-1\n", "-1\n", "-1\n", LIB, PHO, INP, DEC_IRF_IRP, MOV,
                     OUT, PCH, "  END\n", _make_mat_cards(1, activ_isos),
                     _make_mat_cards(2, actin_isos),
                     _make_mat_cards(3, fp_isos), " 0\n"])

    return tape5
