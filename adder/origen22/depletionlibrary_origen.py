import numpy as np
import adder.data
from .constants import *

STANDARD_RXNS = {"(n,gamma)", "(n,2n)", "(n,gamma)*", "(n,2n)*", "fission",
                 "(n,3n)", "(n,p)", "(n,a)"}

_ALLOWED_ISOTOPE_TYPES = ["activation", "actinide", "fp"]


def make_origen_decay_lib(this, isotope_types, start_lib_id=1):
    """Write the data within this object to an ORIGEN2.2 ASCII file.

    Parameters
    ----------
    this : DepletionLibrary
        The depletion library object to print to ORIGEN2.2 format
    isotope_types : dict
        Dictionary with the set of isotopes in actinides, activation, and fp keys
    start_lib_id : int, optional
        The integral identifier for this library; defaults to 1

    Returns
    -------
    lib : str
        The decay libraries to write
    """

    lib_id = start_lib_id
    # For ease of reading, convert the isotope printing
    # order to be by numerical origen nucid number
    sorted_isotopes = sorted(set(this.isotopes.keys()),
        key=GND_to_origen_nucid)
    lib_type = "decay"
    lib = ""
    for iso_type in _ALLOWED_ISOTOPE_TYPES:
        # Write the header
        lib += _write_header(lib_id, this.name, lib_type, iso_type)
        iso_type_set = isotope_types[iso_type]
        # Write the decay data for each channel
        for iso_name in sorted_isotopes:
            if iso_name in iso_type_set:
                lib += _write_decay(this, lib_id, iso_name)
        # Terminate the library
        lib += " {:>3d}\n".format(-1)
        lib_id += 1

    return lib


def make_origen_xs_nfy_lib(this, isotope_types, flux, start_lib_id=4):
    """Write the data within this object to an ORIGEN2.2 ASCII file.

    Parameters
    ----------
    this : DepletionLibrary
        The depletion library object to print to ORIGEN2.2 format
    isotope_types : dict
        Dictionary with the set of isotopes in actinides, activation, and fp keys
    flux : np.ndarray, optional
        Flux to use as a weighting coefficient; defaults to a unit
        flux
    start_lib_id : int, optional
        The integral identifier for this library; defaults to 4
    """

    lib_id = start_lib_id
    lib_type = "xs_nfy"
    lib = ""
    # For ease of reading, convert the isotope printing
    # order to be by numerical origen nucid number
    sorted_isotopes = sorted(set(this.isotopes.keys()),
                             key=GND_to_origen_nucid)
    for iso_type in _ALLOWED_ISOTOPE_TYPES:
        # Write the header
        lib += _write_header(lib_id, this.name, lib_type, iso_type)
        iso_type_set = isotope_types[iso_type]
        for iso_name in sorted_isotopes:
            if iso_name in iso_type_set:
                lib += _write_xs_nfy(this, lib_id, iso_name, iso_type, flux)
        # Terminate the library
        lib += " {:>3d}\n".format(-1)
        lib_id += 1
    return lib


def to_origen(this, isotope_types, filepath, flux=None, start_lib_id=1,
              overwrite=True, decay_lib=None):
    """Write the data within this object to an ORIGEN2.2 ASCII file.

    Parameters
    ----------
    this : DepletionLibrary
        The depletion library object to print to ORIGEN2.2 format
    isotope_types : dict
        Dictionary with the set of isotopes in actinides, activation, and fp keys
    filepath : str
        Path to the file to write to
    flux : np.ndarray, optional
        Flux to use as a weighting coefficient; defaults to a unit
        flux
    start_lib_id : int, optional
        The integral identifier for this library; defaults to 1
    overwrite : bool, optional
        Whether or not to overwrite the information in the file;
        default is to not overwrite (i.e., append to) the file.
    decay_lib : None or str, optional
        The pre-computed decay library string, if available
    """

    # Set defaults
    if flux is None:
        the_flux = np.ones(this.num_neutron_groups)
    else:
        the_flux = flux

    # Open the file for writing
    if overwrite:
        write_mode = 'w'
    else:
        write_mode = 'a'

    if decay_lib is None:
        dk_lib = make_origen_decay_lib(this, isotope_types)
    else:
        dk_lib = decay_lib
    xs_nfy_lib = make_origen_xs_nfy_lib(this, isotope_types, the_flux)

    with open(filepath, mode=write_mode) as file:
        file.write(dk_lib)
        file.write(xs_nfy_lib)


# TODO: When ORIGEN non-std rxns are understood, re-enable this
# def create_nonstandard_rxn_lib(this, filename, flux=None, overwrite=True):
#     """Determine if this library contains any reactions which are not
#     considered standard by ORIGEN2.2; if so return their number and
#     write the corresponding library

#     Parameters
#     ----------
#     this : DepletionLibrary
#         The depletion library object to print to ORIGEN2.2 format
#     filename : str
#         Path and name of the file to write to
#     flux : np.ndarray, optional
#         Flux to use as a weighting coefficient; defaults to a unit
#         flux
#     overwrite : bool, optional
#         Whether or not to overwrite the information in the file;
#         default is to not overwrite (i.e., append to) the file.

#     Returns
#     -------
#     num_nonstd_rxn : int
#         The number of non-standard reactions identified

#     """

#     # Check the types and set defaults
#     check_type("filename", filename, str)
#     if flux is None:
#         the_flux = np.ones(this.num_neutron_groups)
#     else:
#         check_iterable_type("flux", flux, float)
#         the_flux = flux
#     check_type("overwrite", overwrite, bool)

#     # Initialize our output
#     num_nonstd_rxn = 0
#     library = ""

#     # For ease of reading, convert the isotope printing
#     # order to be by numerical origen nucid number
#     sorted_isotopes = sorted(this.isotopes, key=GND_to_origen_nucid)
#     for iso_name in sorted_isotopes:
#         iso = this.isotopes[iso_name]
#         # Determine if we have non-standard rxns

#         # Not every isotope has xs data, if it doesnt then no need
#         # to worry
#         if iso.neutron_xs is not None:
#             xs_lib_rxn_types = set(iso.neutron_xs._products.keys())
#             # Find the values in xs_lib_rxn_types not in STANDARD_RXNS
#             non_std_rxn_types = xs_lib_rxn_types - STANDARD_RXNS
#         else:
#             non_std_rxn_types = set()

#         if len(non_std_rxn_types) > 0:
#             # Then there are some non-std rxns
#             # So, increment the counter
#             num_nonstd_rxn += len(non_std_rxn_types)

#             # Now write the reaction card for each
#             # This card is defined in Table 5.4 of the ORIGEN manual
#             # as "{NPAR} {NDAUG} {XS}\n" with xs in barns
#             parent_id = GND_to_origen_nucid(iso_name)
#             for rxn in non_std_rxn_types:
#                 # Get children names
#                 for type_, (xs, targets, yields, _) in iso.neutron_xs.items():
#                     xs_val = np.dot(xs, the_flux)
#                     for t, target in enumerate(targets):
#                         child_id = GND_to_origen_nucid(target)
#                         xs_yield = xs_val * yields[t]

#                         # Write card
#                         card = "{:>6} {:>6}".format(parent_id, child_id)
#                         card += _f2s(xs_yield) + "\n"
#                         library += card

#     # Temporarily warn the user we found this and note that it is not
#     # yet working
#     if num_nonstd_rxn > 0:
#         # Would like this to be a warning, but it is too many lines of
#         # code to pass the logger all the way down to this level just
#         # for a temporary warning
#         import warnings
#         msg = "Non-Standard Reaction Type Detected; ADDER does not yet" + \
#             " support these reactions"
#         warnings.warn(msg)

#     # Now write the substitution library
#     if num_nonstd_rxn > 0:
#         if overwrite:
#             write_mode = 'w'
#         else:
#             write_mode = 'a'
#         with open(filename, write_mode) as file:
#             file.write(library)

#     return num_nonstd_rxn


########################################################################
# Support functions
########################################################################

def GND_to_origen_nucid(iso_name):
    # THIS IS THE SAME AS IN DEPLETIONLIBRARY.PY, TRYING NOT TO IMPORT
    # TO REDUCE CIRCULAR DEPENDENCIES; IF CHANGED, CHANGE UPSTREAM
    # Converts a GND-formatted name (e.g., "U235") to an origen nucid
    # format, returned as an integer
    Z, A, M = adder.data.zam(iso_name)
    origen_nucid = 10000 * Z + 10 * A + M
    return origen_nucid


def origen_nucid_to_GND(nucid):
    # THIS IS THE SAME AS IN DEPLETIONLIBRARY.PY, TRYING NOT TO IMPORT
    # TO REDUCE CIRCULAR DEPENDENCIES; IF CHANGED, CHANGE UPSTREAM
    # Get the Z, A, M
    Z = int(nucid) // 10000
    AM = int(nucid) % 10000
    A = AM // 10
    M = AM % 10
    return adder.data.gnd_name(Z, A, M)


def _nlb_nuclide(NLB, NUCLID):
    # Create the common NLB NUCLID pair that starts most cards
    return " {:>3d} {:>6}".format(NLB, NUCLID)


def _f2s(value):
    # ORIGEN has rules which are not standard for printing
    # data. This function converts using these rules so that a
    # diff of ASCII output from this function replicates what is
    # in the data libraries provided with ORIGEN.
    s = " {:1.8E}".format(value)
    return s

def _write_header(lib_id, lib_name, lib_type, iso_type):
    if lib_type == "decay":
        title = "{} {} library: {}".format(lib_name, lib_type, iso_type)
    else:
        title = "{} XS and Yield library: {}".format(lib_name, iso_type)
    # Tables 5.1, 5.2, and 5.5 all start with NLB and TITLE
    card = " {:>3d}    {:<72}\n".format(lib_id, title.upper())
    return card


def _write_decay(this, lib_id, iso_name):
    # This writes the specific decay library for a specific isotope
    # Get the abundance, arcg, wrcg from our big key list
    if iso_name in ORIGEN_ISO_DECAY_DATA:
        abund, arcg, wrcg = ORIGEN_ISO_DECAY_DATA[iso_name]
    else:
        abund, arcg, wrcg = (0., 0., 0.)
    # Overwrite abundance from ORIGEN with the data used elsewhere in
    # ADDER
    if iso_name in adder.data.NATURAL_ABUNDANCE:
        abund = adder.data.NATURAL_ABUNDANCE[iso_name] * 100.
    else:
        abund = 0.

    def _write_card(lib_id, origen_nuclid, data_units, thalf, beta_m_star,
                    beta_ec, beta_ec_star, alpha, it, sf, beta_neutron,
                    decay_energy, abund, arcg, wrcg):
        card = _nlb_nuclide(lib_id, origen_nuclid) + \
            " {:1d}".format(data_units) + _f2s(thalf) + _f2s(beta_m_star) + \
            _f2s(beta_ec)  + _f2s(beta_ec_star) + "\n" + " " * 13 + \
            _f2s(alpha) + _f2s(it) + "\n"
        # Second card
        card += " {:>3d}".format(lib_id) + _f2s(sf) + \
            _f2s(beta_neutron) + _f2s(decay_energy)  + _f2s(abund) + \
            _f2s(arcg) + "\n    " + _f2s(wrcg) + "\n"
        return card

    # Get the data
    iso = this.isotopes[iso_name]
    data = iso.decay
    origen_nuclid = str(GND_to_origen_nucid(iso_name))

    if data is None:
        # Write the typical stable card
        return _write_card(lib_id, origen_nuclid, 6, 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., abund, arcg, wrcg)

    # The following is per Table 5.1
    # First card
    # Get the data
    if data.half_life_units in ORIGEN_TIME_UNITS:
        data_units = ORIGEN_TIME_UNITS[data.half_life_units]
    else:
        # Should never get here, but need to not be raising errors
        data_units = ORIGEN_TIME_UNITS["stable"]
    # Initialize the data
    beta_m = 0.
    beta_m_star = 0.
    beta_ec = 0.
    beta_ec_star = 0.
    alpha = 0.
    it = 0.
    sf = 0.
    beta_neutron = 0.

    # Adjust for zero decay constants/stable conditions
    thalf = data.half_life
    if thalf is None:
        # Then denote this as stable
        data_units = ORIGEN_TIME_UNITS["stable"]
        # Origen just uses a 0 for thalf and friends if it is a
        # stable isotopeq
        thalf = 0.
    else:
        def find_state(targets, yields, find_m):
            for i, target in enumerate(targets):
                if target == "fission":
                    return i
                _, _, m = adder.data.zam(target)
                if m == find_m:
                    return i
            # if we got here we didnt find it
            return None

        for type_ in data.keys():
            br, targets, yields = data[type_]
            t0 = find_state(targets, yields, 0)
            t1 = find_state(targets, yields, 1)

            if type_ == "beta-":
                if t0 is not None:
                    beta_m = br * yields[t0]
                if t1 is not None:
                    beta_m_star = br * yields[t1]
            elif type_ == "ec/beta+":
                if t0 is not None:
                    beta_ec = br * yields[t0]
                if t1 is not None:
                    beta_ec_star = br * yields[t1]
            elif type_ == "alpha":
                if t0 is not None:
                    alpha = br * yields[t0]
            elif type_ == "it":
                if t0 is not None:
                    it = br * yields[t0]
            elif type_ == "sf":
                if t0 is not None:
                    sf = br * yields[t0]
            elif type_ == "beta-,n":
                if t0 is not None:
                    beta_neutron = br * yields[t0]

    # If there are other reactions than those above, they will
    # artificially be pushed to beta-. So lets renormalize the values
    # so they add to 1
    norm = np.sum([beta_m, beta_m_star, beta_ec, beta_ec_star, alpha, it, sf,
                   beta_neutron])
    if norm != 0.:
        beta_m_star /= norm
        beta_ec /= norm
        beta_ec_star /= norm
        alpha /= norm
        it /= norm
        sf /= norm
        beta_neutron /= norm

    return _write_card(lib_id, origen_nuclid, data_units, thalf, beta_m_star,
                       beta_ec, beta_ec_star, alpha, it, sf, beta_neutron,
                       data.decay_energy, abund, arcg, wrcg)


def _get_xs(lib, iso_name, xs_type, flux):
    if xs_type.endswith("*"):
        type_ = xs_type[:-1]
        m = 1
    else:
        type_ = xs_type
        m = 0
    return_val = 0.
    if iso_name in lib.isotopes:
        if lib.isotopes[iso_name].neutron_xs is not None:
            xs = lib.isotopes[iso_name].neutron_xs.get_xs(type_, "b", m)
            flux_tot = np.sum(flux)
            if xs is not None:
                if flux_tot == 0.:
                    return_val = np.sum(xs)
                else:
                    return_val = np.dot(xs, flux) / np.sum(flux)
    return return_val


def _write_xs_nfy(this, lib_id, iso_name, iso_type, flux):
    # This writes the specific xs and fy library for a specific isotope

    origen_nuclid = str(GND_to_origen_nucid(iso_name))

    # Get the xs data
    sng = _get_xs(this, iso_name, "(n,gamma)", flux)
    sn2n = _get_xs(this, iso_name, "(n,2n)", flux)
    sngx = _get_xs(this, iso_name, "(n,gamma)*", flux)
    sn2nx = _get_xs(this, iso_name, "(n,2n)*", flux)
    if iso_type == "actinide":
        snf_snp = _get_xs(this, iso_name, "fission", flux)
        sn3n_sna = _get_xs(this, iso_name, "(n,3n)", flux)
    else:
        snf_snp = _get_xs(this, iso_name, "(n,p)", flux)
        sn3n_sna = _get_xs(this, iso_name, "(n,a)", flux)

    tot_xs = sum([sng, sn2n, sngx, sn2nx, snf_snp, sn3n_sna])
    if tot_xs > 0.:
        has_xs = True
    else:
        has_xs = False

    # Now get the yield data
    # Origen has fixed fission yield isotopes so set those
    fiss_isos = ["Th232", "U233", "U235", "U238", "Pu239", "Pu241",
                 "Cm245", "Cf252"]
    yields = [0.0 for i in range(len(fiss_isos))]
    has_yield = -1.0
    if iso_type == "fp":
        # Only fission products have yields
        for i, fiss_iso in enumerate(fiss_isos):
            if fiss_iso in this.isotopes:
                fiss_iso_nfy = this.isotopes[fiss_iso].neutron_fission_yield
                if fiss_iso_nfy is not None:
                    if iso_name in fiss_iso_nfy:
                        # Then there is a yield for creating
                        # iso_name from fission in fiss_iso
                        yields[i] = fiss_iso_nfy[iso_name] * 100.0
                        has_yield = 1.0

    # The following is per Table 5.2
    if has_xs or has_yield > 0.:
        # First card
        card = _nlb_nuclide(lib_id, origen_nuclid) + _f2s(sng) + _f2s(sn2n) + \
            _f2s(sn3n_sna) + _f2s(snf_snp) + "\n" + " " * 11 + \
            _f2s(sngx) + _f2s(sn2nx)
        # To fit within 80 lines, the ORIGEN libraries print
        # YYN as -1.0 or 1.0, so lets just do that explicitly
        card += " {:2.1f}\n".format(has_yield)
        # Second card, if we need it
        if has_yield > 0.0:
            card += " {:>3d}".format(lib_id)
            for i, yield_ in enumerate(yields):
                card += _f2s(yield_)
                if i == 4:
                    card += "\n    "
            card += "\n"
    else:
        card = ""

    return card
