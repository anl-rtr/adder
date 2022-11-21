from dataclasses import dataclass, field

from adder.type_checker import *
from adder.data import atomic_mass, zam


EXISTING_ISOTOPES = {}


@dataclass
class Isotope:
    """This class contains relevant information about an isotope.

    Parameters
    ----------
    name : str
        The isotope name in GND format, e.g., "U235"
    xs_library : str
        Cross section library reference, i.e. "80c" if using MCNP.
    is_depleting : bool, optional
        Whether or not the isotope should be treated as depleting;
        defaults to True.

    Attributes
    ----------
    name : str
        The isotope name in GND format, e.g., "U235"
    Z : int
        Proton number; for example U-235's Z is 92.
    A : int
        Mass number; for example U-235's A is 235.
    M : int
        Metastable state; for example the ground state is 0.
    xs_library : str
        cross section library reference, i.e. "80c" if using MCNP.
    is_depleting : bool
        Whether or not the isotope should be treated as depleting

    """
    name: str
    xs_library: str
    is_depleting: bool = True
    Z: int = field(init=False)
    A: int = field(init=False)
    M: int = field(init=False)

    def __post_init__(self):
        # Do the relevant checks of values
        check_type("name", self.name, str)
        check_type("xs_library", self.xs_library, str)
        check_type("is_depleting", self.is_depleting, bool)
        # TODO: These may be not necessary given reasonableness of names, but
        # are left as this should have no major runtime impact and should
        # provide an easy point to add in pseudo-nuclide control

        Z, A, M = zam(self.name)
        check_type("Z", Z, int)
        check_greater_than("Z", Z, 0, equality=False)
        check_less_than("Z", Z, 118, equality=True)
        check_type("A", A, int)
        check_greater_than("A", A, 0, equality=True)
        check_less_than("A", A, 300, equality=True)
        check_value("M", M, [0, 1, 2, 3, 4])
        if A == 0 and M != 0:
            raise ValueError("Cannot provide M for elemental data!")

        self.Z, self.A, self.M = Z, A, M

    @property
    def atomic_mass(self):
        return atomic_mass(self.name)

    def __repr__(self):
        msg = "<Isotope {}, xs_lib: {}".format(self.name, self.xs_library) + \
            ", is_depleting: {}>".format(self.is_depleting)
        return msg

    def __hash__(self):
        # Control the hash so it is cheaper to create (its executed alot by the
        # isotope_factory), and to guarantee reproducibility even across
        # threads
        return hash((self.name, self.xs_library, self.is_depleting))


def isotope_factory(name, xs_library, is_depleting=True):
    """Creates a new isotope or returns a reference to a previously
    created isotope; this is used so we can treat isotopes as immutable
    to avoid initializing billions of each kind.

    Parameters
    ----------
    name : str
        The isotope name in GND format, e.g., "U235"
    xs_library : str
        Cross section library reference, i.e. "80c" if using MCNP.
    is_depleting : bool, optional
        Whether or not the isotope should be treated as depleting;
        defaults to True.

    Returns
    -------
    iso : Isotope
        Either a new isotope, or a reference to an already created isotope
    """

    iso_hash = hash((name, xs_library, is_depleting))
    if iso_hash in EXISTING_ISOTOPES:
        return EXISTING_ISOTOPES[iso_hash]
    else:
        iso = Isotope(name, xs_library, is_depleting)
        EXISTING_ISOTOPES[iso_hash] = iso
        return iso


def update_isotope_depleting_status(old_iso, new_is_depleting):
    return isotope_factory(old_iso.name, old_iso.xs_library, new_is_depleting)
