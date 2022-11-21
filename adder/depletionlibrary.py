from collections import OrderedDict
from copy import deepcopy
import os
import math

import h5py
import numpy as np
import scipy.sparse as sp

from adder.type_checker import *
from adder.constants import MAX_METASTABLE_STATE, FILETYPE_DEPLETION_LIBRARY, \
    VERSION_DEPLETION_LIBRARY
import adder.data

# Constants specific to obtaining data from ORIGEN2.2
# How many characters are in an ORIGEN library id
CHARACTERS_FOR_LIB_ID = 4
ORIGEN_TIME_INTS = {1: "s", 2: "m", 3: "hr", 4: "d", 5: "yr",
                    6: "stable", 7: "kyr", 8: "Myr", 9: "Gyr"}
ONE_GROUP_STRUCTURE = [0.0, 20.0]  # Units of MeV

# Isotopes in ORIGEN with fission yields
ORIGEN_FISS_YIELD_ISOS = ["Th232", "U233", "U235", "U238", "Pu239", "Pu241",
                          "Cm245", "Cf252"]

_TIME_CONV_CONSTS = {"s": 1., "m": 60., "hr": 3600., "d": 8.64E4,
                     "yr": 3.1536E7, "kyr": 3.1536E10, "Myr": 3.1536E13,
                     "Gyr": 3.1536E16}

_ALLOWED_ISOTOPE_TYPES = ["activation", "actinide", "fp"]

_DECAY_UNITS = ['s', 'm', 'hr', 'd', 'yr', 'kyr', 'Myr', 'Gyr']

    # The following are based on ENDF/B-VII.1 decay data
_DECAY_TYPES = \
    ["alpha", "n", "n,n", "p", "p,p", "beta-", "beta-,n",
        "beta-,n,n", "beta-,n,n,n", "beta-,n,n,n,n", "beta-,alpha",
        "beta-,beta-", "ec/beta+", "ec/beta+,alpha", "ec/beta+,p",
        "ec/beta+,p,p", "ec/beta+,sf", "it", "sf", "removal"]

_DECAY_SECONDARY_PARTICLES = {
    "alpha": [(1., "He4")],
    "p": [(1., "H1")],
    "p,p": [(2., "H1")],
    "beta-,alpha": [(1., "He4")],
    "ec/beta+,alpha": [(1., "He4")],
    "ec/beta+,p": [(1., "H1")],
    "ec/beta+,p,p": [(2., "H1")],
}

_RXN_UNITS = ['b', 'cm2']

# The following are based on ENDF/B-VII.1 decay data
_RXN_TYPES = \
    ['(n,gamma)', '(n,2n)', '(n,3n)', '(n,4n)', 'fission', '(n,p)',
        '(n,d)', '(n,t)', '(n,3He)', '(n,a)', '(n,2nd)', '(n,na)', '(n,3na)',
        '(n,n3a)', '(n,2na)', '(n,np)', '(n,n2a)', '(n,2n2a)', '(n,nd)',
        '(n,nt)', '(n,nHe-3)', '(n,nd2a)', '(n,nt2a)', '(n,2np)', '(n,3np)',
        '(n,n2p)', '(n,2a)', '(n,3a)', '(n,2p)', '(n,pa)', '(n,t2a)',
        '(n,d2a)', '(n,pd)', '(n,pt)', '(n,da)']

_RXN_NEUTRON_MULTIPLICITES = \
    {'(n,gamma)': 0., '(n,2n)': 2., '(n,3n)': 3., '(n,4n)': 4.,
        'fission': 2.43, '(n,p)': 0., '(n,d)': 0., '(n,t)': 0., '(n,3He)': 0.,
        '(n,a)': 0., '(n,2nd)': 2., '(n,na)': 1., '(n,3na)': 3.,
        '(n,n3a)': 1., '(n,2na)': 2., '(n,np)': 1., '(n,n2a)': 1.,
        '(n,2n2a)': 2., '(n,nd)': 1., '(n,nt)': 1., '(n,nHe-3)': 1.,
        '(n,nd2a)': 1., '(n,nt2a)': 1., '(n,2np)': 2., '(n,3np)': 3.,
        '(n,n2p)': 1., '(n,2a)': 0., '(n,3a)': 0., '(n,2p)': 0., '(n,pa)': 0.,
        '(n,t2a)': 0., '(n,d2a)': 0., '(n,pd)': 0., '(n,pt)': 0.,
        '(n,da)': 0.}

_RXN_SECONDARY_PARTICLES = {
    '(n,p)': [(1., 'H1')], '(n,d)': [(1., 'H2')], '(n,t)': [(1., 'H3')],
    '(n,3He)': [(1., 'He3')], '(n,a)': [(1., 'He4')],
    '(n,2nd)': [(1., 'H2')], '(n,na)': [(1., 'He4')],
    '(n,3na)': [(1., 'He4')], '(n,n3a)': [(3., 'He4')],
    '(n,2na)': [(1., 'He4')], '(n,np)': [(1., 'H1')],
    '(n,n2a)': [(2., 'He4')], '(n,2n2a)': [(2., 'He4')],
    '(n,nd)': [(1., 'H2')], '(n,nt)': [(1., 'H3')],
    '(n,nHe-3)': [(1., 'He3')], '(n,nd2a)': [(1., 'H2'), (1., 'He4')],
    '(n,nt2a)': [(1., 'H3'), (2., 'He4')], '(n,2np)': [(1., 'H1')],
    '(n,3np)': [(1., 'H1')], '(n,n2p)': [(1., 'H1')],
    '(n,2a)': [(2., 'He4')], '(n,3a)': [(3., 'He4')],
    '(n,2p)': [(2., 'H1')], '(n,pa)': [(1., 'H1'), (1., 'He4')],
    '(n,t2a)': [(1., 'H3'), (2., 'He4')],
    '(n,d2a)': [(1., 'H2'), (2., 'He4')],
    '(n,pd)': [(1., 'H1'), (1., 'H2')], '(n,pt)': [(1., 'H1'), (1., 'H3')],
    '(n,da)': [(1., 'H2'), (1., 'He4')]
}
# The types to consider when computing the total x/s
_RXN_TOTAL_TYPES = _RXN_TYPES
# The types to consider when computing the fission x/s
_RXN_FISSION_TYPES = ["fission"]
# The types to consider when computing the absorption x/s
_RXN_ABSORB_TYPES = ['(n,gamma)', 'fission', '(n,p)', '(n,d)', '(n,t)',
    '(n,3He)', '(n,a)', '(n,2a)', '(n,3a)', '(n,2p)', '(n,pa)', '(n,t2a)',
    '(n,d2a)', '(n,pd)', '(n,pt)', '(n,da)']


class IsotopeData(object):
    """This class contains the data necessary to describe an isotope in
    the depletion library. This isotopic description includes the cross
    sections, decay data, and fission product yield information.

    Parameters
    ----------
    name : str
        GND name of the isotope represented by this class

    Attributes
    ----------
    name : str
        GND name of the isotope represented by this class
    atomic_mass : float
        The atomic mass, in units of amu. For pseudo-nuclides, this will
        have a value of 1
    neutron_xs : ReactionData
        Neutron transition cross sections.
    neutron_fission_yield : YieldData
        Neutron-induced fission yields.
    decay : DecayData
        Radioactive decay transition data.
    removal : DecayData
        The chemical removal rate of this isotope; currently only
        applicable to MSR's
    """

    def __init__(self, name):
        self.name = name
        is_isotope = adder.data.is_isotope(self.name)

        if is_isotope:
            # Set the atomic mass
            amu = adder.data.atomic_mass(self.name)
            if amu is None:
                # Then this isotope is not present in the source data
                # for the atomic mass info (could be a pseudo-nuc)
                # if so, use the value of A
                _, a, _ = adder.data.zam(self.name)
                amu = float(a)
        else:
            amu = 1.
        self.atomic_mass = amu

        # Initialize the member data containers
        self.neutron_xs = None
        self.neutron_fission_yield = None
        self.decay = None
        self.removal = None

    def __repr__(self):
        return "<IsotopeData: {}>".format(self.name)

    def __deepcopy__(self, memo):
        # This method is called when copy.deepcopy() is called as a result
        # of DepletionLibrary.clone(). To save memory this 'deepcopy' will
        # actually be a deepcopy of only the neutron_xs and removal data.
        # The rest will be a reference to the original (i.e., a shallow copy)

        that = IsotopeData(self._name)
        that._atomic_mass = self._atomic_mass
        that._neutron_xs = deepcopy(self._neutron_xs)
        that._neutron_fission_yield = self._neutron_fission_yield
        that._decay = self._decay
        that._removal = deepcopy(self._removal)
        return that

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def atomic_mass(self):
        return self._atomic_mass

    @atomic_mass.setter
    def atomic_mass(self, atomic_mass):
        check_type("atomic_mass", atomic_mass, float)
        check_greater_than("atomic_mass", atomic_mass, 0., equality=False)
        self._atomic_mass = atomic_mass

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, decay):
        if decay is not None:
            check_type("decay", decay, DecayData)
        self._decay = decay

    @property
    def removal(self):
        return self._removal

    @removal.setter
    def removal(self, removal):
        if removal is not None:
            check_type("removal", removal, DecayData)
        self._removal = removal

    @property
    def neutron_fission_yield(self):
        return self._neutron_fission_yield

    @neutron_fission_yield.setter
    def neutron_fission_yield(self, neutron_fission_yield):
        if neutron_fission_yield is not None:
            check_type("neutron_fission_yield", neutron_fission_yield,
                       YieldData)
        self._neutron_fission_yield = neutron_fission_yield

    @property
    def neutron_xs(self):
        return self._neutron_xs

    @neutron_xs.setter
    def neutron_xs(self, neutron_xs):
        if neutron_xs is not None:
            check_type("neutron_xs", neutron_xs, ReactionData)
        self._neutron_xs = neutron_xs

    def get_total_removal_xs(self, units):
        if self.neutron_xs is not None:
            xs = self.neutron_xs.total_xs
            if self.neutron_xs.xs_units == units:
                return xs
            if units == "b":
                return np.copy(xs) * 1.E24
            elif units == "cm2":
                return np.copy(xs) * 1.E-24
        else:
            return None

    def get_total_decay_const(self, units, datatype="all"):
        # Set up the types to evaluate
        data = []
        if datatype == "all":
            data = [self.decay, self.removal]
        elif datatype == "decay":
            data = [self.decay]
        elif datatype == "removal":
            data = [self.removal]

        if all(v is None for v in data):
            return None

        const = 0.
        for datum in data:
            if datum is not None:
                if datum.half_life_units == units:
                    const += datum.decay_constant
                else:
                    const += datum.decay_constant * \
                        (_TIME_CONV_CONSTS[units] /
                         _TIME_CONV_CONSTS[datum.half_life_units])

        return const

    def to_hdf5(self, iso_group):
        """Adds this isotope to the HDF5 group"""

        # Write the isotope type information
        if self.decay is not None:
            decay_grp = iso_group.create_group("decay")
            self.decay.to_hdf5(decay_grp)

        if self.neutron_xs is not None:
            neutron_xs_grp = iso_group.create_group("neutron_xs")
            self.neutron_xs.to_hdf5(neutron_xs_grp)

        if self.neutron_fission_yield is not None:
            nfy_grp = iso_group.create_group("neutron_fission_yield")
            self.neutron_fission_yield.to_hdf5(nfy_grp)

        if self.removal is not None:
            removal_grp = iso_group.create_group("removal")
            self.removal.to_hdf5(removal_grp)

    @classmethod
    def from_hdf5(cls, iso_group, name):
        """Reads the Isotope Data from HDF5 file.

        Parameters
        ----------
        isotopic_group : h5py.Group
            The hdf5 group with this data
        name : str
            The name of the isotope

        """

        # Get the isotopic data we need to initialize the IsotopeData
        # class
        if "decay" in iso_group:
            decay = DecayData.from_hdf5(iso_group["decay"], name)
        if "neutron_xs" in iso_group:
            neutron_xs = ReactionData.from_hdf5(iso_group["neutron_xs"], name)
        if "neutron_fission_yield" in iso_group:
            nfy = YieldData.from_hdf5(iso_group["neutron_fission_yield"], name)
        if "removal" in iso_group:
            removal = DecayData.from_hdf5(iso_group["removal"], name)

        # Now we can create our object
        this = cls(name)

        if "decay" in iso_group:
            this.decay = decay
        else:
            this.decay = None
        if "neutron_xs" in iso_group:
            this.neutron_xs = neutron_xs
        else:
            this.neutron_xs = None
        if "neutron_fission_yield" in iso_group:
            this.neutron_fission_yield = nfy
        else:
            this.neutron_fission_yield = None
        if "removal" in iso_group:
            this.removal = removal
        else:
            this.removal = None

        return this


class DecayData(object):
    """Decay mode information for an isotope.

    Decay products are accessed via dictionary-like interface where the
    key is the decay type (e.g., "alpha" for an alpha decay) and the
    corresponding value is a 3-tuple of the branching ratio, a list of
    the decay products (including the secondary products, like He4 for
    an alpha decay), and the yield of each of these products.

    Parameters
    ----------
    half_life : float or None
        The half life of the nuclide in the units provided by units
    half_life_units : str
        The units of the half-life
    decay_energy : float
        Average energy deposited from decay in units of MeV

    Attributes
    ----------
    half_life : float or None
        The half life of the nuclide in the units provided by units
    half_life_units : str
        The units of the half-life
    decay_energy : float
        Average energy deposited from decay in units of MeV
    """

    def __init__(self, half_life, half_life_units, decay_energy):
        self.half_life = half_life
        self.half_life_units = half_life_units
        self.decay_energy = decay_energy
        self._products = {}

    @property
    def half_life(self):
        return self._half_life

    @half_life.setter
    def half_life(self, half_life):
        if half_life is not None:
            check_type("half_life", half_life, float)
            check_greater_than("half_life", half_life, 0., equality=True)
        self._half_life = half_life

    @property
    def decay_constant(self):
        if self._half_life is None:
            return 0.
        else:
            return math.log(2.) / self._half_life

    @property
    def decay_energy(self):
        return self._decay_energy

    @decay_energy.setter
    def decay_energy(self, decay_energy):
        check_type("decay_energy", decay_energy, float)
        self._decay_energy = decay_energy

    @property
    def half_life_units(self):
        return self._half_life_units

    @half_life_units.setter
    def half_life_units(self, half_life_units):
        check_value("half_life_units", half_life_units,
                    _DECAY_UNITS)
        self._half_life_units = half_life_units

    @property
    def num_types(self):
        return len(self._products)

    def __contains__(self, type_):
        return type_ in self._products

    def __iter__(self):
        return iter(self._products.values())

    def __getitem__(self, type_):
        if type_ in self._products:
            return self._products[type_]
        else:
            return None

    def keys(self):
        return iter(self._products.keys())

    def items(self):
        return iter(self._products.items())

    def values(self):
        return iter(self._products.values())

    def add_type(self, type_, br, targets, yields_=None, add_secondaries=True):
        # Adds data for the type; this function adds the secondary
        # particles for the user
        if type_ in self._products:
            msg = "Decay type {} already exists in the data!".format(type_)
            raise ValueError(msg)

        if type_ not in _DECAY_TYPES:
            msg = "Invalid decay type! {}".format(type_)
            raise ValueError(msg)

        check_type("br", br, (float, np.float64))
        check_greater_than("br", br, 0., equality=True)
        if isinstance(targets, str):
            # then we need to make a list out of it
            targets_ = [targets]
        else:
            check_iterable_type("targets", targets, str)
            targets_ = targets.copy()
        if yields_ is None:
            yields__ = [1.] * len(targets_)
        else:
            check_iterable_type("yields_", yields_, (float, np.float64))
            check_length("targets", targets_, len(yields_))
            yields__ = yields_.copy()
        check_type("add_secondaries", add_secondaries, bool)

        # Make sure the targets are isotopes
        for target in targets_:
            if target != "fission" and not adder.data.is_isotope(target):
                msg = "Invalid target name ({}) for reaction type {}"
                raise ValueError(msg.format(target, type_))

        # Now add in secondary products if not already included
        if add_secondaries:
            if type_ in _DECAY_SECONDARY_PARTICLES:
                s_data = _DECAY_SECONDARY_PARTICLES[type_]

                for s_yield, s_target in s_data:
                    yields__.append(s_yield)
                    targets_.append(s_target)

        # Build our product
        self._products[type_] = (br, targets_, yields__)

    def to_hdf5(self, group):
        """Writes the DecayData to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        if self.half_life is None:
            group.attrs["half_life"] = "None"
        else:
            group.attrs["half_life"] = self.half_life
        group.attrs["half_life_units"] = np.string_(self.half_life_units)
        group.attrs["decay_energy"] = self.decay_energy

        # Now we have to write the products, they get their own group
        for type_, values in self._products.items():
            # ec/beta+ (and in future? others) have a slash, which
            # can confuse hdf5 as it may look like a group path
            # so lets intercept and convert to a double underscore
            # which we will fix on re-read
            t_group = group.create_group(type_.replace("/", "__"))
            br, targets, yields_ = values
            t_group.attrs["branching_ratio"] = br
            t_group.create_dataset("targets",
                                   data=np.array([np.string_(t)
                                                  for t in targets]))
            t_group.create_dataset("yields", data=np.array(yields_))

    @classmethod
    def from_hdf5(cls, group, name):
        """Initializes a DecayData object from an opened HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
        name : str
            Isotope name

        Returns
        -------
        this : DecayData
            A DecayData object initialized from HDF5
        """

        half_life_tmp = group.attrs["half_life"]
        if half_life_tmp == "None":
            half_life = None
        else:
            half_life = float(half_life_tmp)
        half_life_units = group.attrs["half_life_units"].decode()
        decay_energy = float(group.attrs["decay_energy"])

        # Initialize the object
        this = cls(half_life, half_life_units, decay_energy)

        # Now get the products' data
        for type_ in group.keys():
            t_group = group[type_]
            br = float(t_group.attrs["branching_ratio"])
            targets = [s.decode() for s in t_group["targets"]]
            yields_ = [float(s) for s in t_group["yields"]]

            this.add_type(type_.replace("__", "/"), br, targets, yields_,
                          add_secondaries=False)

        return this


class ReactionData(object):
    """Flux-induced reaction information for an isotope.

    Reaction products are accessed via dictionary-like interface where
    the key is the reaction type (e.g., "(n,alpha)") and the
    corresponding value is a 4-tuple of an np.ndarray of the
    group-wise cross sections, a list of the target isotopes (including
    secondary products like He4 for an (n,alpha) reaction type), the
    yield of each of these targets, and the Q-value of the reaction
    in units of MeV.

    Parameters
    ----------
    xs_units : {"b", "cm2"}
        The units of the cross section
    num_groups : int
        The number of energy groups the data is present in

    Attributes
    ----------
    xs_units : {"b", "cm2"}
        The units of the cross section
    num_groups : int
        The number of energy groups the data is present in
    products : dict of (branching_ratio, List of GND targets, List of yields)
        Keyed by the type of interaction
    """

    def __init__(self, xs_units, num_groups):
        self.xs_units = xs_units
        self.num_groups = num_groups
        self._products = {}

    @property
    def xs_units(self):
        return self._xs_units

    @xs_units.setter
    def xs_units(self, xs_units):
        check_value("xs_units", xs_units, _RXN_UNITS)
        self._xs_units = xs_units

    @property
    def num_groups(self):
        return self._num_groups

    @num_groups.setter
    def num_groups(self, num_groups):
        check_type("num_groups", num_groups, int)
        check_greater_than("num_groups", num_groups, 0)
        self._num_groups = num_groups

    @property
    def num_types(self):
        return len(self._products)

    @property
    def total_xs(self):
        tot_rem_xs = np.zeros(self._num_groups)
        for type_ in _RXN_TOTAL_TYPES:
            if type_ in self._products:
                xs, _, _, _ = self._products[type_]
                tot_rem_xs += xs
        return tot_rem_xs

    def __contains__(self, type_):
        return type_ in self._products

    def __iter__(self):
        return iter(self._products.values())

    def __getitem__(self, type_):
        if type_ in self._products:
            return self._products[type_]
        else:
            return None

    def keys(self):
        return iter(self._products.keys())

    def items(self):
        return iter(self._products.items())

    def values(self):
        return iter(self._products.values())

    def add_type(self, type_, xs_units, xs, targets=None, yields_=None,
                 add_secondaries=True, q_value=0.):
        # Adds data for the type; this function adds the secondary
        # particles for the user
        if type_ in self._products:
            msg = "XS type {} already exists in the data!".format(type_)
            raise ValueError(msg)

        if type_ not in _RXN_TYPES:
            msg = "Invalid XS type! {}".format(type_)
            raise ValueError(msg)

        check_value("xs_units", xs_units, _RXN_UNITS)
        check_iterable_type("xs", xs, (float, np.float64))
        for xs_val in xs:
            check_greater_than("xs value", xs_val, 0., equality=True)
        check_length("xs", xs, self._num_groups)
        if type_ == "fission" and targets is None:
            targets_ = ["fission"]
        elif isinstance(targets, str):
            # then we need to make a list out of it
            targets_ = [targets]
        else:
            check_iterable_type("targets", targets, str)
            targets_ = targets.copy()
        if yields_ is None:
            yields__ = [1.] * len(targets_)
        else:
            check_iterable_type("yields_", yields_, (float, np.float64))
            check_length("targets", targets_, len(yields_))
            yields__ = yields_.copy()
        check_type("add_secondaries", add_secondaries, bool)
        check_type("q_value", q_value, (float, np.float64))

        # Make sure the targets are isotopes
        for target in targets_:
            if target != "fission" and not adder.data.is_isotope(target):
                msg = "Invalid target name ({}) for reaction type {}"
                raise ValueError(msg.format(target, type_))

        # Now add in secondary products if not already included
        if add_secondaries:
            if type_ in _RXN_SECONDARY_PARTICLES:
                s_data = _RXN_SECONDARY_PARTICLES[type_]

                for s_yield, s_target in s_data:
                    yields__.append(s_yield)
                    targets_.append(s_target)

        # Convert xs units
        if self._xs_units != xs_units:
            if xs_units == "b":
                # Then self.xs_units is cm2 and we need to be consistent
                # with that
                xs = np.copy(np.asarray(xs)) * 1.E-24
            elif xs_units == "cm2":
                # Then self.xs_units is b and we need to be consistent
                # with that
                xs = np.copy(np.asarray(xs)) * 1.E24
        else:
            xs = np.array(xs)

        # Build our product
        self._products[type_] = (xs, targets_, yields__, q_value)

    def get_xs(self, type_, output_units, meta_state=0):
        # Finds the cross section to a desired metastable state in the
        # requested units
        if self._xs_units == output_units:
            conv_const = 1.
        elif output_units == "b":
            conv_const = 1.E24
        elif output_units == "cm2":
            conv_const = 1.E-24

        if type_ in self._products:
            xs, targets, yields_, _ = self._products[type_]
            for t, target in enumerate(targets):
                if target != "fission":
                    _, _, m = adder.data.zam(target)
                    if m == meta_state:
                        return xs * yields_[t] * conv_const
                else:
                    return xs * yields_[t] * conv_const
            # If we get here, we didnt find it
            return None

        else:
            return None

    def to_hdf5(self, group):
        """Writes the ReactionData to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        group.attrs["xs_units"] = np.string_(self._xs_units)
        group.attrs["num_groups"] = self._num_groups

        # Now we have to write the products, they get their own group
        for type_, values in self._products.items():
            t_group = group.create_group(type_)
            xs, targets, yields_, q_value = values
            t_group.create_dataset("xs", data=xs)
            t_group.create_dataset("targets",
                                   data=np.array([np.string_(t)
                                                  for t in targets]))
            t_group.create_dataset("yields", data=np.array(yields_))
            t_group.attrs["q_value"] = q_value

    @classmethod
    def from_hdf5(cls, group, name):
        """Initializes a ReactionData object from an opened HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
        name : str
            Isotope name

        Returns
        -------
        this : ReactionData
            A ReactionData object initialized from HDF5
        """
        xs_units = group.attrs["xs_units"].decode()
        num_groups = int(group.attrs["num_groups"])

        # Initialize the object
        this = cls(xs_units, num_groups)

        # Now get the products' data
        for type_ in group.keys():
            t_group = group[type_]
            xs = t_group["xs"][()]
            targets = [s.decode() for s in t_group["targets"]]
            yields_ = [float(s) for s in t_group["yields"]]
            q_value = t_group.attrs["q_value"]

            this.add_type(type_, xs_units, xs, targets, yields_,
                          add_secondaries=False, q_value=q_value)

        return this


class YieldData(object):
    """This class stores the relevant probabilities for an isotope to
    transition to another isotope via complex outgoing distributions
    such as fission.

    """

    def __init__(self):
        self._products = {}

    @property
    def num_isotopes(self):
        return len(self._products)

    def __contains__(self, iso):
        return iso in self._products

    def __getitem__(self, iso):
        if iso in self._products:
            return self._products[iso]
        else:
            return None

    def __iter__(self):
        return iter(self._products.values())

    def keys(self):
        return iter(self._products.keys())

    def items(self):
        return iter(self._products.items())

    def values(self):
        return iter(self._products.values())

    def add_isotope(self, iso, yield_):
        """Adds an isotope to the YieldData object
        """

        check_type("iso", iso, str)
        check_type("yield_", yield_, (float, np.float64))
        self._products[iso] = yield_

    def to_hdf5(self, group):
        """Writes the TransitionData to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        isos = np.array([np.string_(k) for k in self._products.keys()])
        yields_ = np.array([v for v in self._products.values()])

        # And now we can write them
        group.create_dataset("isotopes", data=isos)
        group.create_dataset("yields", data=yields_)

    @classmethod
    def from_hdf5(cls, group, name, num_groups=None):
        """Initializes a TransitionData object from an opened HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
        name : str
            Isotope name
        num_groups : int, optional
            The number fo groups in the library; defaults to None.

        Returns
        -------
        this : TransitionData
            A TransitionData subclass object initialized from HDF5
        """

        this = cls()

        # Now get the channels and the channel_data
        isotopes = [s.decode() for s in group["isotopes"]]
        yields_ = group["yields"][()]

        # Make sure channels and channel_data are same length
        if len(isotopes) != len(yields_):
            raise ValueError("'isotopes' and 'yields' must have same size!")

        for i in range(len(isotopes)):
            this._products[isotopes[i]] = yields_[i]
        return this


class DepletionLibrary(object):
    """This class contains the data necessary for performing depletion
    within ADDER. As such it includes the isotopic multi-group
    transition cross sections, the decay data, fission product yield
    information.

    This library can be read from an HDF5 group and output to relevant
    ASCII formats such as the ORIGEN2.2 format for linking to the
    depletion solver.

    Parameters
    ----------
    name : str
        Descriptive name of the library.
    neutron_group_structure : Iterable of float
        The energy group structure used for incident neutron data,
        ordered from lowest to highest in units of MeV

    Attributes
    ----------
    name : str
        Descriptive name of the library.
    isotopes : dict of IsotopeData
        The isotope names are keys and the values are the IsotopeData
        objects.
    isotope_indices : dict
        A mapping of the isotope names to the index in the resultant
        depletion matrix.
    inverse_isotope_indices : dict
        A mapping of the index in the resultant depletion matrix to the
        isotope names.
    initial_isotopes : set of str
        The names of isotopes present in this library as it exists in the
        reference data; that is, before any modifications are made to handle
        missing isotopes from the neutronics model.
    neutron_group_structure : np.ndarray of float
        The energy group structure used for incident neutron data,
        ordered from lowest to highest in units of MeV
    num_neutron_groups : int
        The number of groups used for incident neutron data
    """

    def __init__(self, name, neutron_group_structure):
        self.name = name
        self.neutron_group_structure = neutron_group_structure
        # Initialize data structs relied on when data is populated
        self.isotopes = OrderedDict()
        self._isotopes_ordered = False
        self.isotope_indices = OrderedDict()
        self.inverse_isotope_indices = OrderedDict()
        self.initial_isotopes = set()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def neutron_group_structure(self):
        return self._neutron_group_structure

    @neutron_group_structure.setter
    def neutron_group_structure(self, neutron_group_structure):
        check_iterable_type("neutron_group_structure",
                            neutron_group_structure, float)
        self._neutron_group_structure = neutron_group_structure

    @property
    def num_neutron_groups(self):
        return len(self._neutron_group_structure) - 1

    @property
    def num_isotopes(self):
        return len(self.isotopes)

    def __repr__(self):
        return "<DepletionLibrary: {}>".format(self.name)

    def clone(self, new_name=None):
        """Create a clone of this library, assigning a new name if
        requested"""

        if new_name is not None:
            check_type("new_name", new_name, str)

        clone = deepcopy(self)
        if new_name is not None:
            clone.name = new_name

        return clone

    def finalize_library(self):
        """Performs operations necessary after manual-creation of a library
        or reading from a file. This must be called manually when a library
        is created by hand, otherwise it is called by the file-loading
        initializers."""

        for iso_name in self.isotopes.keys():
            # Add the isotope name to the initial isotopes list
            self.initial_isotopes.add(iso_name)

    def check_library(self):
        """Makes sure the library contains all isotopes needed to
        represent the chain. These results are returned as a string
        so the calling class can write to the log or screen.
        """

        # And get our isotope ordering
        if not self._isotopes_ordered:
            self.set_isotope_indices()

        msgs = []

        isos_to_add = set()

        for name in self.isotope_indices.keys():
            decay = self.isotopes[name].decay
            if decay is None:
                continue
            # Check decay
            child_names, child_values = self._get_decay_products(name)
            msg_template = \
                "The {} in the {}->{} {} decay is not present in the " + \
                "library; a stable, no cross section version was added to " + \
                "preserve atoms."
            for c_name, c_val in zip(child_names, child_values):
                if c_name != "fission" and c_name not in self.isotope_indices:
                    if c_val > 0.:
                        # Find the first relevant decay and add to msg
                        for type_ in decay.keys():
                            _, targets, _ = decay[type_]
                            if c_name in targets:
                                msgs.append(msg_template.format(c_name, name,
                                                                c_name, type_))
                                break
                        # Remember to add this to the library
                        isos_to_add.add(c_name)

            # Check flux-induced rxns
            n_xs = self.isotopes[name].neutron_xs
            if n_xs is None:
                continue
            flux = np.ones(self.num_neutron_groups)
            child_names, child_values = self._get_burn_products(name, flux)
            msg_template = "The {} in the {}->{} {} transmutation is not " + \
                "present in the library; a stable, no cross section " + \
                "version was added to preserve atoms."
            for c_name, c_val in zip(child_names, child_values):
                if c_name != "fission" and c_name not in self.isotope_indices:
                    if c_val > 0.:
                        # Find the first relevant decay and add to msg
                        for type_ in n_xs.keys():
                            _, targets, _, _ = n_xs[type_]
                            if c_name in targets:
                                msgs.append(msg_template.format(c_name, name,
                                                                c_name, type_))
                                break
                        # Remember to add this to the library
                        isos_to_add.add(c_name)

        # Now add the isotopes to the library
        stable = DecayData(None, "s", 0.)
        noxs = ReactionData("b", self.num_neutron_groups)
        for iso in isos_to_add:
            self.add_isotope(iso, xs=noxs, decay=stable)
        if len(isos_to_add) > 0:
            self.set_isotope_indices()

        return msgs

    def _get_decay_products(self, parent, output_unit='s'):
        """Finds the children of all of parent's reaction channels

        Note that since this function is only called internally to
        build depletion matrices, type-checking of input parameters
        is not performed.

        Parameters
        ----------
        parent : str
            The name of the parent isotope
        output_unit : str, optional
            The units requested; defaults to 's' for seconds

        Returns
        -------
        child_names : List of str
            The children of this isotope
        child_values : List of float
            The transition probabilities for this isotope
        """

        decay = self.isotopes[parent].decay
        rem = self.isotopes[parent].removal
        nfy = self.isotopes[parent].neutron_fission_yield
        fission_keys = ["sf", "ec/beta+,sf"]

        child_names = []
        child_values = []

        decay_lambda_ = \
            self.isotopes[parent].get_total_decay_const(output_unit, "decay")
        rem_lambda_ = self.isotopes[parent].get_total_decay_const(output_unit,
                                                                  "removal")
        if decay_lambda_ is None and rem_lambda_ is None:
            return child_names, child_values

        # Gather the target information from each of decay and removal
        for src, lambda_ in zip([decay, rem], [decay_lambda_, rem_lambda_]):
            if src is None:
                continue
            for child, (br, targets, yields_) in src.items():
                if br <= 0.:
                    continue
                # Get the value to be multiplied by the yields
                # (lambda * branch ratio)
                rxn_val = lambda_ * br

                for t, target in enumerate(targets):
                    # Handle the fission reaction
                    rxn_yield = rxn_val * yields_[t]
                    if child in fission_keys:
                        for fission_child, fission_yield in nfy.items():
                            child_names.append(fission_child)
                            child_values.append(rxn_yield * fission_yield)

                    else:
                        # The direct product is simpler
                        child_names.append(target)
                        child_values.append(rxn_yield)

        return child_names, child_values

    def _get_burn_products(self, parent, flux, output_unit="b"):
        """Finds the children of all of parent's reaction channels

        Note that since this function is only called internally to
        build depletion matrices, type-checking of input parameters
        is not performed.

        Parameters
        ----------
        parent : str
            The name of the parent isotope
        flux : np.ndarray
            The groupwise flux
        output_unit : str, optional
            The units requested; defaults to "b" for barns

        Returns
        -------
        child_names : List of str
            The children of this isotope
        child_values : List of float
            The transition probabilities for this isotope
        """

        child_names = []
        child_values = []

        src = self.isotopes[parent].neutron_xs
        if src is None:
            return child_names, child_values
        nfy = self.isotopes[parent].neutron_fission_yield
        fission_keys = ["fission"]

        # unit conversion constant
        if src.xs_units == output_unit:
            conversion = 1.0
        else:
            if output_unit == "cm2":
                conversion = 1.E-24
            elif output_unit == "b":
                conversion = 1.E24

        # Gather the target information
        for _, (xs, targets, yields_, _) in src.items():
            # Check to see if the xs is 0 over all groups
            if not np.any(xs):
                continue
            rxn_val = np.dot(xs, flux) * conversion
            for t, target in enumerate(targets):
                # Handle the fission reaction
                rxn_yield = rxn_val * yields_[t]
                if target in fission_keys:
                    for fission_child, fission_yield in nfy.items():
                        child_names.append(fission_child)
                        child_values.append(rxn_yield * fission_yield)
                else:
                    # The direct product is simpler
                    child_names.append(target)
                    child_values.append(rxn_yield)

        return child_names, child_values

    def set_isotope_indices(self):
        # We will use the a the NUCID from ORIGEN as a good way to order
        # we want the highest nucid to be first
        # GND_to_origen_nucid
        # Reset the isotope indices map
        self.isotope_indices = OrderedDict()

        # We need an array of iso names and an array of nucids
        # We then sort by nucids and rearrange iso names accordingly
        # Psuedo nuclides will be last, and thus need a small number
        # we will just use negative numbers
        _pseudo_counter = 0

        def _get_index(iso_name, counter):
            if adder.data.is_isotope(iso_name):
                # Note that this function skips over isotopes with
                # metastable states >= MAX_METASTABLE_STATE because
                # these represent isotopes removed via chemical means
                # (i.e., for MSRs) and so for efficiency these all go
                # at the end
                return GND_to_origen_nucid(iso_name)
            else:
                counter -= 1
                return counter

        iso_names = np.array([k for k in self.isotopes.keys()])
        nucids = np.array([_get_index(k, _pseudo_counter)
                           for k in self.isotopes.keys()])

        # Get the sorted indices of nucids, in increasing order
        sorted_indices = nucids.argsort()

        # Now store our map and its inverse
        self.inverse_isotope_indices = OrderedDict()
        for i, key in enumerate(iso_names[sorted_indices]):
            self.isotope_indices[key] = i
            self.inverse_isotope_indices[i] = key

        # Store that we have updated the ordering
        self._isotopes_ordered = True

    def set_atomic_mass_vector(self):
        """Establishes the atomic_mass_vector property, a numpy ndarray
        of the atomic masses for each isotope. These entries are ordered
        consistent with the number density vector which would be used
        with the depletion matrix.
        """
        if not self._isotopes_ordered:
            self.set_isotope_indices()

        self.atomic_mass_vector = np.zeros(self.num_isotopes)
        for iso_name, index in self.isotope_indices.items():
            self.atomic_mass_vector[index] = \
                self.isotopes[iso_name].atomic_mass

    def build_decay_matrix(self, time_units="s"):
        """This builds the depletion matrix that contains only the
        components representing radiaoctive decay.

        This is treated separately since this quantity does not
        depend on flux and thus can be pre-computed"""

        # And get our isotope ordering
        if not self._isotopes_ordered:
            self.set_isotope_indices()

        # Initialize our decay matrix, D
        D = np.zeros((self.num_isotopes, self.num_isotopes),
                        dtype=np.float64)

        # Now we go in and add the components
        for name, i in self.isotope_indices.items():
            # Get the total removal and place in the diagonal
            lambda_removal = \
                self.isotopes[name].get_total_decay_const(time_units,
                                                            "all")

            if lambda_removal is None:
                continue

            D[i, i] -= lambda_removal

            # Now get the children of this isotope and place in the
            # appropriate locations
            child_names, child_values = \
                self._get_decay_products(name, output_unit=time_units)

            # Now progress through each of these children, find
            # their index, and accrue the transmutation value
            for c_name, c_val in zip(child_names, child_values):
                if c_name in self.isotope_indices:
                    j = self.isotope_indices[c_name]
                    D[j, i] += c_val

        return D

    def build_depletion_matrix(self, flux, matrix_format="csr",
        dk_matrix=None):
        """Build the A matrix used for depletion from the library info

        Parameters
        ----------
        flux : np.ndarray
            The group-wise flux to use.
        matrix_format : {"csr", "csc", "dense"}, optional
            The matrix format to keep the matrix in; defaults to "csr"
        dk_matrix : None or np.ndarray
            The pre-computed decay matrix, if available

        Returns
        -------
        A_in_format : scipy.csr_matrix, scipy.csc_matrix, or np.ndarray
            The depletion matrix A in the chosen format
        """

        check_iterable_type("flux", flux, float)
        check_length("flux", flux, length_min=self.num_neutron_groups)
        check_value("matrix_format", matrix_format, ("csr", "csc", "dense"))

        # Make sure the decay matrix is calculated
        if dk_matrix is None:
            decay_matrix = self.build_decay_matrix()
        else:
            if sp.issparse(dk_matrix):
                decay_matrix = dk_matrix.todense()
            else:
                decay_matrix = dk_matrix

        # Now we can move on to building A; start by using decay matrix
        A = np.copy(decay_matrix)
        # Now we go in and add the components
        for name, i in self.isotope_indices.items():
            iso = self.isotopes[name]
            # First the diagonal (transmute from i to any other)
            xs = iso.get_total_removal_xs("cm2")
            if xs is None:
                continue
            else:
                val = np.dot(xs, flux)
            A[i, i] -= val

            # Now get the children of this isotope and place in the
            # appropriate locations
            child_names, child_values = \
                self._get_burn_products(name, flux, output_unit="cm2")

            # Now progress through each of these children, find
            # their index, and accrue the transmutation value
            for c_name, c_val in zip(child_names, child_values):
                if c_name in self.isotope_indices:
                    j = self.isotope_indices[c_name]
                    A[j, i] += c_val

        if matrix_format == "csr":
            A_in_format = sp.csr_matrix(A, dtype=np.float64)
        elif matrix_format == "csc":
            A_in_format = sp.csc_matrix(A, dtype=np.float64)
        else:
            A_in_format = A

        return A_in_format

    def add_isotope(self, iso_name, xs=None, nfy=None, decay=None):
        """This method adds the data for an isotope.

        Parameters
        ----------
        iso_name : str
            Isotope name in GND format (e.g., "Am241_m1")
        xs : ReactionData or None, optional
            Neutron transition cross sections; if None, then it is
            assumed this data is not present. Defaults to None.
        nfy : YieldData or None, optional
            Neutron-induced fission yields; if None, then it is
            assumed this data is not present. Defaults to None.
        decay : DecayData or None, optional
            Radioactive decay transition matrix; if None, then it is
            assumed this data is not present. Defaults to None.
        """

        check_type("iso_name", iso_name, str)
        if xs:
            check_type("xs", xs, ReactionData)
        if nfy:
            check_type("nfy", nfy, YieldData)
        if decay:
            check_type("decay", decay, DecayData)

        if xs or nfy or decay:
            if iso_name in self.isotopes:
                msg = "{} already in exists in Isotopes!".format(iso_name)
                raise ValueError(msg)

            # Work the specific datatypes
            if xs:
                # Check that the groups are consistent with expectations
                if xs.num_groups != self.num_neutron_groups:
                    msg = "xs groups and DepletionLibrary groups do not match!"
                    raise ValueError(msg)

            isotope = IsotopeData(iso_name)
            if xs:
                isotope.neutron_xs = xs
            if nfy:
                isotope.neutron_fission_yield = nfy
            if decay:
                isotope.decay = decay

            # Finished all the work, safe to store the isotope
            self.isotopes[iso_name] = isotope

            # Reset the flags about precomputed components
            self._isotopes_ordered = False

    def get_micro_neutron_xs(self, iso_name, rxn_type):
        """Finds the microscopic xs for a given isotope. If the isotope
        does not exist or the rxn_type does not exist, a zero G vector
        is returned."""

        micro_xs = np.zeros(self.num_neutron_groups)

        # Now if we do not have this isotope, the xs is 0
        if iso_name not in self.isotopes:
            return micro_xs

        iso_xs = self.isotopes[iso_name].neutron_xs
        if iso_xs is None:
            return micro_xs

        # Next, if this isotope has no xs data, then the xs is 0
        if iso_xs.num_types == 0:
            return micro_xs

        # Absorption requires special treatment because we need to take
        # out the multiplicity part
        if rxn_type == "absorb":
            # First we add up the straightforward absorption channels
            for type_ in _RXN_ABSORB_TYPES:
                if type_ in iso_xs.keys():
                    xs, _, _, _ = iso_xs[type_]
                    micro_xs += xs

            # And now we take away the multiplicity part
            # e.g., if (n,Xn) we subtract (X - 1) * sig_(n,2n)
            # where it is (X - 1) to account for the replacement of the
            # incident neutron with one of the outgoing neutrons
            for type_, X in _RXN_NEUTRON_MULTIPLICITES.items():
                if type_ in iso_xs.keys() and type_ != "fission" and X > 0.:
                    xs, _, _, _ = iso_xs[type_]
                    micro_xs -= (X - 1.) * xs
        elif rxn_type == "nu-fission":
            # nu-fission is only used for determining important isotopes
            # the fission multiplicities includes a value of 2.43.
            # This 2.43 does not need to be exact.
            # The problem is that it is not stored in ORIGEN's library
            # and so it would need to be provided from elsewhere; until
            # I have that capability incorporated this constant is used
            for type_ in _RXN_FISSION_TYPES:
                if type_ in iso_xs.keys():
                    xs, _, _, _ = iso_xs[type_]
                    micro_xs += _RXN_NEUTRON_MULTIPLICITES[type_] * xs
        else:
            if rxn_type == "total":
                types = _RXN_TOTAL_TYPES
            elif rxn_type == "fission":
                types = _RXN_FISSION_TYPES
            else:
                types = [rxn_type]
            for type_ in types:
                if type_ in iso_xs.keys():
                    xs, _, _, _ = iso_xs[type_]
                    micro_xs += xs

        return micro_xs

    def get_macro_neutron_xs(self, iso_names, iso_concentrations, rxn_type):
        """Finds the macroscopic xs for a given set of isotopes and
        their concentrations for a requested rxn_type. If the isotopes
        do not exist or the rxn_type does not exist, a zero G vector is
        returned."""

        check_length("iso_names length", iso_names, len(iso_concentrations))

        macro_xs = np.zeros(self.num_neutron_groups)

        for iso_name, conc in zip(iso_names, iso_concentrations):
            macro_xs += conc * self.get_micro_neutron_xs(iso_name, rxn_type)

        return macro_xs

    def get_1g_micro_xs(self, iso_name, rxn_type, flux,
                        leave_as_rxn_rate=False):
        """Compute the one-group cross section for a given isotope and
        reaction channel subject to a provided flux.

        Parameters
        ----------
        iso_name : str
            The isotope name
        rxn_type : str
            The reaction channel name
        flux : Iterable of float
            The flux with the same number of groups as the dataset.
            The group structure itself is not known or checked.
        leave_as_rxn_rate : bool
            Whether or not to divide by the flux (i.e., leave as a
            reaction rate or a cross section)

        Returns
        -------
        xs_or_rxn : float
            The one-group collapsed reaction rate or cross section of
            the type specified by rxn_type
        """

        check_iterable_type("flux", flux, float)
        check_length("flux", flux, self.num_neutron_groups)
        arr_flux = np.asfarray(flux)

        if iso_name in self.isotopes:
            mg_xs = self.get_micro_neutron_xs(iso_name, rxn_type)
            xs_or_rxn = np.dot(mg_xs, arr_flux)
            if leave_as_rxn_rate:
                return xs_or_rxn
            else:
                sum_flux = np.sum(arr_flux)
                if sum_flux <= 0.0:
                    msg = "Flux array cannot have a zero or negative sum!"
                    raise ValueError(msg)
                return xs_or_rxn / sum_flux
        else:
            return 0.0

    def get_1g_macro_xs(self, iso_names, iso_concentrations, rxn_type,
                        flux, leave_as_rxn_rate=False):
        """Compute the one-group cross section for a given set of
        isotopes and their concentrations for a requested
        rxn_type reaction channel subject to a provided flux.

        Parameters
        ----------
        iso_names : iterable of str
            The isotope names
        iso_concentrations : Iterable of float
            The isotope concentrations
        rxn_type : str
            The reaction channel name
        flux : Iterable of float
            The flux with the same number of groups as the dataset.
            The group structure itself is not known or checked.
        leave_as_rxn_rate : bool
            Whether or not to divide by the flux (i.e., laeve as a
            reaction rate or a cross section)

        Returns
        -------
        xs_or_rxn : float
            The one-group collapsed reaction rate or cross section of
            the type specified by rxn_type
        """

        check_iterable_type("flux", flux, float)
        check_length("flux", flux, self.num_neutron_groups)
        arr_flux = np.asfarray(flux)

        mg_xs = self.get_macro_neutron_xs(iso_names, iso_concentrations,
                                          rxn_type)
        xs_or_rxn = np.dot(mg_xs, arr_flux)
        if leave_as_rxn_rate:
            return xs_or_rxn
        else:
            sum_flux = np.sum(arr_flux)
            if sum_flux <= 0.0:
                msg = "flux array cannot have a zero or negative sum!"
                raise ValueError(msg)
            return xs_or_rxn / sum_flux

    def get_composition_Q_fiss_rate(self, iso_names, iso_concentrations, flux,
                                    Q_method="correlation"):
        """Computes the total energy release rate and the total fission
        rate for this material.

        This is used when determining the reactor's average recoverable
        energy per fission.

        This calculation implicitly makes the assumption that only
        materials that are depleting are producing our fissions and thus
        non-depleting materials are ignored in the Q determination.

        The calculation also relies upon the ORIGEN2.2 correlation for
        the recoverable energy per fission for actinides.

        Parameters
        ----------
        iso_names : iterable of str
            The isotope names
        iso_concentrations : Iterable of float
            The isotope concentrations
        flux : Iterable of float
            The flux with the same number of groups as the dataset.
            The group structure itself is not known or checked.
        method : {"correlation"}
            The method to use; correlation uses the ORIGEN2.2
            methodology. This is the only currently supported method.

        Returns
        -------
        comp_Q : float
            The total energy release rate for this composition
        comp_fiss_rate :
            The total fission rate for this composition
        """

        check_length("iso_names length", iso_names, len(iso_concentrations))

        comp_fiss_rate = 0.
        comp_Q = 0.
        for iso_name, concentration in zip(iso_names, iso_concentrations):
            iso_fiss_rate = concentration * \
                self.get_1g_micro_xs(iso_name, "fission", flux,
                                     leave_as_rxn_rate=True)
            comp_fiss_rate += iso_fiss_rate
            iso_Q = self._get_Qrec(iso_name, flux)
            comp_Q += iso_Q * iso_fiss_rate

        return comp_Q, comp_fiss_rate

    def _get_Qrec(self, iso_name, flux, method="correlation"):
        """Compute the total Q-recoverable for a requested isotope in
        units of MeV.

        Parameters
        ----------
        iso_name : str
            The isotope name
        flux : Iterable of float
            The flux with the same number of groups as the dataset.
            The group structure itself is not known or checked.
        method : {"correlation"}
            The method to use; correlation uses the ORIGEN2.2
            methodology. This is the only currently supported method.

        Returns
        -------
        Qrec : float
            The recoverable energy for this isotope in units of MeV
        """

        if iso_name in self.isotopes:
            if method == "correlation":
                # Use the ORIGEN2.2 recoverable fission energy release
                # correlation for the isotope
                Z, A, _ = adder.data.zam(iso_name)
                iso_Q = 1.29927E-3 * Z * Z * np.sqrt(A) + 33.12
            else:
                check_iterable_type("flux", flux, float)
                check_length("flux", flux, self.num_neutron_groups)
                raise NotImplementedError("This has not yet been implemented")
                # arr_flux = np.asfarray(flux)
        else:
            iso_Q = 0.

        return iso_Q

    def to_hdf5(self, filename_or_group, mode='w', revised_name=None):
        """Writes the DepletionLibrary to HDF5 file.

        Parameters
        ----------
        filename_or_group : str or h5py.Group
            Either the file to write to, or the root HDF5 group
        mode : {'r', 'r+', 'w', 'w-', 'x', 'a'}
            Write mode for the H5 file
        revised_name : str or None, optional
            If not desiring to use this name,

        """

        check_value("mode", mode, ['r', 'r+', 'w', 'w-', 'x', 'a'])
        if revised_name is not None:
            check_type("revised_name", revised_name, str)

        if isinstance(filename_or_group, str):
            root = h5py.File(filename_or_group, mode)
        elif isinstance(filename_or_group, h5py.Group):
            root = filename_or_group
        else:
            raise ValueError("Invalid filename_or_group!")

        # Check the filetype, or write it if it doesnt exist yet
        if 'filetype' in root.attrs:
            # Then this file was not just made, ensure this is a
            # matching filetype and version
            check_filetype_version(root, FILETYPE_DEPLETION_LIBRARY,
                                   VERSION_DEPLETION_LIBRARY)
        else:
            # Then we have to write the file status.
            root.attrs['filetype'] = np.string_(FILETYPE_DEPLETION_LIBRARY)
            root.attrs['version'] = [VERSION_DEPLETION_LIBRARY, 0]

        # Create the group for this object
        if revised_name is None:
            name = self.name
        else:
            name = revised_name
        group = root.create_group(name)

        # Add the energy group structure data in datasets
        group.create_dataset("neutron_group_structure",
                             data=self.neutron_group_structure)

        # Add the isotopic information
        isotopic_group = group.create_group("isotopic_data")
        for iso_name in self.isotopes:
            iso_group = isotopic_group.create_group(iso_name)
            self.isotopes[iso_name].to_hdf5(iso_group)

    @classmethod
    def from_hdf5(cls, filename_or_group, name):
        """Reads the DepletionLibrary from HDF5 file.

        Parameters
        ----------
        filename_or_group : str or h5py.Group
            Either the file to read from, or the root HDF5 group

        """

        if isinstance(filename_or_group, str):
            root = h5py.File(filename_or_group, "r")
        elif isinstance(filename_or_group, h5py.Group):
            root = filename_or_group
        else:
            raise ValueError("Invalid filename_or_group!")

        # Check the filetype and version
        check_filetype_version(root, FILETYPE_DEPLETION_LIBRARY,
                               VERSION_DEPLETION_LIBRARY)

        # Create the group for this object
        group = root[name]

        # Get the simple attributes to the group
        neutron_group_structure = np.array(group["neutron_group_structure"],
                                           dtype=float)

        # Now we can create our object
        this = cls(name, neutron_group_structure)

        # Now lets get our isotopic data
        isotopic_group = group["isotopic_data"]
        for iso_name, iso_group in isotopic_group.items():
            iso = IsotopeData.from_hdf5(iso_group, iso_name)
            this.isotopes[iso_name] = iso

        this.finalize_library()

        return this

    @classmethod
    def from_origen(cls, xs_filename, decay_filename, xs_lib_ids,
                    decay_lib_ids, new_name=None, var_actinide_lib=None):
        """This method obtains, from an ORIGEN ASCII file, the
        information contained within an ORIGEN 2.2 decay or cross section
        library.

        Parameters
        ----------
        xs_filename : str
            Path and name of the ASCII containing the cross section data
        decay_filename : str
            Path and name of the ASCII containing the decay data
        xs_lib_ids : dict
            Dictionary where keys are "activation", "actinide", and "fp"
            and the values are the integral origen identifiers for the
            cross section data
        decay_lib_ids : dict
            Dictionary where keys are "activation", "actinide", and "fp"
            and the values are the integral origen identifiers for the
            decay data
        new_name : str, optional
            Descriptive name of the library. Defaults to using the title
            contained within the ASCII file; use this if you would like
            to rename the file.
        var_actinide_lib : int, optional
            If this is a "xs" library type, then this holds the
            variable actinide index that ORIGEN wants.

        Returns
        -------
        DepletionLibrary
            A DepletionLibrary object containing all the information
            present in an ORIGEN2.2 library.
        """

        # Check the file is a valid string and a valid file
        for file_str, filename in zip(["xs_filename", "decay_filename"],
                                      [xs_filename, decay_filename]):
            check_type(file_str, filename, str)
            if not os.path.isfile(filename):
                raise ValueError("{} does not exist".format(filename))

        # Check remaining types
        check_type("xs_lib_ids", xs_lib_ids, dict)
        check_length("xs_lib_ids", xs_lib_ids.keys(), 3)
        for k, v in xs_lib_ids.items():
            check_value("xs_lib_ids key", k, _ALLOWED_ISOTOPE_TYPES)
            check_type("xs_lib_ids value", v, int)
            check_greater_than("xs_lib_ids value", v, minimum=0)
        check_type("decay_lib_ids", decay_lib_ids, dict)
        check_length("decay_lib_ids", decay_lib_ids.keys(), 3)
        for k, v in decay_lib_ids.items():
            check_value("decay_lib_ids key", k, _ALLOWED_ISOTOPE_TYPES)
            check_type("decay_lib_ids value", v, int)
            check_greater_than("decay_lib_ids value", v, minimum=0)
        if new_name is not None:
            check_type("new_name", new_name, str)
        if var_actinide_lib is not None:
            check_type("var_actinide_lib", var_actinide_lib, int)
            check_greater_than("var_actinide_lib", var_actinide_lib, minimum=0,
                               equality=True)

        # Set some initialization parameters that we know in advance
        neutron_group_structure = ONE_GROUP_STRUCTURE
        xs_units = "b"

        # Open and operate on the file data
        # Do the cross section data first
        lib_name = None
        xs_data = OrderedDict()
        decay_data = OrderedDict()
        yield_data = OrderedDict()
        for library_type in ["xs", "decay"]:
            if library_type == "xs":
                filename = xs_filename
                lib_ids = xs_lib_ids
            else:
                filename = decay_filename
                lib_ids = decay_lib_ids

            for isotope_type in ["activation", "actinide", "fp"]:
                lib_id = lib_ids[isotope_type]
                library_cards = []
                with open(filename, 'r') as file:
                    # Find the card with lib_id as the leading number
                    for card in file:
                        NLB = card[:4]
                        if NLB.isspace():
                            continue
                        else:
                            NLB = int(NLB)
                        if NLB == lib_id:
                            # Then we have the start of our table!
                            # Get the TITLE if the user
                            # doesn't want to rename the library.
                            if lib_name is None:
                                # Only do this the 1st time through
                                if new_name is None:
                                    # We do this by "unsplitting" the
                                    # the rest of the card after the
                                    # library id.
                                    lib_name = card[8:].rstrip()
                                else:
                                    lib_name = new_name
                            # exit here so the next read starts here
                            break

                    for card in file:
                        # The library will be terminated by a negative
                        # number in the first four columns
                        first_entry = card[:CHARACTERS_FOR_LIB_ID]
                        if first_entry.find("-") > -1:
                            break
                        else:
                            # Sometimes after an E for an exponent,
                            # the ORIGEN libraries have " " instead of "+".
                            # Lets replace that space with a "+" so no special
                            # code is needed later
                            for i in range(len(card) - 2):
                                if card[i:i + 2] == "E " and card[i + 2].isnumeric():

                                    card = card[:i + 1] + "+" + card[i + 2:]
                            # Now replace comma-delimited with space-delim
                            card = card.replace(",", " ")
                            # And store
                            library_cards.append(card)

                # Now we can gather the data itself
                if library_type == "decay":
                    # This format is defined in Table 5.1 of the manual
                    # This data comes in sets of two lines
                    for i in range(0, len(library_cards) - 1, 2):
                        card1 = library_cards[i].split()
                        card2 = library_cards[i + 1].split()

                        # From table 5.1 of the ORIGEN2.2 manual, we expect
                        # the first card to be 9 entries long; lets check
                        if len(card1) != 9:
                            raise ValueError("Invalid library format")

                        # Ok lets get the data and put it into our data
                        nucid = card1[1]
                        iso_name = origen_nucid_to_GND(nucid)
                        if iso_name in decay_data:
                            # Then this existed for some previous isotope type
                            # so just append this isotope type and move on
                            decay_data[iso_name]["isotope_types"].append(
                                isotope_type)
                        else:
                            # This is new so get it all
                            decay_data[iso_name] = \
                                {"IU": int(card1[2]), "THALF": float(card1[3]),
                                 "FBX": float(card1[4]),
                                 "FPEC": float(card1[5]),
                                 "FPECX": float(card1[6]),
                                 "FA": float(card1[7]),
                                 "FIT": float(card1[8]),
                                 "FSF": float(card2[1]),
                                 "FN": float(card2[2]),
                                 "QREC": float(card2[3]),
                                 "ABUND": float(card2[4]),
                                 "ARCG": float(card2[5]),
                                 "WRCG": float(card2[6]),
                                 'isotope_types': [isotope_type]}

                elif library_type == "xs":
                    # This format is defined in Table 5.2 of the manual and can
                    # have one or two lines per dataset, depending on YYN
                    i = 0
                    while i < len(library_cards):
                        card1 = library_cards[i].split()

                        # From table 5.1 of the ORIGEN2.2 manual, we expect
                        # the first card to be 9 entries long; lets check
                        if len(card1) != 9:
                            raise ValueError("Invalid library format")
                        # Ok lets get the data and put it into our data
                        # from the first card
                        nucid = card1[1]
                        iso_name = origen_nucid_to_GND(nucid)

                        if iso_name in xs_data:
                            # Then this already existed for some other
                            # isotope type, add the type and
                            # type dependent data
                            xs_data[iso_name]['isotope_types'].append(
                                isotope_type)

                        else:
                            # This is new so get it all
                            # ORIGEN stores SN3N or SNA data in "SN3N_SNA"
                            # depending on if it is an actinide or not.
                            if isotope_type == "actinide":
                                sn3n = float(card1[4])
                                sna = 0.
                                snf = float(card1[5])
                                snp = 0.
                            else:
                                sn3n = 0.
                                sna = float(card1[4])
                                snf = 0.
                                snp = float(card1[5])

                            temp_data = \
                                {"SNG": float(card1[2]),
                                 "SN2N": float(card1[3]),
                                 "SN3N": sn3n, "SNA": sna,
                                 "SNF": snf, "SNP": snp,
                                 "SNGX": float(card1[6]),
                                 "SN2NX": float(card1[7])}

                            # If the cross sections are all zero, then
                            # we dont need to store it
                            if not all(v == 0.0 for v in temp_data.values()):
                                xs_data[iso_name] = temp_data
                                # Add on isotope_types (not included in
                                # temp_xs_data to make the all() block
                                # simple, we only have the keys/values
                                # we want to check for)
                                xs_data[iso_name]['isotope_types'] = \
                                    [isotope_type]

                        # We will get the yield data regardless of type;
                        # should only be there for FP but this shouldn't
                        # vary
                        yield_data[iso_name] = \
                            {"YYN": float(card1[8])}

                        # Do we need to get a second line?
                        if yield_data[iso_name]["YYN"] > 0:
                            # Yes we do, get the Y array
                            card2 = library_cards[i + 1].split()
                            yield_data[iso_name]["Y"] = \
                                np.array([float(entry) for entry in card2[1:]])
                            # And since these are in %, scale by 100.
                            yield_data[iso_name]["Y"] /= 100.
                            i += 2
                        else:
                            i += 1

        # Now we can create our class
        this = DepletionLibrary(lib_name, neutron_group_structure)

        # Convert the data for each isotope into our TransitionData format
        # Start with the decay data
        iso_decay = OrderedDict()
        iso_decay_types = {}
        has_fission = set()
        for iso_name in decay_data:
            # Get the library data
            data = decay_data[iso_name]
            data_units = ORIGEN_TIME_INTS[data["IU"]]
            decay_channels = {"beta-": None, "ec/beta+": None, "alpha": None,
                              "it": None, "sf": None, "beta-,n": None}
            if data_units == "stable":
                half_life = None
                data_units = "s"
            else:
                half_life = data["THALF"]
                decay_channels["beta-"] = \
                    1.0 - np.sum([data["FPEC"], data["FPECX"], data["FA"],
                                  data["FIT"], data["FSF"], data["FN"]])
                decay_channels["ec/beta+"] = data["FPEC"] + data["FPECX"]
                decay_channels["alpha"] = data["FA"]
                decay_channels["it"] = data["FIT"]
                decay_channels["sf"] = data["FSF"]
                decay_channels["beta-,n"] = data["FN"]

            if decay_channels["sf"] is not None and decay_channels["sf"] > 0.:
                has_fission.add(iso_name)

            # Create the object
            that = DecayData(half_life, data_units, data["QREC"])

            # Let's store how to find the child isotopes from these
            # ORIGEN channels
            for type_, br in decay_channels.items():
                if br is not None and br > 0.:
                    # Build the targets
                    z, a, m = adder.data.zam(iso_name)
                    targets = []
                    yields_ = []
                    if type_ == "beta-":
                        # May have beta-* which will require a yield
                        # first ground state
                        m1_yield = data["FBX"] / decay_channels["beta-"]
                        m0_yield = 1. - m1_yield
                        iso_0 = adder.data.gnd_name(z + 1, a, 0)
                        iso_1 = adder.data.gnd_name(z + 1, a, 1)
                    elif type_ == "ec/beta+":
                        # May have ec/beta+* which will require a yield
                        # first ground state
                        m1_yield = \
                            data["FPECX"] / decay_channels["ec/beta+"]
                        m0_yield = 1. - m1_yield
                        iso_0 = adder.data.gnd_name(z - 1, a, 0)
                        iso_1 = adder.data.gnd_name(z - 1, a, 1)
                    elif type_ == "alpha":
                        iso_0 = adder.data.gnd_name(z - 2, a - 4, 0)
                        iso_1 = None
                        m0_yield = 1.0
                        m1_yield = 0.
                    elif type_ == "it":
                        iso_0 = adder.data.gnd_name(z, a, 0)
                        iso_1 = None
                        m0_yield = 1.0
                        m1_yield = 0.
                    elif type_ == "sf":
                        iso_0 = "fission"
                        iso_1 = None
                        m0_yield = 1.0
                        m1_yield = 0.
                    elif type_ == "beta-,n":
                        iso_0 = adder.data.gnd_name(z + 1, a - 1, 0)
                        iso_1 = None
                        m0_yield = 1.0
                        m1_yield = 0.

                    targets.append(iso_0)
                    yields_.append(m0_yield)
                    if m1_yield > 0.:
                        targets.append(iso_1)
                        yields_.append(m1_yield)
                    that.add_type(type_, br, targets, yields_=yields_)

            iso_decay[iso_name] = that

            # Store the isotope types
            if iso_name in iso_decay_types:
                for iso_type in data['isotope_types']:
                    # Only store unique entries
                    if iso_type not in iso_decay_types[iso_name]:
                        iso_decay_types[iso_name].append(data['isotope_types'])
            else:
                iso_decay_types[iso_name] = data['isotope_types']

        # Repeat for xs
        iso_xs = OrderedDict()
        iso_xs_types = {}
        for iso_name in xs_data:
            # Create the object
            that = ReactionData(xs_units, num_groups=1)

            # Get the library data
            data = xs_data[iso_name]
            z, a, m = adder.data.zam(iso_name)

            # To add the reactions we need to get the target isotopes
            # and the yields
            types_to_origen = \
                {"(n,gamma)": [data["SNG"], data["SNGX"]],
                 "(n,2n)": [data["SN2N"], data["SN2NX"]],
                 "(n,3n)": [data["SN3N"]], "(n,a)": [data["SNA"]],
                 "fission": [data["SNF"]], "(n,p)": [data["SNP"]]}
            for type_, vals in types_to_origen.items():
                # The xs is an array of dimension 1 (for one-group)
                xs = np.sum([val for val in vals])
                # The yields and targets will have a length that matches
                # the number of targets produced
                if xs > 0.:
                    yields_ = [val / xs for val in vals]
                    xs = np.array([xs])
                else:
                    continue
                # Now find the targets based on the rxn type
                targets = []
                if type_ == "(n,gamma)":
                    targets.extend([adder.data.gnd_name(z, a + 1, 0),
                                    adder.data.gnd_name(z, a + 1, 1)])
                elif type_ == "(n,2n)":
                    targets.extend([adder.data.gnd_name(z, a - 1, 0),
                                    adder.data.gnd_name(z, a - 1, 1)])
                elif type_ == "(n,3n)":
                    targets.extend([adder.data.gnd_name(z, a - 2, 0)])
                elif type_ == "(n,a)":
                    targets.extend([adder.data.gnd_name(z - 2, a - 3, 0)])
                elif type_ == "fission":
                    targets.extend(["fission"])
                elif type_ == "(n,p)":
                    targets.extend([adder.data.gnd_name(z - 1, a, 0)])

                that.add_type(type_, xs_units, xs, targets, yields_)

            iso_xs[iso_name] = that

            if data["SNF"] > 0.:
                has_fission.add(iso_name)

            # Store the isotope types
            if iso_name in iso_xs_types:
                for iso_type in data['isotope_types']:
                    # Only store unique entries
                    if iso_type not in iso_xs_types[iso_name]:
                        iso_xs_types[iso_name].append(data['isotope_types'])
            else:
                iso_xs_types[iso_name] = data['isotope_types']

        # Repeat for yield
        # Origen has fixed fission yield isotopes so set those
        fiss_isos = ORIGEN_FISS_YIELD_ISOS
        iso_yield = {}
        for i, fiss_iso in enumerate(fiss_isos):
            # Create the object
            that = YieldData()

            # For this one we have to step through each of the isotopes
            # in the library to see if it has yield data; these will be
            # the keys for the data that goes in to add_isotope
            for iso_name in yield_data:
                data = yield_data[iso_name]
                if data["YYN"] > 0.:
                    that.add_isotope(iso_name, data["Y"][i])

            iso_yield[fiss_iso] = that

            has_fission.discard(fiss_iso)

        # Now set the isotope for each of ours that has fission but no
        # yield to the default of U235
        for fiss_iso in has_fission:
            iso_yield[fiss_iso] = iso_yield["U235"]

        # Now go through and assign a default fission yield (U-235)
        # to all isotopes which have a s.f. or fission x/s but are
        # not our ORIGEN fission isotopes.

        # Add each data set to the DepletionLibrary
        # First get the set of isotopes that are in all libraries
        # The ORIGEN manual says all isotopes should be in the decay
        # library, but we might as well protect against library format
        # errors at the cost of 2 more lines of code and a few millisec
        # of operations
        set_of_isos = set([k for k in iso_decay.keys()] +
                          [k for k in iso_xs.keys()] +
                          [k for k in iso_yield.keys()])

        # Go through each of these isotopes and figure out what we have
        for iso_name in set_of_isos:
            # Dont add this if it has too large a metastable state
            # (we save states >= MAX_METASTABLE_STATE for MSR removal
            # nuclides)
            z, a, m = adder.data.zam(iso_name)
            if m >= MAX_METASTABLE_STATE:
                msg = "Isotope {} has a metastable state larger than".format(m)
                msg += " allowed!"
                raise ValueError(msg)
            # Figure out what datasets this isotope has
            xs = None
            iso_xs_type = None
            nfy = None
            decay = None
            iso_decay_type = None
            if iso_name in iso_xs:
                xs = iso_xs[iso_name]
                iso_xs_type = iso_xs_types[iso_name]
            if iso_name in iso_yield:
                nfy = iso_yield[iso_name]
            if iso_name in iso_decay:
                decay = iso_decay[iso_name]
                iso_decay_type = iso_decay_types[iso_name]

            # Now that we know it all, store it
            # Notice we dont pass the decay and xs types, despite
            # having them. Leaving the code to track them here just in
            # case we decide to use info from the libs later
            this.add_isotope(iso_name, xs, nfy, decay)

        this.finalize_library()

        return this


def origen_nucid_to_GND(nucid):
    # Get the Z, A, M
    Z = int(nucid) // 10000
    AM = int(nucid) % 10000
    A = AM // 10
    M = AM % 10
    return adder.data.gnd_name(Z, A, M)


def GND_to_origen_nucid(iso_name):
    # Converts a GND-formatted name (e.g., "U235") to an origen nucid
    # format, returned as an integer
    Z, A, M = adder.data.zam(iso_name)
    if M < MAX_METASTABLE_STATE:
        origen_nucid = 10000 * Z + 10 * A + M
    else:
        # Shift these high to be above the real nuclides
        origen_nucid = 10000 * Z + 10 * A + M + 10000000
    return origen_nucid
