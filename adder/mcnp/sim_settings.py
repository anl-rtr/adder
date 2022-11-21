import numpy as np

from adder.type_checker import check_type, check_value, check_greater_than, \
    check_iterable_type
from .input_utils import num_format


# TODO: This right now only applies to KCODE; in the future it would make
# sense to have this also handle the NPS card as is used (at least) for
# stochastic volume calculations. In the grand scheme of things this should
# be very similar to OpenMC's Settings class, but sadly we have to make those
# moves as needed not all at once.

class SimSettings(object):
    """An object containing simulation settings for the MCNP calculation.

    The default values are those specified in the MCNP 5 1.60 and MCNP6.2
    Users Manual; both are consistent with each other.

    At present this is limited to the KCODE card, however, it can eventually
    be extended to encapsulate any simulation settings that ADDER needs to be
    aware of and to modify.

    Parameters
    ----------
    particles : int
        The number of source histories per cycle (NSRCK); defaults to 1000
    keff_guess : float
        The initial guess for keff (RKK); defaults to 1.0
    inactive : int
        The number of inactive batches (IKZ); defaults to 30
    batches : int
        The number of batches to simulate (KCT); defaults to 100 more than
        inactive.
    src_storage : int
        Number of source points for which to allocate storage (MSRK); defaults
        to the largest of 4500 or 2 * particles
    normalize_by_weight : bool
        If true, normalize tallies by weight, otherwise, normalize by
        the number of histories (KNRM); defaults to True
    max_output_batches : int
        The maximum number of batch values to report on the MCTAL
        and/or RUNTPE file (MRKP); defaults to 6500
    max_avg_batches : bool
        Whether to average summary and tally info over active batches (True)
        or all batches (False) (KC8); defaults to True
    additional_cards : List of str
        Additional cards stored on this SimSettings object that are not yet
        intelligently parsed

    Attributes
    ----------
    particles : int
        The number of source histories per cycle (NSRCK)
    keff_guess : float
        The initial guess for keff (RKK)
    inactive : int
        The number of inactive batches (IKZ)
    batches : int
        The number of batches to simulate (KCT)
        inactive.
    src_storage : int
        Number of source points for which to allocate storage (MSRK)
    normalize_by_weight : bool
        If true, normalize tallies by weight, otherwise, normalize by
        the number of histories (KNRM)
    max_output_batches : int
        The maximum number of batch values to report on the MCTAL
        and/or RUNTPE file (MRKP)
    max_avg_batches : bool
        Whether to average summary and tally info over active batches (True)
        or all batches (False) (KC8)
    additional_cards : List of str
        Additional cards stored on this SimSettings object that are not yet
        intelligently parsed
    """

    def __init__(self, particles=1000, keff_guess=1., inactive=30,
                 batches=None, src_storage=None, normalize_by_weight=True,
                 max_output_batches=6500, max_avg_batches=True,
                 additional_cards=None):
        self.particles = particles
        self.keff_guess = keff_guess
        self.inactive = inactive
        self.batches = batches
        self.src_storage = src_storage
        self.normalize_by_weight = normalize_by_weight
        self.max_output_batches = max_output_batches
        self.max_avg_batches = max_avg_batches
        self.additional_cards = additional_cards

    @classmethod
    def from_cards(self, cards):
        """Initialize the SimSettings object from the relevant cards.

        Note these cards should have line-breaks, etc, removed already

        Parameters
        ----------
        cards : Iterable of str
            The kcode line from the input file and others handled by this class

        Returns
        -------
        obj : SimSettings
            The initialized SimSettings object

        """

        # Parse the line and first make sure we have a kcode card
        check_type("cards", cards, (list, tuple))
        check_iterable_type("cards", cards, str)

        # Now find the kcode and additional cards
        additional_cards = []
        kcode_line = ""
        for card in cards:
            if card.lower().startswith("kcode"):
                kcode_line = card
            else:
                additional_cards.append(card)
        data = kcode_line.split()
        check_greater_than("kcode card", len(data), 1, equality=True)
        # Set additional cards to None if empty
        if len(additional_cards) == 0:
            additional_cards = None

        # Now that we know this is acceptable, initialize to the defaults
        obj = SimSettings(additional_cards=additional_cards)

        # Now we can get the data we need. We use a zip as it stops when at the
        # end of the shortest list; and so parameters not checked are going to
        # still be at their default value from above
        params = ["particles", "keff_guess", "inactive", "batches",
                  "src_storage", "normalize_by_weight", "max_output_batches",
                  "max_avg_batches"]
        for param, val in zip(params, data[1:]):
            if val.lower() != "j":
                # Then this is not a jump and we should process it
                if param == "keff_guess":
                    val = num_format(val, 'float')
                elif param == "normalize_by_weight":
                    val = num_format(val, 'int')
                    if val == 0:
                        val = True
                    else:
                        val = False
                elif param == "max_avg_batches":
                    val = num_format(val, 'int')
                    if val == 0:
                        val = False
                    else:
                        val = True
                else:
                    val = num_format(val, 'int')
                setattr(obj, param, val)
            # otherwise it will be at the default value from above

        return obj

    @property
    def particles(self):
        return self._particles

    @particles.setter
    def particles(self, particles):
        check_type("particles", particles, int)
        check_greater_than("particles", particles, 0)
        self._particles = particles

    @property
    def keff_guess(self):
        return self._keff_guess

    @keff_guess.setter
    def keff_guess(self, keff_guess):
        check_type("keff_guess", keff_guess, (float, int))
        check_greater_than("keff_guess", keff_guess, 0.)
        self._keff_guess = num_format(keff_guess, 'float')

    @property
    def inactive(self):
        return self._inactive

    @inactive.setter
    def inactive(self, inactive):
        check_type("inactive", inactive, int)
        check_greater_than("inactive", inactive, 0)
        self._inactive = inactive

    @property
    def batches(self):
        if self._batches is None:
            # The default behavior is 100 + inactive
            return self._inactive + 100
        else:
            return self._batches

    @batches.setter
    def batches(self, batches):
        if batches is None:
            self._batches = batches
        else:
            check_type("batches", batches, int)
            check_greater_than("batches", batches, 0)
            self._batches = batches

    @property
    def src_storage(self):
        if self._src_storage is None:
            return max(4500, 2 * self.particles)
        else:
            return self._src_storage

    @src_storage.setter
    def src_storage(self, src_storage):
        if src_storage is None:
            self._src_storage = src_storage
        else:
            check_type("src_storage", src_storage, int)
            check_greater_than("src_storage", src_storage, 0)
            self._src_storage = src_storage

    @property
    def normalize_by_weight(self):
        return self._normalize_by_weight

    @normalize_by_weight.setter
    def normalize_by_weight(self, normalize_by_weight):
        check_type("normalize_by_weight", normalize_by_weight, bool)
        self._normalize_by_weight = normalize_by_weight

    @property
    def max_output_batches(self):
        return self._max_output_batches

    @max_output_batches.setter
    def max_output_batches(self, max_output_batches):
        check_type("max_output_batches", max_output_batches, int)
        check_greater_than("max_output_batches", max_output_batches, 0)
        self._max_output_batches = max_output_batches

    @property
    def max_avg_batches(self):
        return self._max_avg_batches

    @max_avg_batches.setter
    def max_avg_batches(self, max_avg_batches):
        check_type("max_avg_batches", max_avg_batches, bool)
        self._max_avg_batches = max_avg_batches

    @property
    def additional_cards(self):
        return self._additional_cards

    @additional_cards.setter
    def additional_cards(self, additional_cards):
        if additional_cards is not None:
            check_type("additional_cards", additional_cards, (list, tuple))
            check_iterable_type("additional_cards", additional_cards, str)
        self._additional_cards = additional_cards

    @property
    def kcode_str(self):
        """Returns the kcode card as a string"""

        template = "kcode {} {:1.5f} {} {} {} {} {} {}"

        # Convert booleans to expected values
        if self.normalize_by_weight:
            norm_by_wgt_int = 0
        else:
            norm_by_wgt_int = 1

        if self.max_avg_batches:
            max_avg_batch_int = 1
        else:
            max_avg_batch_int = 0

        card = template.format(self.particles, self.keff_guess, self.inactive,
                               self.batches, self.src_storage, norm_by_wgt_int,
                               self.max_output_batches, max_avg_batch_int)

        return card

    @property
    def all_cards(self):
        if self.additional_cards:
            return [self.kcode_str] + self.additional_cards
        else:
            return [self.kcode_str]
