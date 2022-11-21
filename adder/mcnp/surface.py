import numpy as np

from adder.type_checker import check_type, check_value, check_greater_than, \
    check_less_than, check_iterable_type
from .coord_transform import CoordTransform
from .constants import SURF_MAX_ID, MAX_SURF_NUM_TRANSFORM
from .input_utils import num_format

_SURF_TYPES = ["p", "px", "py", "pz", "so", "s", "sx", "sy", "sz", "c/x",
               "c/y", "c/z", "cx", "cy", "cz", "k/x", "k/y", "k/z", "kx", "ky",
               "kz", "sq", "gq", "tx", "ty", "tz", "x", "y", "z", "p", "box",
               "rpp", "sph", "rcc", "rhp", "hex", "rec", "trc", "ell", "wed",
               "arb"]

_BOUNDARY_TYPES = ['transmission', 'vacuum', 'reflective', 'periodic', 'white']

class Surface(object):
    """A minimal class for an MCNP Surface.

    Parameters
    ----------
    surf_id : int
        The surface ID
    transform: CoordTransform or None
        The transformation associated with this surface
    surf_type : _SURF_TYPES
        The surface type (e.g., "cz")
    surf_params : str
        The remainder of the surface card.
    boundary_type : {'transmission', 'vacuum', 'reflective', 'periodic', 'white'}
        Boundary condition that defines the behavior for particles hitting the
        surface. Defaults to transmissive boundary condition where particles
        freely pass through the surface (i.e., no BC). Note that if a periodic
        BC is specified, then the periodic_id parameter must be provided.
        Further, if the BC is 'periodic', then the transform parameter must be
        None.
    periodic_id : None or int
        The identifier of the surface this one is periodic with, as applicable

    Attributes
    ----------
    id : int
        The surface ID
    coord_transform: CoordTransform or None
        The transformation associated with this surface
    type : _SURF_TYPES
        The surface type ("cz", e.g.)
    params : str
        The remainder of the surface card.
    boundary_type : {'transmission', 'vacuum', 'reflective', 'periodic', 'white'}
        Boundary condition that defines the behavior for particles hitting the
        surface. Defaults to transmissive boundary condition where particles
        freely pass through the surface (i.e., no BC). Note that if a periodic
        BC is specified, then the periodic_id parameter must be provided.
        Further, if the BC is 'periodic', then the transform parameter must be
        None.
    periodic_id : None or int
        The identifier of the surface this one is periodic with, as applicable
    """

    def __init__(self, surf_id, transform, surf_type, surf_params,
                 boundary_type, periodic_id):
        # Assign the parameters
        self.id = surf_id
        self.coord_transform = transform
        self.type = surf_type
        self.params = surf_params
        self.boundary_type = boundary_type
        self.periodic_id = periodic_id

    @classmethod
    def from_card(self, sfc_card, transforms):
        """Initializes the Surface object from the surface card and the list
        of defined coordinate transforms.

        Note the surface card should have line-breaks, comments, etc, removed
        already.

        Parameters
        ----------
        sfc_card : str
            The surface card from the input file
        transforms : dict
            The parsed transforms (CoordTransform objects) keyed by their id

        Returns
        -------
        obj : Surface
            The initialized Surface object

        """

        check_type("sfc_card", sfc_card, str)
        check_type("transforms", transforms, dict)
        check_iterable_type("transforms keys", list(transforms.keys()), int)
        check_iterable_type("transforms values", list(transforms.values()),
                            CoordTransform)

        # Get the data and split it up so we can parse each entry
        data = sfc_card.split()

        # The first entry includes the boundary condition and the surface id
        if data[0].isdigit():
            # Then there is no BC identifier, so it is a transmission bc
            # (may be periodic, we will find out in the next value
            surf_id = num_format(data[0], 'int')
            boundary_type = "transmission"
        elif data[0][0] == "+":
            # Then this is a white BC
            boundary_type = "white"
            surf_id = num_format(data[0][1:], 'int')
        elif data[0][0] == "*":
            # Then this is a reflective BC
            boundary_type = "reflective"
            surf_id = num_format(data[0][1:], 'int')
        else:
            msg = "Invalid first entry of surface: {}".format(data[0])
            raise ValueError(msg)

        # Now check to see if the transform id/periodic surface id is
        # provided. We can tell this because the next card will be a number
        # not a string
        try:
            # If we can convert it, it is a number
            transform_or_periodic_num = num_format(data[1], 'int')
        except ValueError:
            # Then it was a string, like 'cx'.
            transform_or_periodic_num = None
        # now deal with the ramifications of the above; note this could be
        # cleaned up if we put these if branches in the try/except block, but
        # generally you minimize the code within a try/except so you are sure
        # what operation raised the exception
        if transform_or_periodic_num is not None:
            # Then we figure out if this is a transform of a periodic sfc id
            if transform_or_periodic_num > 0:
                # Then we have a transform #!
                transform_id = transform_or_periodic_num
                periodic_num = None
            else:
                # Look at this periodic sfc # over here
                transform_id = None
                periodic_num = -1 * transform_or_periodic_num
                boundary_type = "periodic"
        else:
            # We have none of the above
            transform_id = None
            periodic_num = None
            # Add in a blank entry to data at index 1 so that we dont need to
            # worry about what index the later entries are at
            data.insert(1, '')

        # Ok, next up we can get the sfc type and the remaining parameters
        surf_type = data[2]
        surf_params = " ".join(data[3:])

        # Before we can init, we need to find the right transform
        if transform_id is not None:
            if transform_id in transforms:
                transform = transforms[transform_id]
            else:
                msg = "Invalid Transform Number ({}) on Surface {}!"
                raise ValueError(msg.format(transform_id, surf_id))
        else:
            transform = None

        # Now we can initialize the object
        obj = Surface(surf_id, transform, surf_type, surf_params,
                 boundary_type, periodic_num)

        return obj


    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, surf_id):
        check_type("surf_id", surf_id, int)
        check_greater_than("surf_id", surf_id, 0)
        check_less_than("surf_id", surf_id, SURF_MAX_ID, equality=True)
        self._id = surf_id

    @property
    def coord_transform(self):
        return self._coord_transform

    @coord_transform.setter
    def coord_transform(self, transform):
        if transform is not None:
            check_type("coord_transform", transform, CoordTransform)
            check_less_than("coord_transform id", transform.id,
                            MAX_SURF_NUM_TRANSFORM, equality=True)
        self._coord_transform = transform

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, surf_type):
        check_type("surf_type", surf_type, str)
        st_lo = surf_type.lower()
        check_value("surf_type", st_lo, _SURF_TYPES)
        self._type = st_lo

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, surf_params):
        check_type("surf_params", surf_params, str)
        self._params = surf_params

    @property
    def boundary_type(self):
        return self._boundary_type

    @boundary_type.setter
    def boundary_type(self, boundary_type):
        check_type("boundary_type", boundary_type, str)
        btype_lo = boundary_type.lower()
        check_value("boundary_type", btype_lo, _BOUNDARY_TYPES)
        if btype_lo == "periodic" and self.coord_transform is not None:
            msg = "Cannot set a periodic boundary on a surface with a " + \
                "Coordinate Transformation!"
            raise ValueError(msg)
        self._boundary_type = btype_lo
        # This should be handled by init, but just in case a user assigns this
        # directly, lets also update periodic_id
        if btype_lo != "periodic":
            self.periodic_id = None

    @property
    def periodic_id(self):
        return self._periodic_id

    @periodic_id.setter
    def periodic_id(self, periodic_id):
        if periodic_id is None:
            if self.boundary_type == "periodic":
                msg = "A periodic boundary requires a numeric periodic id!"
                raise ValueError(msg)
            self._periodic_id = periodic_id
        elif self.boundary_type == "periodic":
            check_type("periodic_id", periodic_id, int)
            check_greater_than("periodic_id", periodic_id, 0)
            check_less_than("periodic_id", periodic_id, SURF_MAX_ID,
                            equality=True)
            self._periodic_id = periodic_id
        else:
            msg = "A periodic boundary requires a numeric periodic id!"
            raise ValueError(msg)

    def __str__(self):
        return self.to_str()

    def to_str(self):
        """Returns this surface as a surface card string"""

        # Get the j, n, A, params
        # j is the surface id
        j = self.id

        # Now get n, the coord transform ID, as needed
        if self.coord_transform is not None and self.coord_transform.id != 0:
            n = self.coord_transform.id
        else:
            n = ""
        # Next, get the boundary flag
        if self.boundary_type in ["transmission", "vacuum"]:
            bflag = ""
        elif self.boundary_type == "reflective":
            bflag = "*"
        elif self.boundary_type == "periodic":
            bflag = ""
            # We also have to adjust n
            n = str(-self.periodic_id)
        elif self.boundary_type == "white":
            bflag = "+"

        # Update j with the boundary flag
        j = bflag + str(j)

        # Now get A and params
        A = self.type
        params = self.params

        if n == "":
            card = "{} {} {}".format(j, A, params)
        else:
            card = "{} {} {} {}".format(j, n, A, params)

        return card
