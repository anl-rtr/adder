from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from adder.type_checker import *
from adder.utils import get_id
from .input_utils import num_format
from .constants import TRANSFORM_MAX_ID, DEFAULT_DISPLACEMENT, DEFAULT_ROT_MAT


class CoordTransform(object):
    """A Coordinate transform, including a translation vector and
    rotation matrix. This corresponds to the TR card in MCNP.

    Parameters
    ----------
    id_ : int, optional
        The transform's ID. If none provided, a new integer is generated
        based on which transform IDs are currently in use
    displacement : Iterable of float, optional
        The displacement vector. Defaults to array([0., 0., 0.])
    rotation_angles : Iterable of float, optional
        Angles of rotation with respect to z, y, and x axes
        (i.e., yaw, pitch, and roll)
    rotation_matrix : 3x3 Iterable of float
        The rotation matrix
    m_flag : bool, optional
        If True, then the displacement vector is the location of the
        origin of the auxiliary coordinate system, defined in the main
        system. If False, then the displacement vector is the location
        of the origin of the main coordinate system, defined in the
        auxiliary system. True corresponds to m=1, False to m=-1 in
        MCNP. Defaults to True
    in_degrees : bool, optional
        Whether the rotation angles or matrix are provided in
        degrees (True)


    Attributes
    ----------
    id : int
        The transform's ID
    displacement : Iterable of float, optional
        The displacement vector. Defaults to array([0., 0., 0.])
    rotation_matrix : 3x3 Iterable of float
        The rotation matrix
    m_flag : bool
        If True, then the displacement vector is the location of the
        origin of the auxiliary coordinate system, defined in the main
        system. If False, then the displacement vector is the location
        of the origin of the main coordinate system, defined in the
        auxiliary system. True corresponds to m=1, False to m=-1 in MCNP

    """

    _USED_IDS = set([])

    def __init__(self, id_=None, displacement=None, rotation_angles=None,
                 rotation_matrix=None, m_flag=True, in_degrees=False):
        self.id = id_
        self.m_flag = m_flag
        self.displacement = displacement

        if rotation_angles:
            if rotation_matrix:
                msg = "Cannot use rotation_angles and rotation_matrix " + \
                    "for CoordTransform initialization; use one or the other!"
                raise ValueError(msg)

            # Compute and set the corresponding rotation matrix
            if in_degrees:
                const = (np.pi / 180.)  # Convert from deg to rad
            else:
                const = 1.  # No conversion

            # Now build the rotation matrices
            alpha, beta, gamma = np.asarray(rotation_angles) * const
            ca, sa = np.cos(alpha), np.sin(alpha)
            cb, sb = np.cos(beta), np.sin(beta)
            cg, sg = np.cos(gamma), np.sin(gamma)

            R = np.array([
                [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [-sb, cb * sg, cb * cg]])
            # It is very likely for there to be values very nearly 0 (6e-17)
            # after the sin/cos. Just set them to zero
            R[np.abs(R) < 5. * np.finfo(np.float).eps] = 0.
            self._rotation_matrix = R
        else:
            self.set_rotation_matrix(rotation_matrix, in_degrees)

    def __del__(self):
        # When the transform instance goes out of scope or is deleted with the
        # del keyword we shall de-register the ID
        try:
            CoordTransform._USED_IDS.remove(self._id)
        except KeyError:
            pass

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if id_ is None:
            # Then we have to auto-generate an ID, use our generalized function
            self._id = get_id(CoordTransform._USED_IDS, TRANSFORM_MAX_ID)
        else:
            check_type("id", id_, int)
            check_greater_than("id", id_, 0, equality=True)
            check_less_than("id", id_, TRANSFORM_MAX_ID, equality=True)
            self._id = id_
            CoordTransform._USED_IDS.add(self._id)

    @property
    def m_flag(self):
        return self._m_flag

    @m_flag.setter
    def m_flag(self, m):
        check_type("m_flag", m, bool)
        self._m_flag = m

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, d):
        if d is not None:
            check_type("displacement", d, Iterable)
            check_length("displacement", d, 3)
            for i in range(len(d)):
                if d[i] == "j":
                    # replace with default
                    d[i] = DEFAULT_DISPLACEMENT[i]
            # Now we can assign it
            self._displacement = np.array(d)
        else:
            # Then set to the default
            self._displacement = DEFAULT_DISPLACEMENT

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def is_null(self):
        is_null = False
        if np.allclose(self._displacement, np.zeros(3), atol=1.E-13):
            if np.allclose(self._rotation_matrix, np.eye(3), atol=1.E-13):
                is_null = True
        return is_null

    def set_rotation_matrix(self, matrix, in_degrees):
        # We go through this as the rotation matrix can be a mix of
        # floats and strings (for jumps) and thus it can also be a
        # list of lists
        if matrix is not None:
            check_type("rotation_matrix", matrix, np.ndarray)
            check_value("rotation_matrix shape", matrix.shape, ((3, 3),))

            # Convert from degrees, if needed
            if in_degrees:
                matrix = np.cos(matrix * np.pi / 180.)
            matrix[np.abs(matrix) < 5. * np.finfo(np.float).eps] = 0.
            # Now we can make an ndarray out of it and store, ensuring
            # it is a float64
            self._rotation_matrix = np.array(matrix, dtype=np.float64)

        else:
            # Then set to the default
            self._rotation_matrix = DEFAULT_ROT_MAT

    def short_str(self):
        """Reproduces the transform as a string, but without the tr# component.
        This is used specifically for comparing TR cards"""

        tr_str = ""
        # Append displacement
        for i in range(len(self.displacement)):
            tr_str += " {}".format(self.displacement[i])

        # Append rotation
        for j in range(self.rotation_matrix.shape[0]):
            for i in range(self.rotation_matrix.shape[1]):
                tr_str += " {}".format(self.rotation_matrix[j][i])

        # And finally the m flag
        if self.m_flag:
            tr_str += " 1"
        else:
            tr_str += " -1"

        return tr_str


    def __str__(self):
        """Reproduces the transform as a string to be used in the MCNP
        input.

        Returns
        -------
        tr_str : str
            The transform input file string
        """

        tr_str = "tr{}".format(self.id) + self.short_str()

        return tr_str

    def clone(self, new_id=None, memo=None):
        """Create a copy of this Transform with a new unique ID

        Returns
        -------
        clone : CoordTransform
            The clone of this transform

        """

        if memo is None:
            memo = {}

        # If no nemoize'd clone exists, instantiate one
        if self not in memo:
            clone = deepcopy(self)

            # Set the new id (The setter will handle the rest, including
            # handling if new_id is None, and updating the counter)
            clone.id = new_id

            # Memoize the clone
            memo[self] = clone

        return memo[self]

    def combine(self, that, in_place=True, new_id=None):
        """Adds the displacement vectors and multiplies the rotation
        matrices. This is done as self * that (i.e., that rotates self)

        Parameters
        ----------
        that : CoordTransform
            The CoordTransform to combine with self
        in_place : bool, optional
            Whether or not to perform the combination in-place
        new_id : None or int, optional
            If in_place is False, and the user wishes to set the id,
            this parameter sets it.

        Returns
        -------
        result : CoordTransform
            If in_place is True, this will be a new CoordTransform; if
            in_place is False, then self will be modified and returned
        """

        # Do checking
        check_type("that", that, CoordTransform)
        check_type("in_place", in_place, bool)
        if not in_place:
            if new_id is not None:
                check_type("new_id", new_id, int)

        # Ensure both have the same m value, since its unclear how
        if self.m_flag != that.m_flag:
            msg = "Cannot combine two CoordTransform objects with " + \
                "differing values of m!"
            raise ValueError(msg)

        if in_place:
            mod = self
        else:
            mod = self.clone(new_id)

        # Update displacement vector
        mod.displacement += that.displacement

        # Update rotation matrix
        mod.set_rotation_matrix(np.matmul(that.rotation_matrix,
                                          mod.rotation_matrix), False)

        return mod

    @classmethod
    def from_card(cls, keyword, card_data):
        # Instantiates and returns a CoordTransform class given the MCNP
        # card and the keyword associated with that card.
        # The card data is after *TR#, TR#, TRCL=, or *TRCL=.
        # The keyword is either *TR#, TR#, TRCL=, or *TRCL=;
        # these are (case-insensitive)

        # First we can learn what we can from the keyword
        # This will be either if it is in degrees, and possibly what the
        # transformation id is
        low_key = keyword.lower().strip()
        # Start with in-degrees or not, and if we have the *, take out
        # the rest
        if low_key[0] == "*":
            in_degrees = True
            low_key = low_key[1:]
        else:
            in_degrees = False

        # Now see if it starts with tr or trcl
        if low_key.startswith("trcl"):
            card_type = "trcl"
        elif low_key.startswith("tr"):
            card_type = "tr"
        else:
            msg = "Invalid Keyword for CoordTransform.from_card!"
            raise ValueError(msg)

        # Now see if there is an id
        if len(low_key) > len(card_type):
            # Make sure we have digits in the remainder
            if low_key[len(card_type):].isdigit():
                # Ok, we have our ID
                id_ = num_format(low_key[len(card_type):], 'int')
            else:
                msg = "Keyword {} should end in an integer id!".format(keyword)
                raise ValueError(msg)
        else:
            # No id
            id_ = None

        # Now we operate on the card data
        # First strip the parentheses (present for TRCL)
        card = card_data.strip("()")

        # Now split the cards in to the values for each
        str_vals = card.lower().split()

        # First get the value of the m_flag. This is only present if the
        # length of values is 14.
        m_flag = True
        if len(str_vals) == 13:
            str_val = str_vals.pop(-1)
            if str_val != "j":
                if num_format(str_val, 'float') < 0.:
                    # The MCNP source (trfmat.F90) sets a 0 value of
                    # this flag to 1, we handle with the < here.
                    m_flag = False

        # Get the displacement and rotation matrix, including handling
        # jump values
        displacement = []
        flat_matrix = []
        defaults = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]
        i = 0
        for str_val in str_vals:
            if i < 3:
                if str_val == "j":
                    displacement.append(defaults[i])
                else:
                    val = num_format(str_val, 'float')
                    displacement.append(val)
            else:
                if str_val == "j":
                    flat_matrix.append(defaults[i])
                else:
                    val = num_format(str_val, 'float')
                    if in_degrees:
                        val = np.cos(val * np.pi / 180.)
                    flat_matrix.append(val)
            i += 1

        # Since we converted from degrees, set in_degrees accordingly
        in_degrees = False

        # The following is adapted from the MCNP6.2 src (trfmat.F90) to
        # convert the translation matrix for each of the patterns
        num_vals = len(flat_matrix)
        # We will first build a tr array as used in MCNP, including
        # enabling 1-based indexing. That is, we want our rotation
        # matrix to start at index 5. Further, the data after the
        # rotation matrix should be filled with nan, similar to
        # MCNP's huge_float
        tr = [None] * 5 + flat_matrix[:] + [None] * (14 - num_vals - 5)

        # From here, the code is effectively trfmat.F90 though
        # translated from F90 to python and using numpy when possible
        ki = 0
        exit_flag = True
        if num_vals == 0:
            # Then its just the default
            tr[5:] = defaults[3:]
        elif num_vals == 3:
            # One vector is provided, pattern 4
            for k in [1, 2]:
                for j in [1, 2, 3]:
                    if np.all(tr[3 * j + 2: 3 * j + 5] is not None):
                        r = np.sqrt(tr[3 * j + 2]**2 + tr[3 * j + 3]**2 +
                                    tr[3 * j + 4]**2)
                        # Normalize the data we have
                        if r != 0.:
                            for i in [2, 3, 4]:
                                tr[3 * j + i] /= r
                        else:
                            error()

                        tr = arbitrary_orth(tr, j)

                        if ki != 0:
                            swap_pos(tr)
                            exit_flag = True
                if exit_flag:
                    break
                ki = 1
                swap_pos(tr)

        elif num_vals == 6:
            # Two vectors in the same system is provided, this is
            # pattern 2
            for k in [1, 2]:
                for j in [1, 2, 3]:
                    if all(x is None for x in tr[3 * j + 2: 3 * j + 5]):
                        for ell in [1, 2]:
                            m = ((j + ell - 1) % 3) + 1
                            r = np.sqrt(tr[3 * m + 2]**2 + tr[3 * m + 3]**2 +
                                        tr[3 * m + 4]**2)
                            # Normalize the data we have
                            if r != 0.:
                                for i in [2, 3, 4]:
                                    tr[3 * m + i] /= r
                            else:
                                error()

                        # Generate with the cross product
                        tr = cross_tr(tr, j)

                        if ki != 0:
                            swap_pos(tr)
                            exit_flag = True
                if exit_flag:
                    break
                ki = 1
                swap_pos(tr)

        elif num_vals == 5:
            # One vector in each system, pattern 3
            for i1 in [1, 2, 3]:
                if np.all(tr[3 * i1 + 2: 3 * i1 + 5] is not None):
                    for j1 in [1, 2, 3]:
                        if np.all([tr[j1 + 4], tr[j1 + 7],
                                   tr[j1 + 10]] is not None):
                            r = tr[3 * i1 + j1 + 1]**2
                            for i in range(5, 14):
                                if tr[i] is not None:
                                    r += tr[i]**2

                            r = np.sqrt(0.5 * r)
                            if r == 0.:
                                error()

                            for i in range(5, 14):
                                if tr[i] is not None:
                                    tr[i] /= r

                            if tr[3 * i1 + j1 + 1]**2 == 1.:
                                error()

                            i2 = (i1 % 3) + 1
                            i3 = (i2 % 3) + 1
                            j2 = (j1 % 3) + 1
                            j3 = (j2 % 3) + 1
                            r = 1. / (1. - tr[3 * i1 + j1 + 1]**2)
                            a = tr[3 * i1 + j3 + 1] * tr[3 * i3 + j1 + 1] * r
                            b = tr[3 * i1 + j2 + 1] * tr[3 * i2 + j1 + 1] * r
                            c = tr[3 * i2 + j1 + 1] * tr[3 * i1 + j3 + 1] * r
                            d = tr[3 * i3 + j1 + 1] * tr[3 * i1 + j2 + 1] * r
                            tr[3 * i2 + j2 + 1] = -a - b * tr[3 * i1 + j1 + 1]
                            tr[3 * i2 + j3 + 1] = d - c * tr[3 * i1 + j1 + 1]
                            tr[3 * i3 + j2 + 1] = c - d * tr[3 * i1 + j1 + 1]
                            tr[3 * i3 + j3 + 1] = -b - a * tr[3 * i1 + j1 + 1]

                            exit_flag = True

                    if exit_flag:
                        break

        elif num_vals == 9:
            # The entire matrix is provided
            # We have nothing to do here
            pass

        # Convert the end result to a 3x3 array
        rot_mat = np.array(tr[5:])
        rot_mat.shape = (3, 3)

        rot_ang = None
        coord_tr = cls(id_, displacement, rot_ang, rot_mat, m_flag, in_degrees)

        return coord_tr

    @staticmethod
    def merge_and_clean(coord_transforms, cells, surfaces):
        """Combines the coordinate transforms so that the transforms with the
        same tr card are kept and the rest are discarded. The parameters are
        modified in place.

        This is done in two parts, first we compare, and then we replace the
        cell and surface references to these transforms.

        Parameters
        ----------
        coord_transforms : dict
            The model's coordinate transforms (CoordTransform objects) keyed
            by the transform's id
        cells : dict
            The parsed cells (Cell objects) keyed by the cell id
        surfaces : dict
            The parsed surfaces (Surface objects) keyed by the surface id
        """

        # This must be done in place so that coord_transform from upstream is
        # the same as the one we have modified here

        # First just create the items we will compare
        short_strs = {}
        for id_, tr in coord_transforms.items():
            short_strs[id_] = tr.short_str()

        # Now find the unique members with a flipped dictionary
        flipped = {}
        for k, v in short_strs.items():
            if v not in flipped:
                flipped[v] = [k]
            else:
                flipped[v].append(k)

        # Now convert to a set of all the transform IDs and the transforms to
        # merge with
        # A transform id is only present if there is merging to be done
        to_merge = {}
        for ids in flipped.values():
            if len(ids) > 1:
                for id_ in ids[1:]:
                    to_merge[id_] = ids[0]

        # Never allow id 0 to be merged (shouldnt get here, but for robustness)
        if 0 in to_merge:
            del to_merge[0]

        # Do not continue if there is nothing to merge
        if len(to_merge) == 0:
            return

        # Parse the cells and surfaces and perform the combining. We only
        # want to parse cells/surfs once each since there will be many and this
        # could get expensive
        # We also will be keeping track of unused transforms so we can free
        # those resources as well
        transform_used = {id_: False for id_ in coord_transforms}
        transform_used[0] = True
        # First do cells
        for cell in cells.values():
            cell_transforms = cell.get_transforms()
            for cell_transform in cell_transforms:
                # See if this is one to be merged
                if cell_transform.id in to_merge:
                    # Then get the references to the transforms and use the
                    # one we will keep
                    old_tr = cell_transform
                    new_id = to_merge[old_tr.id]
                    if new_id == 0:
                        new_tr = None
                    else:
                        new_tr = coord_transforms[new_id]
                        transform_used[new_id] = True
                    cell.update_transforms(old_tr, new_tr)
                else:
                    # If not, then mark that it is used (we dont need to mark
                    # the used status of ones we are removing by merging since
                    # we will remove the to_merge keys anyways; this saves us
                    # from calling get_transforms again and paying that cost)
                    transform_used[cell_transform.id] = True

        # Now we can do surfaces
        for surface in surfaces.values():
            if surface.coord_transform is not None:
                old_tr_id = surface.coord_transform.id
                if old_tr_id in to_merge:
                    # Then use the transform we will keep
                    new_id = to_merge[old_tr_id]
                    if new_id == 0:
                        surface.coord_transform = None
                    else:
                        surface.coord_transform = coord_transforms[new_id]
                        transform_used[new_id] = True
                else:
                    transform_used[old_tr_id] = True

        # Remove the coord transforms that we merged from
        # the registry and those we no longer use
        for tr_id in to_merge.keys():
            del coord_transforms[tr_id]
        for tr_id, tr_status in transform_used.items():
            if not tr_status and tr_id in coord_transforms:
                # Then it isnt used. Yerrrrr outta here!
                del coord_transforms[tr_id]


def error():
    msg = "Coordinate Transform is incorrectly defined"
    raise ValueError(msg)


def arbitrary_orth(tr, j):
    alo = 3 * j + 2
    blo = 3 * (j % 3 + 1) + 2
    clo = 3 * ((j + 1) % 3 + 1) + 2

    # a through b will be 0-indexed unlike tr
    a = np.array(tr[alo: alo + 3])
    b = np.array(tr[blo: blo + 3])
    c = np.array(tr[clo: clo + 3])

    j = 1
    if np.abs(a[1]) > np.abs(a[0]):
        j = 2
    if np.abs(a[2]) > np.abs(a[j]):
        j = 3
    k = j % 3 + 1
    ell = 6 - j - k
    # Adjust for 0 indexing
    j -= 1
    k -= 1
    ell -= 1
    t = a[j]**2 + a[k]**2
    if t == 0.:
        error()
    b[ell] = 0.0637922
    b[k] = (a[j] * np.sqrt(t - (t + a[ell]**2) * b[ell]**2) -
            a[k] * a[ell] * b[ell]) / t
    b[j] = -(b[k] * a[k] + b[ell] * a[ell]) / a[j]
    c = np.cross(a, b)

    # Now put the values back in place
    tr[alo: alo + 3] = a[:]
    tr[blo: blo + 3] = b[:]
    tr[clo: clo + 3] = c[:]
    return tr


def swap_pos(tr):
    iy = np.reshape([6, 8, 7, 11, 10, 12], (2, 3))
    for i in range(3):
        t = tr[iy[0, i]]
        tr[iy[0, i]] = tr[iy[1, i]]
        tr[iy[1, i]] = t


def cross_tr(tr, j):
    alo = 3 * (j % 3 + 1) + 2
    blo = 3 * ((j + 1) % 3 + 1) + 2
    clo = 3 * j + 2
    a = np.array(tr[alo: alo + 3])
    b = np.array(tr[blo: blo + 3])
    c = np.cross(a, b)

    tr[alo: alo + 3] = a[:]
    tr[blo: blo + 3] = b[:]
    tr[clo: clo + 3] = c[:]
    return tr
