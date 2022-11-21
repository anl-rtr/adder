from collections import OrderedDict
from copy import deepcopy
from warnings import warn

import numpy as np

from adder.type_checker import *
from adder.material import Material
from adder.utils import get_id

from .input_utils import split_data_and_keywords, num_format
from .constants import CELL_KW, MATL_MAX_ID, CELL_MAX_ID, ROOT_UNIV, \
    VALID_FILL_TYPES, IRREG_LAT_ARRAY, USE_LAT_IRREG, USE_LAT_MAT, LIKE_CELL_KW, \
    CELL_DENSITY_FMT
from .coord_transform import CoordTransform


class Cell(object):
    """An MCNP Cell object.

    Parameters
    ----------
    cell_id : int
        The cell's ID
    material_id : int
        The id of the material in this cell
    density : float
        Density of the material in this cell in units of a/b-cm
    surfaces : str
        Surfaces defining the region of this cell
    volume : float or None, optional
        Volume of the region; defaults to None
    coord_transform : CoordTransform, optional
        The coordinate transform to apply to this cell; defaults to None
    universe_id : int, optional
        The universe id of the region; defaults to
        the root universe
    lattice : int, optional
        The lattice type; defaults to None
    fill_ids : np.ndarray of int, or None, optional
        The universe ids to fill with, defaults to None, meaning no fill;
        this is indexed as [k, j, i]
    fill_type : {"single", "array"}, or None, optional
        The type of fill, defaults to None
    fill_transforms : 3D Iterable of CoordTransform, or None, optional
        The id of the transform for each entry of the fill universe
    fill_dims : 6-tuple of int, or None, optional
        if fill_type is "array", this is the index ranges in first z,
        y, and then x dimensions; defaults to None
    other_kwargs : OrderedDict, optional
        Key/value pairs of strings for each with remaining keyword
        parameters; defaults to an empty dictionary

    Attributes
    ----------
    id : int
        The cell's ID
    material_id : int
        The ID of the material in this cell
    material : Material
        The material object itself
    density : float
        Density of the material in this cell in units of a/b-cm
    surfaces : str
        Surfaces defining the region of this cell
    volume : float or None
        Volume of the region
    coord_transform : CoordTransform, optional
        The coordinate transform to apply to this cell; defaults to None
    universe_id : int
        The universe id of the region
    lattice : int
        The lattice type
    fill_ids : np.ndarray of int, or None
        The universe ids to fill with; this is indexed as [k, j, i]
    fill : 3D Lists of {McnpUniverse or int}, or None
        The universe(s) to fill with, defaults to None; this is indexed
        as [k][j][i]. If an integer, then it is either USE_LAT_IRREG, or
        USE_LAT_MAT
    fill_type : {"single", "array"}, or None
        The type of fill
    fill_transforms : 3D Iterable of CoordTransform, or None
        The id of the transform for each entry of the fill universe
    fill_dims : 6-tuple of int, or None
        if fill_type is "array", this is the index ranges in first z,
        y, and then x dimensions
    other_kwargs : OrderedDict
        Key/value pairs of strings for each with remaining keyword
        parameters

    """

    _USED_IDS = set([])

    def __init__(self, cell_id, material_id, density, surfaces, volume=None,
                 coord_transform=None, universe_id=ROOT_UNIV,
                 lattice=None, fill_ids=None, fill_type=None,
                 fill_transforms=None, fill_dims=None,
                 other_kwargs=OrderedDict()):
        self.id = cell_id
        self.material_id = material_id
        self.material = None
        self.density = density
        self.surfaces = surfaces
        self.coord_transform = coord_transform
        self.universe_id = universe_id
        self.lattice = lattice
        self.fill_type = fill_type
        self.fill_dims = fill_dims
        self.fill_transforms = fill_transforms
        self._fill = None
        self.fill_ids = fill_ids
        self.volume = volume
        self.other_kwargs = other_kwargs

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if id_ is None:
            # Then we have to auto-generate an ID, use our generalized function
            self._id = get_id(Cell._USED_IDS, CELL_MAX_ID)
        else:
            check_type("id", id_, int)
            check_greater_than("id", id_, 0, equality=False)
            check_less_than("id", id_, CELL_MAX_ID, equality=True)
            self._id = id_
            Cell._USED_IDS.add(self._id)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, mat):
        if mat is not None:
            check_type("material", mat, Material)
            self._material = mat
            self._material_id = None
            self._density = None
        else:
            self._material = mat

    @property
    def material_id(self):
        if self._material_id is None:
            return self._material.id
        else:
            return self._material_id

    @material_id.setter
    def material_id(self, mat_id):
        if mat_id is not None:
            check_type("mat_id", mat_id, int)
            check_greater_than("mat_id", mat_id, 0, equality=True)
            check_less_than("mat_id", mat_id, MATL_MAX_ID, equality=True)
        self._material_id = mat_id

    @property
    def density(self):
        if self._density is None and self._material_id is None:
            # Get density from the assigned material in units of a/b-cm
            return self._material.neutronics_density
        else:
            return self._density

    @density.setter
    def density(self, density):
        if self._material is None:
            self._density = density
        else:
            # Raise an error because this density is already established
            # by the material object
            msg = "Cell {} already has material assigned;".format(self._id)
            msg += " density must be modified through material"
            raise AttributeError(msg)

    @property
    def volume(self):
        to_check = [self._fill_type, self._volume, self._material_id]
        if to_check.count(None) == len(to_check):
            # Above ensures all are None
            # Then we have a material object assigned, just get from
            # there if available
            if self._material.volume is not None and self._material.is_depleting:
                # Dont use the material's volume if it isnt depleting as it
                # could become out of sync since we arent duplicating these
                # materials
                return self._material.volume
            else:
                return self._volume
        else:
            return self._volume

    @volume.setter
    def volume(self, volume):
        if self.fill_type is None:
            # Then we will use the material's volume, if present
            if self._material is None:
                if volume is not None:
                    check_type("volume", volume, float)
                    check_greater_than("volume", volume, 0., equality=True)
                self._volume = volume
            else:
                # Raise an error because this volume is already established
                # by the material object
                msg = "Cell {} already has material assigned;".format(self._id)
                msg += " volume must be modified through material"
                raise AttributeError(msg)
        else:
            # Otherwise, just use the provided volume
            if volume is not None:
                check_type("volume", volume, float)
                check_greater_than("volume", volume, 0., equality=True)
            self._volume = volume

    @property
    def coord_transform(self):
        return self._coord_transform

    @coord_transform.setter
    def coord_transform(self, coord_transform):
        if coord_transform is not None:
            check_type("coord_transform", coord_transform, CoordTransform)
            # Now, if the coordinate transform is actually the identity,
            # no need to write it.
            if coord_transform.is_null:
                self._coord_transform = None
            else:
                self._coord_transform = coord_transform
        else:
            self._coord_transform = coord_transform

    @property
    def universe_id(self):
        return self._universe_id

    @universe_id.setter
    def universe_id(self, uni_id):
        check_type("uni_id", uni_id, int)
        check_greater_than("uni_id", uni_id, 0, equality=True)
        check_less_than("uni_id", uni_id, MATL_MAX_ID, equality=True)
        self._universe_id = uni_id

    @property
    def fill_type(self):
        return self._fill_type

    @fill_type.setter
    def fill_type(self, fill_type):
        check_value("fill_type", fill_type, VALID_FILL_TYPES)
        self._fill_type = fill_type

    @property
    def fill_ids(self):
        if self._fill_ids is None:
            if self._fill is not None:
                fill_ids = np.zeros_like(self.fill, dtype=int)
                for k in range(fill_ids.shape[0]):
                    for j in range(fill_ids.shape[1]):
                        for i in range(fill_ids.shape[2]):
                            univ = self._fill[k][j][i]
                            if univ not in (USE_LAT_IRREG, USE_LAT_MAT):
                                fill_ids[k, j, i] = univ.id
                            else:
                                if univ == USE_LAT_IRREG:
                                    fill_ids[k, j, i] = IRREG_LAT_ARRAY
                                elif univ == USE_LAT_MAT:
                                    fill_ids[k, j, i] = self.universe_id
                return fill_ids
            else:
                return None
        else:
            return self._fill_ids

    @fill_ids.setter
    def fill_ids(self, ids):
        if ids is not None:
            check_ndarray_shape_and_type("fill_ids", ids, self.fill_shape, int)
        self._fill_ids = ids

    @property
    def fill(self):
        return self._fill

    @property
    def fill_dims(self):
        if self.fill_type is None:
            return None
        elif self.fill_type == "single":
            return (0, 0, 0, 0, 0, 0)
        else:
            return self._fill_dims

    @fill_dims.setter
    def fill_dims(self, fill_dims):
        if fill_dims is not None:
            check_length("fill_dims", fill_dims, 6)
            check_iterable_type("fill_dims", fill_dims, int)
        self._fill_dims = fill_dims

    @property
    def fill_shape(self):
        if self._fill_type is None:
            return None
        else:
            delta1 = (self.fill_dims[1] - self.fill_dims[0]) + 1
            delta2 = (self.fill_dims[3] - self.fill_dims[2]) + 1
            delta3 = (self.fill_dims[5] - self.fill_dims[4]) + 1
            return (delta1, delta2, delta3)

    @property
    def num_fill(self):
        if self._fill_type is None:
            return 0
        else:
            fill_shape = self.fill_shape
            return fill_shape[0] * fill_shape[1] * fill_shape[2]

    @property
    def status(self):
        statuses = []
        if self.material is not None:
            statuses.append(self.material.status)

        if self.fill is not None:
            for k in range(len(self.fill)):
                for j in range(len(self.fill[k])):
                    for i in range(len(self.fill[k][j])):
                        u = self.fill[k][j][i]
                        if u not in [USE_LAT_IRREG, USE_LAT_MAT]:
                            statuses.append(u.status)

        if np.all(np.asarray(statuses) == statuses[0]):
            return statuses[0]
        else:
            msg = "Material and/or Universes of Cell {} do not have the" \
                "same status!"
            warn(msg.format(self.id))

    @status.setter
    def status(self, status):
        if self.material is not None:
            self.material.status = status

        if self.fill is not None:
            for k in range(self.fill_shape[0]):
                for j in range(self.fill_shape[1]):
                    for i in range(self.fill_shape[2]):
                        u = self.fill[k][j][i]
                        if u not in (USE_LAT_IRREG, USE_LAT_MAT):
                            u.status = status

    def assign_fill(self, universes):
        if self.fill_type is not None:
            # Use a quick way to get a 3d list of the right dimensions
            fill_lists = np.zeros_like(self.fill_ids).tolist()
            for k in range(self.fill_shape[0]):
                for j in range(self.fill_shape[1]):
                    for i in range(self.fill_shape[2]):
                        u_id = self.fill_ids[k, j, i]
                        if u_id == USE_LAT_IRREG or u_id == USE_LAT_MAT:
                            u = u_id
                        else:
                            u = universes[u_id]
                        fill_lists[k][j][i] = u
            self._fill = fill_lists
            self._fill_ids = None
        else:
            self._fill = None

    def __str__(self):
        return self.to_str()

    def to_str(self, is_volume_calc=False):
        """Reproduces the cell as a string to be used in the MCNP
        input.

        Parameters
        ----------
        is_volume_calc : bool, optional
            If this is for a volume calculation, we need 0 importance
            cells to have a non-zero importance so that our randomly
            chosen samples, if outside the geometry, do not cause MCNP
            to quit. We also want to not print the volume so that we
            can see if MCNP computes an analytic value to compare with.
            Defaults to False

        Returns
        -------
        cell_str : str
            The cell input file string
        """

        # Get the non-keyword args
        cell_str = "{} {} ".format(self.id, self.material_id)
        if self.material_id != 0:
            cell_str += CELL_DENSITY_FMT.format(self.density, self.surfaces)
        cell_str += " {}".format(self.surfaces)

        # Do the keywords that are straight-forward
        if self.volume is not None:
            cell_str += " vol={}".format(self.volume)
        else:
            # We need to get the volume from the material if present
            if self.material is not None:
                if self.material.volume is not None:
                    cell_str += " vol={}".format(self.material.volume)
        cell_str += " u={}".format(self.universe_id)

        if self.lattice is not None:
            cell_str += " lat={}".format(self.lattice)

        for key, val in self.other_kwargs.items():
            perturbed_val = val
            if is_volume_calc:
                # Then intercept the importance parameters, and if 0,
                # write a 1
                if key.startswith("imp:"):
                    if num_format(val, 'float') <= 0.:
                        perturbed_val = 1.
            cell_str += " {}={}".format(key, perturbed_val)

        # Now do TRCL
        if self.coord_transform is not None:
            cell_str += " trcl={}".format(self.coord_transform.id)

        # Finally, the lattice
        if self.fill_type == "single":
            # Nice and easy, we have a univ and a transform, possible
            fill_id = self.fill_ids[0][0][0]
            if fill_id == USE_LAT_MAT:
                fill_id = self.universe_id
            elif fill_id == USE_LAT_IRREG:
                fill_id = 0
            fill_transform = self.fill_transforms[0][0][0]
            if fill_transform is None:
                cell_str += " fill={}".format(fill_id)
            else:
                cell_str += " fill={} ({})".format(fill_id, fill_transform.id)
        elif self.fill_type == "array":
            # Start with the array dimensions
            # MCNP wants these printed such that the order is x, y, z so
            # we must reverse from how fill_dims is
            d = self.fill_dims
            cell_str += " fill={}:{} {}:{} {}:{}".format(d[4], d[5], d[2],
                                                         d[3], d[0], d[1])

            # Now move on to the array itself
            for f_id, f_tform in zip(self.fill_ids.flatten(),
                                     np.array(self.fill_transforms).flatten()):
                if f_id == USE_LAT_MAT:
                    f_id = self.universe_id
                elif f_id == USE_LAT_IRREG:
                    f_id = 0
                if f_tform is None:
                    cell_str += " {}".format(f_id)
                else:
                    cell_str += " {}({})".format(f_id, f_tform.id)

        return cell_str

    def get_transforms(self):
        """Returns the transforms used by this cell, as a list

        Returns
        -------
        transforms : List of CoordTransform
            The list of transforms used in this cell
        """

        if self.coord_transform is not None:
            transforms = [self.coord_transform]
        else:
            transforms = []

        if self.fill is not None:
            for k in range(len(self.fill_transforms)):
                for j in range(len(self.fill_transforms[k])):
                    for i in range(len(self.fill_transforms[k][j])):
                        if self.fill_transforms[k][j][i] is not None:
                            transforms.append(self.fill_transforms[k][j][i])
        return transforms

    def update_transforms(self, old_tr, new_tr):
        # Looks through all transforms and replaces old_tr instances with
        # new_tr instances
        if self._coord_transform == old_tr:
            self.coord_transform = new_tr

        if self.fill is not None:
            for k in range(len(self.fill_transforms)):
                for j in range(len(self.fill_transforms[k])):
                    for i in range(len(self.fill_transforms[k][j])):
                        if self.fill_transforms[k][j][i] == old_tr:
                            self.fill_transforms[k][j][i] = new_tr

    def clone(self, depl_libs, model_univs, model_cells, model_mats):
        """Create a copy of this Cell with a new unique ID and new IDs
        for all constituents, who are also cloned.

        Parameters
        ----------
        depl_libs : OrderedDict of DepletionLibrary
            The depletion libraries in use in the model
        model_univs : Dict
            The universes in use in the model, keyed by their ID
        model_cells : Dict
            The cells in use in the model, keyed by their ID
        model_mats : List
            The materials in use in the model

        Returns
        -------
        clone : Cell
            The clone of this cell
        logs : Iterable of (str, str, int or None)
            The log messages to pass up to a higher level of executor

        """

        # Get the initialization arguments, aside from cell_id
        material_id = deepcopy(self.material_id)
        density = deepcopy(self.density)
        surfaces = deepcopy(self.surfaces)
        volume = deepcopy(self.volume)
        coord_transform = deepcopy(self.coord_transform)
        universe_id = deepcopy(self.universe_id)
        lattice = deepcopy(self.lattice)
        fill_ids = deepcopy(self.fill_ids)
        fill_type = deepcopy(self.fill_type)
        fill_transforms = deepcopy(self.fill_transforms)
        fill_dims = deepcopy(self.fill_dims)
        other_kwargs = deepcopy(self.other_kwargs)

        # Now set the new id to be figured out by the setter
        cell_id = None

        # Create the clone
        clone = Cell(cell_id, material_id, density, surfaces, volume,
                     coord_transform, universe_id, lattice, fill_ids,
                     fill_type, fill_transforms, fill_dims, other_kwargs)

        msg = "Cell {} cloned as {}".format(self.id, clone.id)
        logs = [("info_file", msg, None)]
        # Update the global cells
        model_cells[clone.id] = clone

        # Now clone the children
        if self.material is not None:
            # Then we make a clone
            clone.material = self.material.clone(depl_libs)
            msg = "Material {} cloned as material {}".format(
                self.material.name, clone.material.name)
            logs.append(("info_file", msg, None))
            # Add to the global list
            model_mats.append(clone.material)
        if self.fill is not None:
            fill = np.empty_like(self.fill).tolist()
            for k in range(self.fill_shape[0]):
                for j in range(self.fill_shape[1]):
                    for i in range(self.fill_shape[2]):
                        u = self.fill[k][j][i]
                        if u == USE_LAT_IRREG or u == USE_LAT_MAT:
                            fill[k][j][i] = u
                        else:
                            new_u, new_logs = \
                                u.clone(depl_libs, model_univs, model_cells,
                                        model_mats)
                            fill[k][j][i] = new_u
                            logs.extend(new_logs)
            clone._fill = fill
            clone._fill_ids = None

        return clone, logs

    def get_all_materials(self):
        mats = OrderedDict()
        # Traverses through the cell's hierarchy to find all the mats
        if self.material is not None:
            mats[self.material_id] = self.material

        # Now traverse through the universes in fill
        if self.fill is not None:
            for k in range(self.fill_shape[0]):
                for j in range(self.fill_shape[1]):
                    for i in range(self.fill_shape[2]):
                        fill = self.fill[k][j][i]
                        if fill not in (USE_LAT_IRREG, USE_LAT_MAT):
                            univ_mats = fill.get_all_materials()
                            for key, value in univ_mats.items():
                                mats[key] = value
        return mats

    def get_all_cells(self):
        cells = OrderedDict()
        # Traverses through the cell's hierarchy to find the cells
        if self.fill is not None:
            for k in range(self.fill_shape[0]):
                for j in range(self.fill_shape[1]):
                    for i in range(self.fill_shape[2]):
                        fill = self.fill[k][j][i]
                        if fill not in (USE_LAT_IRREG, USE_LAT_MAT):
                            univ_cells = fill.get_all_cells()
                            for key, value in univ_cells.items():
                                cells[key] = value
        return cells

    @classmethod
    def from_not_like_line(cls, line, transforms):
        """ Initializes a Cell object from an input file line that does
        NOT use a LIKE * BUT form.

        Parameters
        ----------
        line : str
            The input file line to parse to initialize this cell.
            The line should be uncommented, combined (if it was multi-line),
            and have repeated entries expanded.
        transforms : dict
            The parsed transforms (CoordTransform objects) keyed by
            their id

        Returns
        -------
        this : Cell
            The initialized Cell

        """

        # Set the default values for the keywords that need it
        surfaces = None
        volume = None
        transform = None
        universe_id = 0
        lattice = None
        fill_ids = None
        fill_type = None
        fill_transforms = None
        fill_dims = None
        unparsed_keywords = {}

        # Per Section 3.2, Form 1 of a cell card is defined as:
        # "<cell id> <material id> <density> <geom> <...params...>" where
        # <density> is not provided if the cell is a void

        # Get the first two entries of the cell card, the cell # and
        # the material number (or void indicator)
        split_card = line.split(maxsplit=2)
        cell_id = num_format(split_card[0], 'int')
        material_id = num_format(split_card[1], 'int')

        if material_id > 0:
            # Then we do not have a void and the next entry is density
            split_card = split_card[-1].split(maxsplit=1)
            density = num_format(split_card[0], 'float')
        else:
            # Then we do have a void and density is 0
            density = 0.

        surfaces, volume, transform, universe_id, lattice, fill_ids, \
            fill_type, fill_transforms, fill_dims, unparsed_keywords = \
            _parse_line(split_card[-1], surfaces, volume, transform,
                        universe_id, lattice, fill_ids, fill_type,
                        fill_transforms, fill_dims, unparsed_keywords,
                        transforms)

        this = cls(cell_id, material_id, density, surfaces, volume, transform,
                   universe_id, lattice, fill_ids, fill_type,
                   fill_transforms, fill_dims, unparsed_keywords)

        return this

    @classmethod
    def from_like_line(cls, line, other_cells, transforms):
        """ Initializes a Cell object from an input file line

        Parameters
        ----------
        line : str
            The input file line to parse to initialize this cell.
            The line should be uncommented, combined (if it was
            multi-line), and have repeated entries expanded.
        other_cells : OrderedDict of Cell objects
            The set of already initialized cells to find the "like" match
        transforms : dict
            The parsed transforms (CoordTransform objects) keyed by
            their id

        Returns
        -------
        this : Cell
            The initialized Cell

        """

        # Get the first two entries of the cell card, the cell # and
        # the other cell to be like
        split_card = line.split(maxsplit=4)
        # Should yield: ["j", "LIKE", "n", "BUT", "..."]
        cell_id = num_format(split_card[0], 'int')
        like_id = num_format(split_card[2], 'int')

        found_it = False
        for i, c in enumerate(other_cells):
            if c.id == like_id:
                like_cell_index = i
                found_it = True
                break

        if not found_it:
            raise KeyError("Illegal LIKE BUT Cell card: " +
                           "Cell {} not found".format(like_id))
        like_cell = other_cells[like_cell_index]

        # So now we set our defaults to the values from like_cell, and
        # then continue parsing the line
        material_id = like_cell.material_id
        density = like_cell.density
        surfaces = like_cell.surfaces
        volume = like_cell.volume
        transform = like_cell.coord_transform
        universe_id = like_cell.universe_id
        lattice = like_cell.lattice
        fill_ids = like_cell.fill_ids
        fill_type = like_cell.fill_type
        fill_transforms = like_cell.fill_transforms
        fill_dims = like_cell.fill_dims
        orig_unparsed_keywords = like_cell.other_kwargs

        # Parse the rest of the line
        surfaces, volume, transform, universe_id, lattice, fill_ids, \
            fill_type, fill_transforms, fill_dims, unparsed_keywords = \
            _parse_line(split_card[-1], surfaces, volume, transform,
                        universe_id, lattice, fill_ids, fill_type,
                        fill_transforms, fill_dims, orig_unparsed_keywords,
                        transforms, is_like=True)

        # The material and rho keywords, if different than default, will
        # be in unparsed_keywords
        if "mat" in unparsed_keywords:
            material_id = num_format(unparsed_keywords["mat"], 'int')
            del unparsed_keywords["mat"]

        if material_id > 0:
            if "rho" in unparsed_keywords:
                # Then we can also check for density
                density = num_format(unparsed_keywords["rho"], 'float')
                del unparsed_keywords["rho"]
            # Otherwise, the like material's density still holds
        else:
            # Then we do have a void and density is 0
            density = 0.

        this = cls(cell_id, material_id, density, surfaces, volume, transform,
                   universe_id, lattice, fill_ids, fill_type, fill_transforms,
                   fill_dims, unparsed_keywords)

        return this


def _parse_line(rest_of_line, surfaces, volume, transform, universe_id,
                lattice, fill_ids, fill_type, fill_transforms, fill_dims,
                unparsed_keywords, transforms, is_like=False):
    """This method assumes the default values have been set and just
    populates the data before returning it
    """

    # At this point, the remainder of our split cards is the
    # geometry and parameters data
    geom_params = rest_of_line

    if not is_like:
        surfaces, keywords = split_data_and_keywords(geom_params, CELL_KW)

    else:
        _, keywords = split_data_and_keywords(geom_params, LIKE_CELL_KW)

    # Now set the keywords we want to keep
    for key, value in keywords.items():
        if key == "vol":
            volume = num_format(value, 'float')
        elif key == "trcl" or key == "*trcl":
            if value.startswith("(") and value.endswith(")"):
                # Then the TRCL array is included here
                # We need to instatiate a new transform
                transform = CoordTransform.from_card(key, value)
                # Store the new transform in the global set
                transforms[transform.id] = transform
            else:
                # Then this is just a pointer to a TR card of the
                # id given in value
                t_id = num_format(value.strip(), 'int')
                transform = transforms[t_id]
        elif key == "u":
            universe_id = abs(num_format(value, 'int'))
        elif key == "lat":
            lattice = num_format(value, 'int')
            # Since we need to know if there is a lattice before we
            # grab fill data, we will do "fill" after the loop
        elif key not in ("fill", "*fill"):
            unparsed_keywords[key] = value

    # Now get the fill and lattice info
    fill_ids, fill_type, fill_dims, fill_transforms = \
        _parse_fill(fill_ids, fill_type, fill_dims, fill_transforms, keywords,
                    lattice is not None, transforms, universe_id)

    return (surfaces, volume, transform, universe_id, lattice, fill_ids,
            fill_type, fill_transforms, fill_dims, unparsed_keywords)


def _parse_fill(fill_ids, fill_type, fill_dims, fill_transforms, keywords,
                has_lattice, transforms, universe_id):
    # Parses the fill data to extract the fill type, fill
    # dimensionality, fill universe ids, and any fill transforms

    # First check the fill parameter for correctness
    if "fill" not in keywords and "*fill" not in keywords:
        if has_lattice:
            raise ValueError("A fill parameter must be provided with "
                             "lattice cells")
    if "fill" in keywords and "*fill" in keywords:
        msg = "Only one of `fill` and `*fill` can be present on a " + \
            "cell card"
        raise ValueError(msg)

    # Now get the fill card that is present
    if "fill" in keywords:
        fill_card = keywords["fill"]
        has_star = False
    elif "*fill" in keywords:
        fill_card = keywords["*fill"]
        has_star = True
    else:
        return fill_ids, fill_type, fill_dims, fill_transforms

    # If there is a colon, then we know this is an array
    trial_fill_card = fill_card.split(":")

    if len(trial_fill_card) <= 2:
        # Then we have either a single universe, or a
        # single universe and, potentially, a transformation #
        fill_type = "single"

        fill_id, fill_transform = \
            _parse_specific_fill_entry(fill_card, has_star, transforms)

        fill_ids = np.array([[[fill_id]]])
        fill_transforms = [[[fill_transform]]]
        fill_dims = None

    else:
        fill_type = "array"
        # We want the array dimensions and the values so lets first
        # separate the dims from the values
        last_colon_index = fill_card.rfind(":")
        end_dim = fill_card.find(" ", last_colon_index)
        dim_strings = fill_card[:end_dim].split(" ")
        # Now we will get the dimensions themselves
        # Note that since the fill array progreses through
        # i, j, then k (i.e., the numpy array should be [k, j, i]),
        # but the dimensionality provided in dims is ordered i, j, k
        # we need to reverse the orders that we grab dims from below

        fill_dims = []
        for dim_string in reversed(dim_strings):
            low_high_string = dim_string.split(":")
            fill_dims.append(num_format(low_high_string[0], 'int'))
            fill_dims.append(num_format(low_high_string[1], 'int'))

        # Find the dimensions
        N0 = fill_dims[1] - fill_dims[0] + 1
        N1 = fill_dims[3] - fill_dims[2] + 1
        N2 = fill_dims[5] - fill_dims[4] + 1

        # The remainder of fill_card contains the numbers, and
        # sometimes, the fill trcl. The latter are denoted by a "()"
        entry_strings = fill_card[end_dim + 1:].split(" ")
        # Now entry_strings contains each entry in the fill array

        fill_ids = np.zeros((N0, N1, N2), dtype=int)
        fill_transforms = np.empty_like(fill_ids).tolist()
        index = 0
        for k in range(N0):
            for j in range(N1):
                for i in range(N2):
                    fill_id, fill_transform = \
                        _parse_specific_fill_entry(entry_strings[index],
                                                   has_star, transforms)
                    fill_ids[k, j, i] = fill_id
                    fill_transforms[k][j][i] = fill_transform
                    if fill_ids[k, j, i] == IRREG_LAT_ARRAY:
                        fill_ids[k, j, i] = USE_LAT_IRREG
                        fill_transforms[k][j][i] = None
                    elif fill_ids[k, j, i] == universe_id:
                        fill_ids[k, j, i] = USE_LAT_MAT
                        fill_transforms[k][j][i] = None
                    index += 1

    return fill_ids, fill_type, fill_dims, fill_transforms


def _parse_specific_fill_entry(fill_card, has_star, transforms):
    left_parenth = fill_card.find("(")
    right_parenth = fill_card.find(")")
    if left_parenth != -1 and right_parenth != -1:
        # Then we have a fill transform and a fill universe id
        fill_id = num_format(fill_card[:left_parenth].strip(), 'int')
        # Get the contents of the "(...)", without the ()
        fill_transform_card = \
            fill_card[left_parenth + 1: right_parenth]
        fill_transform_data = fill_transform_card.strip().split()
        # If there is only one entry in fill_transform_data,
        # then it is refers to a TR card. Otherwise it is new
        # TR data
        if len(fill_transform_data) > 1:
            if has_star:
                key = "*trcl"
            else:
                key = "trcl"
            fill_transform = \
                CoordTransform.from_card(key, fill_transform_card)
            transforms[fill_transform.id] = fill_transform
        else:
            fill_transform = transforms[num_format(fill_transform_data[0],
                                                   'int')]
    else:
        # No transform, just the fill id
        fill_id = num_format(fill_card.split(" ")[0], 'int')
        fill_transform = transforms[0]

    if fill_transform.is_null:
                fill_transform = None

    return fill_id, fill_transform
