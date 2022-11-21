from collections import OrderedDict

from adder.utils import get_id
from adder.constants import ALLOWED_STATUSES
from .constants import USE_LAT_MAT, USE_LAT_IRREG, UNIV_MAX_ID
from adder.type_checker import *
from .cell import Cell


class Universe(object):
    """This class contains relevant information about a universe.

    Parameters
    ----------
    name : str
        The name of the universe.
    id_ : int
        The identifier of the universe, as referred to in the neutronics
        solver.

    Attributes
    ----------
    name : str
        The name of the universe.
    id_ : int
        The identifier of the universe, as referred to in the neutronics
        solver.
    status : {IN_CORE, STORAGE, SUPPLY}
        Whether the universe is in-core, from the supply chain or in
        storage.
    cells : OrderedDict
        The cells contained by this universe; indexed by cell id,
        value is the Cell
    nested_materials : OrderedDict
        All nested materials contained in this universe; indexed by cell id,
        and a value of the Material
    nested_cells : OrderedDict
        All nested cells contained in this universe; indexed by cell id, and
        a value of the Cell
    nested_universe_ids : Iterable of int
        All nested universe ids contained in this universe.
    """

    _USED_IDS = set([])

    def __init__(self, name, id_):
        self.name = name
        self.id = id_
        self.num_copies = 0
        self.cells = OrderedDict()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if id_ is None:
            # Then we have to auto-generate an ID, use our generalized function
            self._id = get_id(Universe._USED_IDS, UNIV_MAX_ID)
        else:
            check_type("id", id_, int)
            check_greater_than("id", id_, 0, equality=True)
            check_less_than("id", id_, UNIV_MAX_ID, equality=True)
            self._id = id_
            Universe._USED_IDS.add(self._id)

    @property
    def status(self):
        statuses = [cell.status for cell in self.cells.values()]

        if np.all(np.asarray(statuses) == statuses[0]):
            return statuses[0]
        else:
            msg = "Material and/or Universes of Cell {} do not have the" \
                " same status!"
            raise ValueError(msg.format(self.id))

    @status.setter
    def status(self, status):
        check_value("status", status, ALLOWED_STATUSES)
        for cell in self.cells.values():
            cell.status = status

    def add_cells(self, cells):
        for cell in cells:
            check_type("cells", cell, Cell)
            if cell.id not in self.cells:
                self.cells[cell.id] = cell
                cell.universe_id = self.id
            else:
                msg = "Cell {} already exists in Universe {}"
                raise ValueError(msg.format(cell.name, self.name))

    def clone(self, depl_libs, model_univs, model_cells,
              model_mats, new_name=None):
        """Create a copy of this Universe with a new unique ID and new
        IDs for all constituents, who are also cloned.

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
        new_name : str or None, optional
            The name to apply to the clone; if None then [#] will be
            appended where # is the number of copies

        Returns
        -------
        clone : Universe
            The clone of this Universe
        logs : Iterable of (str, str, int or None)
            The log messages to pass up to a higher level of executor

        """

        # Set the new id to be determined by the setter
        id_ = None

        # Update the number of copies
        self.num_copies += 1

        # Set the new name
        if new_name is None:
            new_u_name = self.name + "[{}]".format(self.num_copies)
        else:
            new_u_name = new_name

        clone = Universe(new_u_name, id_)
        model_univs[clone.id] = clone

        # Now clone the children
        cloned_cells = []
        msg = "Universe {} cloned as {}".format(self.name, clone.name)
        logs = [("info_file", msg, None)]
        for cell in self.cells.values():
            cloned_cell, cell_logs = cell.clone(depl_libs, model_univs,
                model_cells, model_mats)
            cloned_cells.append(cloned_cell)
            logs.extend(cell_logs)
        clone.add_cells(cloned_cells)

        return clone, logs

    def __repr__(self):
        return "<Universe Name: {}, Id: {}>".format(self.name, self.id)

    def update_nested_children(self):
        self.nested_materials = self.get_all_materials()
        self.nested_cells = self.get_all_cells()
        self.nested_universe_ids = self.get_all_universe_ids(self.nested_cells)

    def get_all_materials(self):
        mats = OrderedDict()
        # Traverses through the universe hierarchy to figure out all
        # materials within this universe
        for cell in self.cells.values():
            cell_mats = cell.get_all_materials()
            for key, value in cell_mats.items():
                mats[key] = value

        return mats

    def get_all_cells(self):
        cells = OrderedDict()
        # Traverses through the universe hierarchy to figure out all
        # cells within this universe
        for cell in self.cells.values():
            # Start with the obvious
            cells[cell.id] = cell

            # Now traverse these cells
            cells_cells = cell.get_all_cells()
            for key, value in cells_cells.items():
                cells[key] = value

        return cells

    def get_all_universe_ids(self, nested_cells=None):
        universe_ids = []

        if nested_cells:
            all_cells = nested_cells
        else:
            all_cells = self.get_all_cells()
        for cell in all_cells.values():
            if cell.universe_id != self.id:
                universe_ids.append(cell.universe_id)
            if cell.fill_ids is not None:
                dataset = cell.fill_ids.flatten().tolist()
                for u_id in dataset:
                    if u_id not in [USE_LAT_IRREG, USE_LAT_MAT]:
                        universe_ids.append(u_id)

        return universe_ids
