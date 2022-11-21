from adder.type_checker import *
from adder.loggedclass import LoggedClass
from adder.constants import VALID_GEOM_SWEEP_AXES, VALID_GEOM_SWEEP_TYPES, \
    VALID_ANGLE_TYPES


class ControlGroup(LoggedClass):
    """Class containing data and current state of a control group

    Parameters
    ----------
    name : str
        The name of the control group
    input_data : dict
        The parsed and checked input file data

    Attributes
    ----------
    name : str
        The name of the control group
    type : str
        The type of objects represented by this
    axis : str
        The axis the control group works along
    set : List of str
        The ids of the entries in this set.
    angle_units : str
        The units to use for angular axes.
    displacement : float
        The displacement of the control group from its' original value along
        the axis defined by the ``axis`` parameter.
    """

    def __init__(self, name, input_data):
        self.name = name
        self.type = input_data["type"]
        self.axis = input_data["axis"]
        self.set = input_data["set"]
        self.angle_units = input_data["angle_units"]
        self.displacement = 0.

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        check_type("name", name, str)
        self._name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_):
        check_value("type", type_, VALID_GEOM_SWEEP_TYPES)
        self._type = type_

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis_):
        check_value("axis", axis_, VALID_GEOM_SWEEP_AXES)
        self._axis = axis_

    @property
    def set(self):
        return self._set

    @set.setter
    def set(self, set_):
        if isinstance(set_, str):
            msg = 'Unable to set "{0}" to "{1}" which is not Iterable'
            raise TypeError(msg.format(set_, "set"))
        check_iterable_type("set", set_, str)
        self._set = set_

    @property
    def angle_units(self):
        return self._angle_units

    @angle_units.setter
    def angle_units(self, angle_units_):
        check_value("angle_units", angle_units_, VALID_ANGLE_TYPES)
        self._angle_units = angle_units_

    @property
    def displacement(self):
        return self._displacement

    @displacement.setter
    def displacement(self, displacement):
        check_type("displacement", displacement, float)
        self._displacement = displacement

    def to_hdf5(self, group):
        """Writes the control group to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to

        """

        self_grp = group.create_group(self.name, track_order=True)
        self_grp.attrs["name"] = np.string_(self.name)
        self_grp.attrs["type"] = np.string_(self.type)
        self_grp.attrs["axis"] = np.string_(self.axis)
        self_grp.attrs["set"] = self.set
        self_grp.attrs["angle_units"] = np.string_(self.angle_units)
        self_grp.attrs["displacement"] = self.displacement

    @classmethod
    def from_hdf5(cls, group):
        name = group.name.split("/")[-1]
        type_ = group.attrs["type"].decode()
        axis = group.attrs["axis"].decode()
        set_ = group.attrs["set"].tolist()
        angle_units = group.attrs["angle_units"].decode()
        displacement = float(group.attrs["displacement"])

        input_data = {}
        input_data["type"] = type_
        input_data["axis"] = axis
        input_data["set"] = set_
        input_data["angle_units"] = angle_units

        this = cls(name, input_data)
        this.displacement = displacement

        return this
