from adder.control_group import ControlGroup
import pytest
import numpy as np
import h5py


def test_cg_init():
    name = "bank 1"
    type_ = "surface"
    axis = "x"
    set_ = ["1", "2"]
    angle_units = "radians"
    input_data = {"type": type_, "axis": axis, "set": set_,
                  "angle_units": angle_units}

    # Now check type-checking
    # name
    with pytest.raises(TypeError):
        cg = ControlGroup(1, input_data)
    # type
    input_data["type"] = 1
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["type"] = "not a surface"
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["type"] = type_
    # axis
    input_data["axis"] = 1
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["axis"] = "not x"
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["axis"] = axis
    # set
    input_data["set"] = "123"
    with pytest.raises(TypeError):
        cg = ControlGroup(name, input_data)
    input_data["set"] = [False, "1"]
    with pytest.raises(TypeError):
        cg = ControlGroup(name, input_data)
    input_data["set"] = set_
    # angle_units
    input_data["angle_units"] = 1
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["angle_units"] = "not degrees"
    with pytest.raises(ValueError):
        cg = ControlGroup(name, input_data)
    input_data["angle_units"] = angle_units

    # Now initialize and check the values of each
    cg = ControlGroup(name, input_data)
    assert cg.name == name
    assert cg.type == type_
    assert cg.axis == axis
    assert cg.set == set_
    assert cg.angle_units == angle_units
    assert cg.displacement == 0.

    # Check setting of displacement with wrong type
    with pytest.raises(TypeError):
        cg.displacement = "not a float"
    with pytest.raises(TypeError):
        cg.displacement = 1
    assert cg.displacement == 0.
    cg.displacement += 10.
    assert cg.displacement == 10.


def test_cg_hdf5():
    # This test will be performed by initializing a test object
    # writing to an HDF5 file, reading it back in, and then comparing
    # values
    name = "bank 1"
    type_ = "surface"
    axis = "x"
    set_ = ["1", "2"]
    angle_units = "radians"
    input_data = {"type": type_, "axis": axis, "set": set_,
                  "angle_units": angle_units}

    # Make the starting contrl group
    orig_cg = ControlGroup(name, input_data)
    orig_cg.displacement = 7.

    # Now write it to an hdf5 file
    with h5py.File("test.h5", "w") as temp_h5:
        temp_grp = temp_h5.create_group("control_groups")
        orig_cg.to_hdf5(temp_grp)

    # Now reopen the file and recreate the control group to test
    with h5py.File("test.h5", "r") as temp_h5:
        temp_grp = temp_h5["control_groups/" + name]
        test_cg = ControlGroup.from_hdf5(temp_grp)

    assert orig_cg is not test_cg
    assert orig_cg.name == test_cg.name
    assert orig_cg.type == test_cg.type
    assert orig_cg.axis == test_cg.axis
    assert orig_cg.set == test_cg.set
    assert orig_cg.angle_units == test_cg.angle_units
    assert orig_cg.displacement == test_cg.displacement
