from adder.mcnp.surface import Surface
from adder.mcnp.coord_transform import CoordTransform
import pytest


def test_surface_init():
    def_id = 1
    def_xform = CoordTransform(id_=1)
    def_type = "cx"
    def_params = "10."
    def_btype = "transmission"
    def_periodic_id = None

    # CHECK TYPE-CHECKING
    # surf_id
    with pytest.raises(TypeError):
        test_s = Surface("a", def_xform, def_type, def_params, def_btype,
                          def_periodic_id)
    with pytest.raises(ValueError):
        test_s = Surface(0, def_xform, def_type, def_params, def_btype,
                          def_periodic_id)
    with pytest.raises(ValueError):
        test_s = Surface(9999999999999, def_xform, def_type, def_params,
                          def_btype, def_periodic_id)
    # transform
    with pytest.raises(TypeError):
        test_s = Surface(def_id, "s", def_type, def_params, def_btype,
                          def_periodic_id)
    test_xform = CoordTransform(id_=2)
    test_xform._id = 100000
    with pytest.raises(ValueError):
        test_s = Surface(def_id, test_xform, def_type, def_params, def_btype,
                          def_periodic_id)
    # type
    with pytest.raises(TypeError):
        test_s = Surface(def_id, def_xform, 1, def_params, def_btype,
                          def_periodic_id)
    with pytest.raises(ValueError):
        test_s = Surface(def_id, def_xform, "test", def_params, def_btype,
                          def_periodic_id)
    # params
    with pytest.raises(TypeError):
        test_s = Surface(def_id, def_xform, def_type, 0, def_btype,
                          def_periodic_id)
    # boundary type
    with pytest.raises(TypeError):
        test_s = Surface(def_id, def_xform, def_type, def_params, 0,
                          def_periodic_id)
    with pytest.raises(ValueError):
        test_s = Surface(def_id, def_xform, def_type, def_params, "notright",
                          def_periodic_id)
    with pytest.raises(ValueError):
        # Make sure we get the error for requesting periodic w/ a transform
        test_s = Surface(def_id, def_xform, def_type, def_params, "periodic",
                          def_periodic_id)
    # periodic id
    with pytest.raises(TypeError):
        test_s = Surface(def_id, None, def_type, def_params, "periodic",
                          "0")
    with pytest.raises(ValueError):
        test_s = Surface(def_id, None, def_type, def_params, "periodic",
                          0)
    with pytest.raises(ValueError):
        test_s = Surface(def_id, None, def_type, def_params, "periodic",
                          100000000000)
    # Do without a periodic id but with a periodic boundary
    with pytest.raises(ValueError):
        test_s = Surface(def_id, None, def_type, def_params, "periodic", None)
    # Do without a periodic boundary but with a periodic id
    with pytest.raises(ValueError):
        test_s = Surface(def_id, None, def_type, def_params, "vacuum", 1)

    # Now set the defaults and make sure values are as expected
    test_s = Surface(def_id, def_xform, def_type, def_params, def_btype,
                     def_periodic_id)
    assert test_s.id == def_id
    assert test_s.coord_transform == def_xform
    assert test_s.type == def_type
    assert test_s.params == def_params
    assert test_s.boundary_type == def_btype
    assert test_s.periodic_id == def_periodic_id

    # Now check some perturbations:
    # - A transform and a periodic (done above, not repeated)
    # - No transform, not periodic
    test_s = Surface(def_id, None, def_type, def_params, def_btype,
                     def_periodic_id)
    assert test_s.id == def_id
    assert test_s.coord_transform is None
    assert test_s.type == def_type
    assert test_s.params == def_params
    assert test_s.boundary_type == def_btype
    assert test_s.periodic_id == def_periodic_id
    # - No transform, periodic, no periodic id (done above, not repeated)
    # - No transform, periodic, with periodic id
    test_s = Surface(def_id, None, def_type, def_params, "periodic", 1)
    assert test_s.id == def_id
    assert test_s.coord_transform is None
    assert test_s.type == def_type
    assert test_s.params == def_params
    assert test_s.boundary_type == "periodic"
    assert test_s.periodic_id == 1
    # - All BC types
    for bc in ['transmission', 'vacuum', 'reflective', 'periodic', 'white']:
        if bc == "periodic":
            p_id = 1
        else:
            p_id = None
        test_s = Surface(def_id, None, def_type, def_params, bc, p_id)
        assert test_s.boundary_type == bc
        assert test_s.periodic_id == p_id

    # - All surf types
    for st in ["p", "px", "py", "pz", "so", "s", "sx", "sy", "sz", "c/x",
               "c/y", "c/z", "cx", "cy", "cz", "k/x", "k/y", "k/z", "kx", "ky",
               "kz", "sq", "gq", "tx", "ty", "ty", "x", "y", "z", "p", "box",
               "rpp", "sph", "rcc", "rhp", "hex", "rec", "trc", "ell", "wed",
               "arb"]:
        test_s = Surface(def_id, def_xform, st, def_params, def_btype,
                         def_periodic_id)
        assert test_s.type == st


def test_surface_str():
    # We already tested getting the values, now we just need to verify we
    # print the right values
    # Will do for each boundary type, and a tranform and not
    xform = CoordTransform(id_=27)
    xform_str = "tr27 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1"
    for bc in ["transmission", "vacuum"]:
        # vacuum/transmission, no transform:
        ref_string = "1 px 1.0"
        test_s = Surface(1, None, "px", "1.0", bc, None)
        assert ref_string == str(test_s)
        assert ref_string == test_s.to_str()

        # vacuum/transmission, transform:
        ref_string = "2 27 px 1.0"
        test_s = Surface(2, xform, "px", "1.0", bc, None)
        assert ref_string == str(test_s)
        assert xform_str == str(test_s.coord_transform)
        assert ref_string == test_s.to_str()

    # reflective, no transform:
    ref_string = "*1 px 1.0"
    test_s = Surface(1, None, "px", "1.0", "reflective", None)
    assert ref_string == str(test_s)
    assert ref_string == test_s.to_str()

    # reflective, transform:
    ref_string = "*2 27 px 1.0"
    test_s = Surface(2, xform, "px", "1.0", "reflective", None)
    assert ref_string == str(test_s)
    assert xform_str == str(test_s.coord_transform)
    assert ref_string == test_s.to_str()

    # white, no transform:
    ref_string = "+1 px 1.0"
    test_s = Surface(1, None, "px", "1.0", "white", None)
    assert ref_string == str(test_s)
    assert ref_string == test_s.to_str()

    # white, transform:
    ref_string = "+2 27 px 1.0"
    test_s = Surface(2, xform, "px", "1.0", "white", None)
    assert ref_string == str(test_s)
    assert xform_str == str(test_s.coord_transform)
    assert ref_string == test_s.to_str()

    # periodic, no transform:
    ref_string = "1 -5 px 1.0"
    test_s = Surface(1, None, "px", "1.0", "periodic", 5)
    assert ref_string == str(test_s)
    assert ref_string == test_s.to_str()


def test_surface_from_cards():
    # Test the from_card method
    # we will need a transform for this
    transforms = {27: CoordTransform(id_=27), 34: CoordTransform(id_=34)}

    # First check the initial type-checking of arguments
    with pytest.raises(TypeError):
        test_s = Surface.from_card(1, transforms)
    with pytest.raises(TypeError):
        test_s = Surface.from_card("1 px 1.0", 1)
    with pytest.raises(TypeError):
        test_s = Surface.from_card("1 px 1.0", [1])
    with pytest.raises(TypeError):
        test_s = Surface.from_card("1 px 1.0", {1: 1})
    with pytest.raises(TypeError):
        test_s = Surface.from_card("1 px 1.0", {"1": 1})

    # There will be an error if we put in an invalid first value.
    with pytest.raises(ValueError):
        test_s = Surface.from_card("j 7 px 1.0", transforms)

    # There also will be an error if we put in an invalid transform number.
    with pytest.raises(ValueError):
        test_s = Surface.from_card("1 7 px 1.0", transforms)

    # Now we will test the init with each boundary type, and a tranform and not
    # vacuum/transmission, no transform:
    card = "1 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform is None
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "transmission"
    assert test_s.periodic_id is None

    # transmission but with a transformation
    card = "1 27 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform == transforms[27]
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "transmission"
    assert test_s.periodic_id is None

    # reflective, no transform:
    card = "*1 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform is None
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "reflective"
    assert test_s.periodic_id is None

    # reflective, transform:
    card = "*1 27 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform == transforms[27]
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "reflective"
    assert test_s.periodic_id is None

    # white, no transform:
    card = "+1 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform is None
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "white"
    assert test_s.periodic_id is None

    # white, transform:
    card = "+1 27 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform == transforms[27]
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "white"
    assert test_s.periodic_id is None

    # periodic, no transform:
    card = "1 -7 px 1.0"
    test_s = Surface.from_card(card, transforms)
    assert test_s.id == 1
    assert test_s.coord_transform is None
    assert test_s.type == "px"
    assert test_s.params == "1.0"
    assert test_s.boundary_type == "periodic"
    assert test_s.periodic_id is 7
