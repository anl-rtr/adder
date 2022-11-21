import adder.mcnp
import pytest
from tests import xsdir
import numpy as np
import os

@pytest.fixture(scope='function')
def cleanup():
    # This fixture cleans up after each of the test functions in this file
    # use a yield None so nothing happens on normal calls, upon initialization
    # but on cleanup the content after yield is called
    yield None
    for file in ["xsdir", "test.inp"]:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass



# Create the blocks we will use for evaluating MCNP's handling of cards
# The cell block will include, by default, a test of line and in-line comments
cell_block = """Title
C upper-case comment
c lower-case comment
1 1 1.0 -2 imp:n=1 imp:p 2
2 2 -1.0 2 -3 IMP:N=1 ImP:P=2.5
3 0 3 imp:n,p=0 $ inline comment
"""

surf_block = """
2 cz 2.
*3 cx 3.
"""

# The material block will include a default material with a default nlib
# A material with an explicitly-specified isotope library, and one that relies
# on the m0 default
# and finally a material with an explicitly-specified isotope library, and one
# that relies on the material's nlib default
# This will be done with atom fractions and weight fractions inputs.
mat_block = """
m0 nlib=70c
m1 92235.71c 1.0 8016 2.0
m2 92238.72c -0.88 8016 -.12 nlib=71c
"""

# The data block will not include specific cards to actually test handling of,
# but will test: multi-line data by use of a line return and 5 blanks and an
# ampersand, the addition of a line break, and also handle Jumps and Repeats
# The jumps shall be ignored as they are not part of a data card for a cell.
# We also need to include multiplication and interpolation to verify they are
# ignored
# Finally, data cards that start with m but are not materials are included so
# that we can make sure adder doesn't screw up our material definitions
data_block = """kcode &
100 1.0 10
ksrc
     1. 1. 1.
dumy but_long 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
jump 5J 2j
rpt 1.0 20R
lin 1.5 2I 3.0
log 0.01 2LOG 10
ilog 0.01 2ILOG 10
tmul 1 1 2M 2M
c m commands that are not materials
mesh 1
mgopt 1
mode 1
mphys 1
mx 1
mp 1
c tally commands
f1:p 1
c1 0.5 1.0
cf1 2
cm1 1.
sf1 1
f4:n 1
fm4 1
fq4 1
ft4 1
fs4 1
sd4 1
fu4 1
pert4:n 1
fc4 comment
e4 0.5 20.0
Em4 1
t4 1
tf4 1
tm4 1
F14X:n 1
f14y:n 1
f14z:n 1
kpert14 1
ksen14 1
fmesh24:n data=1.0
fic34:n 1
fip34:n 1
fir34:n 1
de44 1
df44 1
c tmesh block
tmesh
    cmesh1:n 1
    cora1 1
    corb1 1
    corc1 1
    ergsh1 1
    mshmf1 1
    rmesh1:n 1
    smesh1:n
    c and a fake since everything in tmesh gets passed
    gobble
endmd
c output commands
mplot 1
print 10
prdmp 1 1 1 1 1
ptrac file=asc
histp -lhist=-1 1
dbcn 1
"""


def test_mcnp_readwrite(simple_lib):
    # Tests the ability to get data from an mcnp input file and then
    # regurgitate that file

    # This test will make sure we can process the basic input which tests:
    # comment styles, repeats, jumps, and the default material library info

    # We will do this test by using McnpNeutronics class' read_input method
    # This will store the input data and return materials information.
    # we will check the materials info and then write a new mcnp input file
    # and make sure we got it right.
    depl_libs = {0: simple_lib}

    # We will need to initialize an McnpNeutronics class to do this
    # The only input that matters here is "test.inp", we will have to write
    # our file here
    fname = "test.inp"
    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE",
                                          fname, 1, 1, True, 1.e-3, True)

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": [],
                          "apply_reactivity_threshold_to_initial_inventory":
                          False}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the input file
    start_inp = "".join([cell_block, surf_block, mat_block, data_block, ""])
    with open(fname, "w+") as f:
        f.write(start_inp)
    # And the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # Now we can get the data
    test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs, depl_libs)

    # Now lets check test_mats and the assigned parameters of test_neut
    assert test_mats[0].name == user_mats_info[1]["name"]
    assert test_mats[0].id == 1
    assert test_mats[0].is_depleting == user_mats_info[1]["depleting"]
    assert test_mats[0].isotopes[0].name == "U235"
    assert test_mats[0].isotopes[0].is_depleting is True
    assert test_mats[0].isotopes[0].xs_library == "71c"
    assert test_mats[0].isotopes[1].name == "O16"
    assert test_mats[0].isotopes[1].is_depleting is True
    assert test_mats[0].isotopes[1].xs_library == "70c"
    assert len(test_mats[0].isotopes_to_keep_in_model) == 2
    assert sorted(test_mats[0].isotopes_to_keep_in_model) == \
        sorted({'U235', 'O16'})
    assert test_mats[1].isotopes[0].name == "U238"
    assert test_mats[1].isotopes[0].is_depleting is True
    assert test_mats[1].isotopes[0].xs_library == "72c"
    assert test_mats[1].isotopes[1].name == "O16"
    assert test_mats[1].isotopes[1].is_depleting is False
    assert test_mats[1].isotopes[1].xs_library == "71c"
    assert len(test_mats[1].isotopes_to_keep_in_model) == 0
    np.testing.assert_allclose(test_mats[0].atom_fractions,
                               [1. / 3., 2. / 3.], rtol=1e-15)
    m_238 = 238.050786996
    m_16 = 15.9949146196
    frac_238 = 0.88
    frac_16 = 0.12
    ref_fracs = np.array([frac_238 / m_238, frac_16 / m_16])
    ref_fracs /= np.sum(ref_fracs)
    np.testing.assert_allclose(test_mats[1].atom_fractions,
                               ref_fracs, rtol=1e-15)
    assert test_mats[0].density == 1.
    ref_num_density = 1. * 0.6022140857 / np.dot([m_238, m_16], ref_fracs)
    assert test_mats[1].density == ref_num_density

    # Now check the data of the neutronics class
    assert test_neut.xsdir_file == lib_file
    ref_xsdir_data = \
        {"92238.70c": 236.005800, "92238.71c": 236.005800,
         "92238.72c": 236.005800, "92235.70c": 233.024800,
         "92235.71c": 233.024800, "92235.72c": 233.024800,
         "54135.70c": 133.748000, "54135.71c": 133.748000,
         "54135.72c": 133.748000, "53135.70c": 133.750000,
         "53135.71c": 133.750000, "53135.72c": 133.750000,
         "8016.70c": 15.857510, "8016.71c": 15.857510, "8016.72c": 15.857510,
         "1001.70c":0.999167, "1001.71c": 0.999167, "1001.72c": 0.999167}
    # Check the dictionary
    assert len(test_neut.neutronics_isotopes) == len(ref_xsdir_data)
    assert sorted(test_neut.neutronics_isotopes.keys()) == \
        sorted(ref_xsdir_data.keys())
    for k, v in test_neut.neutronics_isotopes.items():
        assert v == ref_xsdir_data[k]
    assert test_neut.use_depletion_library_xs is True
    assert len(test_neut.universes) == 1
    assert 0 in test_neut.universes
    assert test_neut.universes[0].name == "0"
    assert test_neut.universes[0].id == 0
    assert test_neut.universes[0].num_copies == 0

    # Check the cells
    ref_cell_data = \
        {"id": [1, 2, 3], "material_id": [1, 2, 0],
         "material": [test_mats[0], test_mats[1], None],
         "density": [1., ref_num_density, 0.], "surfaces": ["-2", "2 -3", "3"],
         "coord_transform": [None, None, None], "universe_id": [0, 0, 0],
         "lattice": [None, None, None], "fill_type": [None, None, None],
         "fill_dims": [None, None, None],
         "fill_transforms": [None, None, None], "_fill": [None, None, None],
         "fill_ids": [None, None, None], "volume": [None, None, None],
         "other_kwargs": [{"imp:n": "1", "imp:p": "2"},
                          {"imp:n": "1", "imp:p": "2.5"},
                          {"imp:n,p": "0"}]}
    for c in range(3):
        cell = test_neut.cells[ref_cell_data["id"][c]]
        for attrib in ref_cell_data.keys():
            # Density needs a floating point compare, the rest are direct:
            val = getattr(cell, attrib)
            if isinstance(val, float):
                if ref_cell_data[attrib][c] != 0.:
                    error = np.abs(val - ref_cell_data[attrib][c]) / \
                        ref_cell_data[attrib][c]
                else:
                    error = np.abs(val - ref_cell_data[attrib][c])
                assert error < 1e-15
            elif isinstance(val, str):
                assert val.strip() == ref_cell_data[attrib][c].strip()
            else:
                assert val == ref_cell_data[attrib][c]

    # We already checked the specific cells, lets just make sure the right
    # ones were assigned.
    assert list(test_neut.universes[0].cells.keys()) == [1, 2, 3]
    assert [c.id for c in test_neut.universes[0].cells.values()] == [1, 2, 3]

    assert test_neut.max_user_tally_id == 44
    assert test_neut.multiplier_mat_ids == set()
    assert len(test_neut.coord_transforms) == 1
    # Should only have the default null transform
    assert test_neut.coord_transforms[0].is_null is True

    # Now check the SimSettings object
    assert test_neut.sim_settings.particles == 100
    assert test_neut.sim_settings.keff_guess == 1.
    assert test_neut.sim_settings.inactive == 10
    assert test_neut.sim_settings.batches == 110
    assert test_neut.sim_settings.src_storage == 4500
    assert test_neut.sim_settings.normalize_by_weight == True
    assert test_neut.sim_settings.max_output_batches == 6500
    assert test_neut.sim_settings.max_avg_batches == True
    assert test_neut.sim_settings.additional_cards == ["ksrc 1. 1. 1."]

    # Finally we test the revised version of the input in the base_input
    # attribute.
    # This will be a dictionary with keys for message, title, surface, material
    # tally, output, and other

    # From out input we can tell what each needs to be, and that will form
    # the reference
    # We expect the only improvement to be the removal of line breaks and
    # replacing repeats
    ref_message = ["message: xsdir=xsdir"]
    ref_title = "Title"
    ref_surface = ["2 cz 2.", "*3 cx 3."]
    ref_material = \
        ["m0 nlib=70c", "m1 92235.71c 1.0 8016 2.0",
         "m2 92238.72c -0.88 8016 -.12 nlib=71c"]
    ref_tally = \
        ["f1:p 1", "c1 0.5 1.0", "cf1 2", "cm1 1.", "sf1 1", "f4:n 1", "fm4 1",
         "fq4 1", "ft4 1", "fs4 1", "sd4 1", "fu4 1", "pert4:n 1",
         "fc4 comment", "e4 0.5 20.0", "Em4 1", "t4 1", "tf4 1", "tm4 1",
         "F14X:n 1", "f14y:n 1", "f14z:n 1", "kpert14 1", "ksen14 1",
         "fmesh24:n data=1.0", "fic34:n 1", "fip34:n 1", "fir34:n 1",
         "de44 1", "df44 1", "tmesh", "cmesh1:n 1", "cora1 1", "corb1 1",
         "corc1 1", "ergsh1 1", "mshmf1 1", "rmesh1:n 1", "smesh1:n",
         "gobble", "endmd"]
    ref_output = ["mplot 1", "print 10", "prdmp 1 1 1 1 1", "ptrac file=asc",
                  "histp -lhist=-1 1", "dbcn 1"]
    ref_other = \
        ["dumy but_long 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 "
            "1.0000 1.0000 1.0000 1.0000",
         "jump 5J 2j",
         "rpt 1.0" + 20 * " 1.0",
         "lin 1.5 2I 3.0",
         "log 0.01 2LOG 10",
         "ilog 0.01 2ILOG 10",
         "tmul 1 1 2M 2M",
         "mesh 1",
         "mgopt 1",
         "mode 1",
         "mphys 1",
         "mx 1",
         "mp 1"]

    for key, ref in zip(["message", "title", "material", "tally",
                         "output", "other"],
                        [ref_message, ref_title, ref_material,
                         ref_tally, ref_output, ref_other]):
        assert test_neut.base_input[key] == ref
    # Now check the ref surfaces
    for i, surf in enumerate([test_neut.surfaces[2], test_neut.surfaces[3]]):
        assert str(surf) == ref_surface[i]

    # Now we can write the output file for comparison
    # This one will include the user's tallies and output
    out_name = "output.inp"
    test_neut.write_input(out_name, "test", test_mats, depl_libs, False, False,
                          True, True)
    with open(out_name, "r") as f:
        test_out = f.readlines()
    os.remove(out_name)

    # Now we can check output
    # Create the reference solutions
    ref_cell_block = ["1 1 1.0000000000000E+00 -2 u=0 imp:n=1 imp:p=2",
                      "2 2 6.7442405203693E-03 2 -3 u=0 imp:n=1 imp:p=2.5",
                      "3 0 3 u=0 imp:n,p=0"]

    ref_mat_block = \
        ["c ADDER Material Name: mat1",
         "m1 92235.71c 3.333333E-01 8016.70c 6.666667E-01",
         "c ADDER Material Name: 2",
         "m2 92238.72c {:1.6E} 8016.71c {:1.6E}".format(ref_fracs[0],
                                                        ref_fracs[1])]
    ref_out_file = ref_message + [""] + [ref_title + " test"] +  \
        ref_cell_block + [""] + ref_surface + [""] + ref_mat_block + \
        ["PRINT 10 60 128 130", "PRDMP 1 1 1 1 1", "mplot 1", "ptrac file=asc",
         "histp -lhist=-1 1", "dbcn 1"] + \
        ["kcode 100 1.00000 10 110 4500 0 6500 1", "ksrc 1. 1. 1."] + \
        ["dumy but_long 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000",
         "     1.0000 1.0000",
         "jump 5J 2j",
         "rpt 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0",
         "     1.0 1.0",
         "lin 1.5 2I 3.0",
         "log 0.01 2LOG 10",
         "ilog 0.01 2ILOG 10",
         "tmul 1 1 2M 2M",
         "mesh 1",
         "mgopt 1",
         "mode 1",
         "mphys 1",
         "mx 1",
         "mp 1"] + ref_tally + [""]
    ref_out = [line + "\n" for line in ref_out_file]
    assert len(test_out) == len(ref_out)
    for i in range(len(test_out)):
        assert test_out[i] == ref_out[i]

    # Now we need to verify that it does not write each of user output
    # and user tallies. We will do this one at a time.
    # Without print output, we lose the user's print table 10 command and the
    # mplot command
    ref_out[14] = "PRINT 60 128 130\n"
    ref_out[15] = "PRDMP 0 0 1\n"
    del ref_out[16:20]
    test_neut.write_input(out_name, "test", test_mats, depl_libs, False, False,
                          True, False)
    with open(out_name, "r") as f:
        test_out = f.readlines()
    os.remove(out_name)
    assert len(test_out) == len(ref_out)
    for i in range(len(test_out)):
        assert test_out[i] == ref_out[i]

    # And now we will lose the user's tallies
    del ref_out[33: -1]
    test_neut.write_input(out_name, "test", test_mats, depl_libs, False, False,
                          False, False)
    with open(out_name, "r") as f:
        test_out = f.readlines()
    os.remove(out_name)
    assert len(test_out) == len(ref_out)
    for i in range(len(test_out)):
        assert test_out[i] == ref_out[i]

    # The remainder of write_input is thoroughly tested elsewhere in the
    # integration suite

    # Cleanup
    for file in ["xsdir", "test.inp"]:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def test_mcnp_input_errors(simple_lib):
    # Tests for failures in the MCNP input files
    depl_libs = {0: simple_lib}
    # These are specifically those in the following list
    to_check = ["burn", "embeb", "embed", "embee", "embem", "embtb", "embtm",
                "notrn", "talnp", "read", "continue", "#"]

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # all of these are data cards
    for card in to_check:
        # Write the input file
        start_inp = "".join([cell_block, surf_block, mat_block, data_block,
                             card + " 1", ""])
        with open(fname, "w+") as f:
            f.write(start_inp)

        test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                              True, 1.e-3, False)
        # The next command should raise a ValueError due to the presence of the
        # error card.
        with pytest.raises(ValueError):
            test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                             user_mats_info, user_univ_info,
                                             shuffled_mats, shuffled_univs,
                                             depl_libs)


def test_mcnp_pass_input(simple_lib):
    # Tests that remaining input cards are passed along untouched
    depl_libs = {0: simple_lib}
    # These are those not included elsewhere
    cards = ["act", "area", "bbrem", "bdlf#", "cosyp", "ctme", "cut:",
             "dawwg", "dd#", "dm", "drxs", "ds#", "dxt:", "esplt:", "field",
             "files", "fmult", "hsrc", "idum", "kopts", "lca", "lcb", "lcc",
             "lea", "leb", "lost", "nps", "otfdb", "phys:", "pikmt", "rand",
             "rdum", "sb#", "sc#", "sdef", "si#", "spabi:n", "spdtl", "sp#",
             "ssr", "ssw", "stop", "thtme", "totnu", "tropt", "tsplt:",
             "var", "vect", "void", "wwe:", "wwg", "wwge:", "wwgt:", "wwp:",
             "wwt:", "za", "zb", "zc", "zd"]
    particles = ["n", "p", "e", "|", "q", "u", "v", "f", "h", "l", "+", "-",
                 "x", "y", "o", "1", "<", ">", "g", "/", "z", "k", "%", "^",
                 "b", "_", "~", "c", "w", "@", "d", "t", "s", "a", "*", "?",
                 "#"]

    to_check = []
    for card in cards:
        if card.endswith("#"):
            # Then we should add variants of this with numbers, just enough to
            # do double digits
            to_check.extend([card + str(i) for i in range(1, 11)])
        elif card.endswith(":"):
            # Then we add the particle designators
            to_check.extend([card + p for p in particles])
        else:
            to_check.append(card)

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # all of these are data cards
    out_name = "output.inp"
    for card in to_check:
        # Write the input file
        start_inp = "".join([cell_block, surf_block, mat_block, data_block,
                             card + " 1\n", ""])
        with open(fname, "w+") as f:
            f.write(start_inp)
        test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                              True, 1.e-3, False)
        test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                         user_mats_info, user_univ_info,
                                         shuffled_mats, shuffled_univs,
                                         depl_libs)
        test_neut.write_input(out_name, "test", test_mats, depl_libs, False,
                              False, False, False)
        with open(out_name, "r") as f:
            test_out = f.readlines()
        os.remove(out_name)

        # Now we have the output file, lets just make sure our card has made
        # it in there and there is only one of them
        assert test_out.count(card + " 1\n") == 1


def test_mcnp_card_to_cell(simple_lib):
    # Tests that cell data block cards are correctly passed
    depl_libs = {0: simple_lib}
    cards = ["vol", "pwt", "u", "nonu", "cosy", "bflcl", "imp:", "ext:",
             "fcl:", "elpt:", "unc:", "dxc", "pd", "tmp", "wwn"]
    # Note the above doesnt include *fill/fill/lat as these require a more
    # complicated model. The code for these is the same as for the above,
    # however, so there is no loss in accuracy.
    particles = ["n", "p", "e", "|", "q", "u", "v", "f", "h", "l", "+", "-",
                 "x", "y", "o", "!", "<", ">", "g", "/", "z", "k", "%", "^",
                 "b", "_", "~", "c", "w", "@", "d", "t", "s", "a", "*", "?",
                 "#"]

    # We cant use the test-level cell block bc we need to have multiple
    # universes
    test_cell_block = """Title
1 1 1.0 -2
2 2 -1.0 2 -3
3 0 3
"""
    to_check = []
    for card in cards:
        if card.endswith(":"):
            # Then we add the particle designators
            to_check.extend([card + p for p in particles])
        else:
            to_check.append(card)

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # all of these are data cards
    out_name = "output.inp"
    values = " 1 2j"
    for card in to_check:
        # Write the input file
        param = card + values + "\n"

        start_inp = "".join([test_cell_block, surf_block, mat_block,
                             data_block, param, ""])
        with open(fname, "w+") as f:
            f.write(start_inp)
        test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                              True, 1.e-3, False)
        test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                         user_mats_info, user_univ_info,
                                         shuffled_mats, shuffled_univs,
                                         depl_libs)
        test_neut.write_input(out_name, "test", test_mats, depl_libs, False,
                              False, False, False, False)
        with open(out_name, "r") as f:
            test_out = f.readlines()
        os.remove(out_name)

        # Now we have the output file, lets just make sure our card has not
        # made it in there as it should be moved to a parameter on the cell
        assert test_out.count(card + " 1\n") == 0

        # Now also make sure it is on the cell
        # Verify we have the right behavior on our cells
        if card == "u":
            # Then cell 1 wont be printed since it was assigned to universe 1,
            # but the others will be (still in univ 0)
            # For these, to verify correctness we need to just compare the
            # line since u=0 will be there no matter what
            assert test_out[3] == "2 2 6.7442405203693E-03 2 -3 u=0\n"
            assert test_out[4] == "3 0 3 u=0\n"
            assert test_out[5] == "\n"
        else:
            assert card + "=1" in test_out[3]
            assert card not in test_out[4]
            assert card not in test_out[5]


def test_mcnp_like_but_cell(simple_lib):
    # Tests that the like-but functionality is correctly applied for cells
    depl_libs = {0: simple_lib}

    test_cell_block = """Title
1 1 1.0 -2
2 liKe 1 but mat=2 rho=-1
3 0 3
"""

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # all of these are data cards
    out_name = "output.inp"
    start_inp = "".join([test_cell_block, surf_block, mat_block,
                         data_block, ""])
    with open(fname, "w+") as f:
        f.write(start_inp)
    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                          True, 1.e-3, False)
    test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs, depl_libs)
    test_neut.write_input(out_name, "test", test_mats, depl_libs, False, False,
                          False, False, False)
    with open(out_name, "r") as f:
        test_out = f.readlines()
    os.remove(out_name)

    # Now we need to just check our second cell to see if it is correct
    m_238 = 238.050786996
    m_16 = 15.9949146196
    frac_238 = 0.88
    frac_16 = 0.12
    ref_fracs = np.array([frac_238 / m_238, frac_16 / m_16])
    ref_fracs /= np.sum(ref_fracs)
    ref_num_density = 1. * 0.6022140857 / np.dot([m_238, m_16], ref_fracs)
    # Get the params of the cell card
    test_cell_params = test_out[4].split()
    # This should be " 2 2 0.006744240520369279 -2 u=0"
    assert test_cell_params[0] == "2"
    assert test_cell_params[1] == "2"
    assert np.abs(ref_num_density - float(test_cell_params[2])) < 1e-15
    assert test_cell_params[3] == "-2"
    assert test_cell_params[4] == "u=0"


def test_mcnp_xscards(simple_lib):
    # This tests the cross section data cards, XSn and AWTAB to make sure their
    # information is captured in the input
    depl_libs = {0: simple_lib}
    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # We will test adding a new isotope to xsdir with XSn, updating awr
    # with an XSn card, and then updating with awtab. The isotopes and values
    # will be chosen so we an verify awtab over-rules xsn's change
    new_data_block = data_block + \
        "XS1 5010.70c 10.0 5010.70c 0 1 1 16 0 0 2.5301E-8\n" + \
        "xs2 5010.71c 10.1 5010.71c 0 1 1 16 0 0 2.5301E-8\n" + \
        "xS3 92238.70c 237. 92238.70c  0 1 1 60  0 0 2.5301e-08\n" + \
        "Xs4 92235.70c 234. 92235.70c  0 1 1 60  0 0 2.5301e-08\n" + \
        "AWtAB 92235.70 236.0"

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # Write the input file
    start_inp = "".join([cell_block, surf_block, mat_block,
                         new_data_block, ""])
    with open(fname, "w+") as f:
        f.write(start_inp)

    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1, True,
                                          1.e-3, False)
    _ = test_neut.read_input(lib_file, num_neutron_groups, user_mats_info,
                             user_univ_info, shuffled_mats, shuffled_univs,
                             depl_libs)
    # Now check test_neut.allowed_isotopes for valid values based on
    # expectations
    # Start with xsdir-based info and perturb from there so reader of this
    # test sees the explicit differences
    ref_xsdir_data = \
        {"92238.70c": 236.005800, "92238.71c": 236.005800,
         "92238.72c": 236.005800, "92235.70c": 233.024800,
         "92235.71c": 233.024800, "92235.72c": 233.024800,
         "54135.70c": 133.748000, "54135.71c": 133.748000,
         "54135.72c": 133.748000, "53135.70c": 133.750000,
         "53135.71c": 133.750000, "53135.72c": 133.750000,
         "8016.70c": 15.857510, "8016.71c": 15.857510, "8016.72c": 15.857510,
         "1001.70c": 0.999167, "1001.71c": 0.999167, "1001.72c": 0.999167}
    # XS1 and XS2 cards add in 2 5010 values
    ref_xsdir_data["5010.70c"] = 10.0
    ref_xsdir_data["5010.71c"] = 10.1
    # XS3 modifies 92238.70c in the above
    ref_xsdir_data["92238.70c"] = 237.
    # XS4 will get overruled by AWTAB. AWTAB changes U235.70c
    ref_xsdir_data["92235.70c"] = 236.

    # Check the dictionary
    assert len(test_neut.neutronics_isotopes) == len(ref_xsdir_data)
    assert sorted(test_neut.neutronics_isotopes.keys()) == \
        sorted(ref_xsdir_data.keys())
    for k, v in test_neut.neutronics_isotopes.items():
        assert v == ref_xsdir_data[k]


def test_mcnp_material_parsing_error(caplog, simple_lib):
    root_logger = adder.init_root_logger("adder")
    # Tests for unique behavior related to isotopics of material cards
    depl_libs = {0: simple_lib}

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": True, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # Now create a material card that is depleting but has no default xslib
    # identified. In this case we expect the Material initialization to raise
    # an error that the neutronics class has to log for it. This error shows up
    # with a depleting material that has no default xslib. This is present in
    # the mats block for line 1 below. The error is expected for m1 but not m2
    mats = "\nm1 92235.71c 1.0 8000.70c 2.0\n" \
        "m2 92238.72c -0.88 8016 -.12 nlib=71c\n"
    ref_error = 'Material mat1 (id: 1) is depleting without a default ' \
        'cross section library defined!'

    # Write the input file
    start_inp = "".join(
        [cell_block, surf_block, mats, data_block, ""])
    with open(fname, "w+") as f:
        f.write(start_inp)

    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                          True, 1.e-3, False)
    # The next command should raise an error via the logger. In this case
    # we look for SystemExit and then read in the captured logger output
    # to compare with expectations
    caplog.clear()
    with pytest.raises(SystemExit):
        test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                         user_mats_info, user_univ_info,
                                         shuffled_mats, shuffled_univs,
                                         depl_libs)
    # Now make sure we have the right logged message to ensure the
    # failure was for the expected reason
    assert ref_error in caplog.text

    # Now test the conditions where we have
    # (1) invalid constituent with m0 card
    # (2) invalid constituent without m0 card
    # (3) duplicate isotope present in a material
    # We do this with a non-depleting material to get the error message we want
    user_mats_info = {1: {"name": "mat1", "depleting": False, "density": None,
                          "non_depleting_isotopes": []}}
    mat_cases = [
        "\nm0 nlib=70c\nm1 92235.71c 1.0 8000 2.0\n"
        "m2 92238.72c -0.88 8016 -.12 nlib=71c\n",
        "\nm1 92235.71c 1.0 8000.70c 2.0\n"
        "m2 92238.72c -0.88 8016 -.12 nlib=71c\n",
        "\nm1 92235.71c 1.0 8016.70c 2.0 8016.71c 2.0\n"
        "m2 92238.72c -0.88 8016 -.12 nlib=71c\n",
    ]
    ref_errors = [
        'Material mat1 (id: 1) contains 8000.70c '
        'which is not present in the xsdir file',
        'Material mat1 (id: 1) contains 8000.70c '
        'which is not present in the xsdir file',
        'Material id: 1 contains O16 multiple times. '
        'Only one entry is supported.']

    for mats, ref_error in zip(mat_cases, ref_errors):
        # Write the input file
        start_inp = "".join(
            [cell_block, surf_block, mats, data_block, ""])
        with open(fname, "w+") as f:
            f.write(start_inp)

        test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                              True, 1.e-3, False)
        # The next command should raise an error via the logger. In this case
        # we look for SystemExit and then read in the captured logger output
        # to compare with expectations
        caplog.clear()
        with pytest.raises(SystemExit):
            test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                             user_mats_info, user_univ_info,
                                             shuffled_mats, shuffled_univs,
                                             depl_libs)
        # Now make sure we have the right logged message to ensure the
        # failure was for the expected reason
        assert ref_error in caplog.text


def test_mcnp_material_is_depleting(simple_lib):
    # Tests for correct assignment of is_depleting status to materials and
    # isotopes
    depl_libs = {0: simple_lib}

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": True, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # Now make sure the is-depleting status is as expected given the presence
    # of isotopes in the depletion library.
    # Specifically, O16 is not in the depletion library (simple_lib) and so
    # it should be flagged as non-depleting. Since we have been bit before by
    # an error where this was incorrectly done when a non-depleting iso exists
    # in multiple depletion materials

    # Write the input file
    start_inp = "".join(
        [cell_block, surf_block, mat_block, data_block, ""])
    with open(fname, "w+") as f:
        f.write(start_inp)

    # Have the parserload up the info on this model
    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                          True, 1.e-3, False)
    test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs,
                                     depl_libs)
    # And now we can check the material objects
    # mat 1 and 2 will have following depleting status.
    # The iso names and xs lib values from inspection of mat_block and the
    # True/False is whether or not it is depleting (based on if present in
    # simplelib)
    ref_mat_iso_info = [[('U235', '71c', True), ('O16', '70c', False)],
                        [('U238', '72c', True), ('O16', '71c', False)]]
    for i, mat in enumerate(test_mats):
        for j, iso in enumerate(mat.isotopes):
            assert iso.name == ref_mat_iso_info[i][j][0]
            assert iso.xs_library == ref_mat_iso_info[i][j][1]
            assert iso.is_depleting == ref_mat_iso_info[i][j][2]

    # The above verified that McnpNeutronics correctly sets the isotopic
    # depleting status based on if an isotope is in the depleting library. This
    # next test checks to make sure McnpNeutronics correctly applies the user
    # option for isotopic depleting status
    # To do this, we simply tell the read_input method, via user_mats_info,
    # that U235 should be depleting. Then, we look for that status in the
    # same way we looked in the above. All other values should be the same as
    # in the above test.
    user_mats_info[1]["non_depleting_isotopes"] = ['U235']
    test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                          True, 1.e-3, False)
    test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                     user_mats_info, user_univ_info,
                                     shuffled_mats, shuffled_univs,
                                     depl_libs)
    ref_mat_iso_info[0][0] = ('U235', '71c', False)
    for i, mat in enumerate(test_mats):
        for j, iso in enumerate(mat.isotopes):
            assert iso.name == ref_mat_iso_info[i][j][0]
            assert iso.xs_library == ref_mat_iso_info[i][j][1]
            assert iso.is_depleting == ref_mat_iso_info[i][j][2]


def test_mcnp_material_zero_fractions(caplog, simple_lib):
    root_logger = adder.init_root_logger("adder")
    # Tests for correct handling of zero-fraction isotopes/elements
    # This is done by including test cases of various situations to ensure the
    # 0-fraction isotope is not included and the density is as expected.

    depl_libs = {0: simple_lib}

    fname = "test.inp"

    # Set the parameters to pass
    lib_file = "xsdir"
    num_neutron_groups = 1
    # Lets put some names on one of the materials to check how this works
    user_mats_info = {1: {"name": "mat1", "depleting": True, "density": None,
                          "non_depleting_isotopes": []}}
    user_univ_info = {}
    shuffled_mats = set()
    shuffled_univs = set()

    # Write the xsdir file
    with open(lib_file, "w") as f:
        f.write(xsdir)

    # Build a cell block that has the same density so we can compare the
    # interpreted material densities and make sure they are consistent
    cell_block_data = """Title
C upper-case comment
c lower-case comment
1 1 1.0 -2 imp:n=1 imp:p 2
2 2 1.0 -2 imp:n=1 imp:p 2
3 0 3 imp:n,p=0 $ inline comment
"""
    # Now build the material block test cases.
    mat_cases = [
        "\nm0 nlib=70c\nm1 92235.70c 1.0 8016.70c 2.0\n"
        "m2 92235.70c 1.0 8016.70c 2.0 92238.70c 0.0\n",
        "\nm0 nlib=70c\nm1 92235.70c -1.0 8016.70c -2.0\n"
        "m2 92235.70c -1.0 8016.70c -2.0 92238.70c -0.0\n",
        "\nm0 nlib=70c\nm1 92235.70c 1.0 8016.70c 2.0\n"
        "m2 92235.70c 1.0 8016.70c 2.0 92238.70c -0.0\n"
    ]
    ref_warning = 'U238 in Material 2 (id: 2) ' \
        'was removed as it has a zero atom fraction'

    for mats in mat_cases:
        # Write the input file
        start_inp = "".join(
            [cell_block_data, surf_block, mats, data_block, ""])
        with open(fname, "w+") as f:
            f.write(start_inp)

        test_neut = adder.mcnp.McnpNeutronics("", "mcnp.EXE", fname, 1, 1,
                                              True, 1.e-3, False)
        # The next command should create our materials and leave log msgs
        caplog.clear()

        # Now parse the input
        test_mats = test_neut.read_input(lib_file, num_neutron_groups,
                                         user_mats_info, user_univ_info,
                                         shuffled_mats, shuffled_univs,
                                         depl_libs)
        # Now make sure we have the expected logged warning message
        assert ref_warning in caplog.text

        # Check the densities.
        assert test_mats[0].mass_density == test_mats[1].mass_density
