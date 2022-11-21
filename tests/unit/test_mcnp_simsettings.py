from adder.mcnp.sim_settings import SimSettings
import pytest


def test_simsettings_init():
    # Test the following initialization perturbations
    # - For each parameter, do type-checking
    # - All defaults, check all values set as expected
    # - For each parameter verify successful setting of values
    # - For each parameter that has a dependent-default, make sure that
    #   dependency is obeyed

    # CHECK TYPE-CHECKING
    # particles
    with pytest.raises(TypeError):
        test_ss = SimSettings(particles=0.1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(particles=-1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(particles=0)
    # keff_guess
    with pytest.raises(TypeError):
        # Expects int or float, so make try something that isnt one of those
        test_ss = SimSettings(keff_guess='a')
    with pytest.raises(ValueError):
        test_ss = SimSettings(keff_guess=0.)
    with pytest.raises(ValueError):
        test_ss = SimSettings(keff_guess=-1.)
    with pytest.raises(ValueError):
        test_ss = SimSettings(keff_guess=0)
    with pytest.raises(ValueError):
        test_ss = SimSettings(keff_guess=-1)
    # inactive
    with pytest.raises(TypeError):
        test_ss = SimSettings(inactive=0.1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(inactive=-1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(inactive=0)
    # batches
    with pytest.raises(TypeError):
        test_ss = SimSettings(batches=0.1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(batches=-1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(batches=0)
    # src_storage
    with pytest.raises(TypeError):
        test_ss = SimSettings(src_storage=0.1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(src_storage=-1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(src_storage=0)
    # normalize_by_weight
    with pytest.raises(TypeError):
        test_ss = SimSettings(normalize_by_weight=100)
    # max_output_batches
    with pytest.raises(TypeError):
        test_ss = SimSettings(max_output_batches=0.1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(max_output_batches=-1)
    with pytest.raises(ValueError):
        test_ss = SimSettings(max_output_batches=0)
    # max_avg_batches
    with pytest.raises(TypeError):
        test_ss = SimSettings(max_avg_batches=100)
    # additional_cards
    with pytest.raises(TypeError):
        test_ss = SimSettings(additional_cards=0)
    with pytest.raises(TypeError):
        test_ss = SimSettings(additional_cards="test")
    with pytest.raises(TypeError):
        test_ss = SimSettings(additional_cards=[1, 2])

    # Now set the defaults and make sure values are as expected
    test_ss = SimSettings()
    assert test_ss.particles == 1000
    assert test_ss.keff_guess == 1.
    assert test_ss.inactive == 30
    assert test_ss.batches == 130
    assert test_ss.src_storage == 4500
    assert test_ss.normalize_by_weight == True
    assert test_ss.max_output_batches == 6500
    assert test_ss.max_avg_batches == True
    assert test_ss.additional_cards is None

    # Try not defaults and make sure it is correct
    test_ss = SimSettings(particles=3000, keff_guess=3, inactive=7,
        batches=90, src_storage=10, normalize_by_weight=False,
        max_output_batches=123, max_avg_batches=False,
        additional_cards=["test1", "test2"])
    assert test_ss.particles == 3000
    assert test_ss.keff_guess == 3.
    assert test_ss.inactive == 7
    assert test_ss.batches == 90
    assert test_ss.src_storage == 10
    assert test_ss.normalize_by_weight == False
    assert test_ss.max_output_batches == 123
    assert test_ss.max_avg_batches == False
    assert test_ss.additional_cards == ["test1", "test2"]

    # Repeat but with a keff_guess that is a float
    test_ss = SimSettings(particles=3000, keff_guess=3., inactive=7,
        batches=90, src_storage=10, normalize_by_weight=False,
        max_output_batches=123, max_avg_batches=False,
        additional_cards=["test"])
    assert test_ss.keff_guess == 3.

    # Now test batches as the default, but inactive at 7, should have batches
    # be 100 + 7
    test_ss = SimSettings(inactive=7)
    assert test_ss.batches == 107

    # Now test src_storage as the default, but change what is maximum (4500 or
    # 2 * particles) as that sets src_storage's value from the getter
    test_ss = SimSettings(particles=1)
    assert test_ss.src_storage == 4500
    test_ss = SimSettings(particles=2300)
    assert test_ss.src_storage == 4600


def test_simsettings_kcode_str():
    # We already tested getting the values, now we just need to verify we
    # print the right values
    # Since we already checked the getters/setters now we just need to test
    # the formatting of the string with default values
    ref_string = "kcode 1000 1.00000 30 130 4500 0 6500 1"
    test_ss = SimSettings()
    assert test_ss.kcode_str == ref_string

    # Test the remaining items to get full code coverage of SimSettings
    # We are missing a normalize_by_weight and max_output_batches case
    test = "kcode 10 1.00000 30 130 4500 0 6500 1"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.kcode_str == test
    test = "kcode 10 1.00000 30 130 4500 1 6500 0"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.kcode_str == test


def test_simsettings_all_cards():
    # Test ability to set and print all cards
    ref_strings = ["kcode 1000 1.00000 30 130 4500 0 6500 1", "test1 1.0",
        "test2 1.0"]
    test_ss = SimSettings(additional_cards=ref_strings[1:])
    assert test_ss.all_cards == ref_strings


def test_simsettings_from_cards():
    # Repeats test_simsettings_init but interfaced through the kcode card
    # Also tests that jumps lead to defaults and that values not present (at
    # end) become defaults (happens throughout)

    # First if the first card isnt kcode, it wont work
    with pytest.raises(ValueError):
        test_ss = SimSettings.from_cards(["kc0de 1 1 1"])

    # Now we can do type checking on our values. Here we get ValueErrors if
    # a parameter isn't the correct type, we will jump through and use 'a'
    # to be the bad type
    tests = ["a", "j a", "j j a", "j j j a", "j j j j a", "j j j j j a",
             "j j j j j j a", "j j j j j j j a"]
    for test in tests:
        with pytest.raises(ValueError):
            test_ss = SimSettings.from_cards(["kcode " + test])

    # Now set with defaults (no jumps and w/ jumps, and half empty)
    tests = ["kcode ", "kcode j j j j j j j j", "kcode 1000 1. 30 j"]
    for test in tests:
        test_ss = SimSettings.from_cards([test])
        assert test_ss.particles == 1000
        assert test_ss.keff_guess == 1.
        assert test_ss.inactive == 30
        assert test_ss.batches == 130
        assert test_ss.src_storage == 4500
        assert test_ss.normalize_by_weight == True
        assert test_ss.max_output_batches == 6500
        assert test_ss.max_avg_batches == True
        assert test_ss.additional_cards is None

    # Try not defaults and make sure it is correct (do with a keff guess that
    # is both an int and a float)
    tests = ["kcode 3000 3 7 90 10 1 123 0", "kcode 3000 3. 7 90 10 1 123 0"]
    for test in tests:
        test_ss = SimSettings.from_cards([test])
        assert test_ss.particles == 3000
        assert test_ss.keff_guess == 3.
        assert test_ss.inactive == 7
        assert test_ss.batches == 90
        assert test_ss.src_storage == 10
        assert test_ss.normalize_by_weight == False
        assert test_ss.max_output_batches == 123
        assert test_ss.max_avg_batches == False
        assert test_ss.additional_cards is None

    # Now test batches as the default, but inactive at 7, should have batches
    # be 100 + 7
    test = "kcode j j 7"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.batches == 107

    # Now test src_storage as the default, but change what is maximum (4500 or
    # 2 * particles) as that sets src_storage's value from the getter
    test = "kcode 1"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.src_storage == 4500
    test = "kcode 2300"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.src_storage == 4600

    # Test the remaining items to get full code coverage of SimSettings
    # We are missing a normalize_by_weight and max_avg_batches case
    test = "kcode j j j j j 0 j 1"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.normalize_by_weight == True
    assert test_ss.max_avg_batches == True
    test = "kcode j j j j j 1 j 0"
    test_ss = SimSettings.from_cards([test])
    assert test_ss.normalize_by_weight == False
    assert test_ss.max_avg_batches == False

    # Now make sure additional_cards gets passed
    add_cards = ["test1 1.0", "test2 1.0"]
    test_ss = SimSettings.from_cards([test] + add_cards)
    assert test_ss.additional_cards == add_cards
