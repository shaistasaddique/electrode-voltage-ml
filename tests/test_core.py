"""Tests for the prediction core. Run from the repository root with:  pytest -q"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import core  # noqa: E402


def test_lifepo4_is_close_to_dft():
    p = core.predict_one("FePO4", "Li", 1)
    assert 3.0 < p.voltage < 3.9              # DFT 3.50 V, experiment about 3.45 V
    assert abs(p.capacity - 170) < 2          # theoretical capacity of LiFePO4
    assert p.discharged == "LiFePO4"
    assert p.label == "trained"


def test_oxidation_states_fe3_to_fe2():
    p = core.predict_one("FePO4", "Li", 1)
    assert p.ox_charged == pytest.approx(3.0)
    assert p.ox_discharged == pytest.approx(2.0)


def test_zinc_is_lower_than_lithium():
    # Zn/Zn2+ sits about 2.3 V above Li/Li+, so the same host gives a lower voltage vs Zn
    li = core.predict_one("MnO2", "Li", 1).voltage
    zn = core.predict_one("MnO2", "Zn", 0.5).voltage
    assert zn < li - 1.0


@pytest.mark.parametrize("ion", core.ALL_IONS)
def test_every_ion_runs(ion):
    p = core.predict_one("V2O5", ion, 1)
    assert 0 <= p.voltage < 7


@pytest.mark.parametrize("bad, ion", [("XyZ2", "Li"), ("", "Li"), ("LiFePO4", "Li"), ("Fe((PO4", "Na")])
def test_bad_input_gives_friendly_error(bad, ion):
    with pytest.raises(core.InputError):
        core.predict_one(bad, ion, 1)


def test_non_positive_x_rejected():
    with pytest.raises(core.InputError):
        core.predict_one("FePO4", "Li", 0)


def test_batch_parsing():
    assert core.parse_batch("FePO4\nMnO2, V2O5; # comment\n\nTiS2") == [
        ("FePO4", None), ("MnO2", None), ("V2O5", None), ("TiS2", None)]


def test_batch_parsing_with_x():
    assert core.parse_batch("Fe2(SO4)3 2\nV2O5, 1.5\nFePO4") == [
        ("Fe2(SO4)3", 2.0), ("V2O5", 1.5), ("FePO4", None)]


def test_reference_data_loaded():
    assert core.REF is not None and len(core.REF) > 2000
    assert len(core.reference_matches("FePO4", "Li")) >= 1
