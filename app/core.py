"""
core.py -- prediction logic for the Electrode Voltage Predictor (no user-interface code).

The model is an XGBoost regressor trained on 2,634 DFT insertion electrodes from the
Materials Project. It predicts the host potential  E_host = V + E0(ion)  and the voltage is
recovered as  V = E_host - E0(ion),  which lets the model transfer to ions it was not trained on.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from xgboost import XGBRegressor

from featurize import E0, ION_CHARGE, TRAINED_IONS, discharged_formula, featurize

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "app_metadata.json")))
MODEL = XGBRegressor()
MODEL.load_model(os.path.join(HERE, META["model_file"]))
COLUMNS = META["feature_columns"]
FARADAY_MAH = 26801.48                       # mAh per mole of electrons

ZERO_SHOT_IONS = ["Rb", "Cs"]
EXPERIMENTAL_IONS = ["Al"]
ALL_IONS = TRAINED_IONS + ZERO_SHOT_IONS + EXPERIMENTAL_IONS


class InputError(ValueError):
    """A problem with what the user typed; the message is shown to the user."""


# ---------------------------------------------------------------------------
# Reference data (Materials Project voltages, with capacities computed here)
# ---------------------------------------------------------------------------
def _load_reference() -> pd.DataFrame | None:
    path = os.path.join(HERE, "reference_voltages.csv")
    if not os.path.exists(path):
        return None
    ref = pd.read_csv(path)
    caps, xs, elems = [], [], []
    for fw, ion, dis in zip(ref["framework_formula"], ref["working_ion"], ref["formula_discharge"]):
        try:
            d = Composition(dis)
            n_ion = d[ion]
            n_fu = Composition(fw).num_atoms
            # ions per formula unit of the framework as written
            host_atoms = d.num_atoms - n_ion
            x = n_ion * n_fu / host_atoms if host_atoms else np.nan
            caps.append(float(n_ion * ION_CHARGE.get(ion, 1) * FARADAY_MAH / float(d.weight)))
            xs.append(float(x))
            elems.append(" ".join(sorted(str(e) for e in Composition(fw).elements)))
        except Exception:
            caps.append(np.nan); xs.append(np.nan); elems.append("")
    ref["capacity_mAh_g"] = caps
    ref["x_per_fu"] = xs
    ref["energy_Wh_kg"] = ref["average_voltage"] * ref["capacity_mAh_g"]
    ref["elements"] = elems
    return ref


REF = _load_reference()


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def parse_host(host: str, ion: str | None = None) -> Composition:
    host = (host or "").strip()
    if not host:
        raise InputError("Please enter a host formula, for example FePO4 or MnO2.")
    try:
        comp = Composition(host)
        if comp.num_atoms == 0 or any(not isinstance(e, Element) for e in comp.elements):
            raise ValueError
    except Exception:
        raise InputError(f"Could not read '{host}' as a chemical formula. "
                         "Use element symbols, e.g. FePO4, Na2FeP2O7, Fe2(SO4)3.")
    if ion and ion in [str(e) for e in comp.elements]:
        raise InputError(f"The host already contains {ion}. Enter the charged (ion-free) host, "
                         f"e.g. FePO4 instead of LiFePO4.")
    return comp


def ion_from_choice(choice: str) -> str:
    return (choice or "Li").split()[0]


def reliability(ion: str) -> tuple[str, float | None]:
    """Return a short label and a typical error (V) for this ion."""
    if ion in TRAINED_IONS:
        return "trained", META["mae_in_distribution"]
    if ion in ZERO_SHOT_IONS:
        return "zero-shot", META["mae_zero_shot_known_hosts"]
    return "experimental", None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    host: str
    ion: str
    x: float
    voltage: float
    raw_voltage: float
    capacity: float
    energy: float
    discharged: str
    ox_charged: float
    ox_discharged: float
    label: str
    typical_error: float | None
    features: dict = field(repr=False, default_factory=dict)
    notes: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "Host": self.host, "Ion": self.ion, "x": self.x,
            "Voltage (V)": round(self.voltage, 2),
            "Capacity (mAh/g)": round(self.capacity, 0),
            "Energy (Wh/kg)": round(self.energy, 0),
            "Discharged": self.discharged,
            "Reliability": self.label,
        }


def _pretty_discharged(host, ion, x, dis):
    if float(x).is_integer():
        return dis.reduced_formula
    return f"{ion}{x:g}{Composition(host).reduced_formula}"


def predict_many(hosts_ions_x: list[tuple[str, str, float]]) -> list[Prediction]:
    """Vectorised prediction for several (host, ion, x) triples; inputs must be valid."""
    rows, feats = [], []
    for host, ion, x in hosts_ions_x:
        r, f = featurize(host, ion, float(x), COLUMNS)
        rows.append(r); feats.append(f)
    e_host = MODEL.predict(np.array(rows, dtype=float))
    out = []
    for (host, ion, x), e, f in zip(hosts_ions_x, e_host, feats):
        raw = float(e) - float(E0[ion])
        v = max(raw, 0.0)
        dis = Composition(discharged_formula(host, ion, float(x)))
        cap = float(float(x) * ION_CHARGE[ion] * FARADAY_MAH / float(dis.weight))
        label, err = reliability(ion)
        notes = []
        if f.get("no_redox_metal", 0) == 1 or f.get("elemental_host", 0) == 1:
            notes.append("No transition metal for redox (anode-type host such as graphite, Si or Sb); "
                         "few such hosts were in the training data, so expect larger errors.")
        if f.get("ox_unreliable", 0) == 1:
            notes.append("Oxidation states could not be balanced sensibly; the prediction is less reliable.")
        if f.get("ox_above_max", 0) == 1:
            notes.append("Info: the charged host contains a highly oxidised metal (e.g. Co⁴⁺, Ni⁴⁺), "
                         "as in most high-voltage cathodes.")
        if raw < 0:
            notes.append("The raw prediction was below 0 V and has been set to 0 V.")
        if label == "zero-shot":
            notes.append(f"{ion} was never in the training data; the prediction relies on the E° reference shift.")
        if label == "experimental":
            notes.append("Al³⁺ is trivalent and no trivalent ion was in the training data; treat as a rough guide.")
        out.append(Prediction(
            host=Composition(host).reduced_formula, ion=ion, x=float(x), voltage=v, raw_voltage=raw,
            capacity=cap, energy=v * cap, discharged=_pretty_discharged(host, ion, float(x), dis),
            ox_charged=float(f.get("ox_charged", np.nan)), ox_discharged=float(f.get("ox_discharged", np.nan)),
            label=label, typical_error=err, features=f, notes=notes))
    return out


def predict_one(host: str, ion: str, x: float) -> Prediction:
    parse_host(host, ion)
    if x is None or float(x) <= 0:
        raise InputError("The number of inserted ions per formula unit must be positive.")
    try:
        return predict_many([(host.strip(), ion, float(x))])[0]
    except InputError:
        raise
    except Exception as err:
        raise InputError(f"Features could not be computed for '{host}' ({type(err).__name__}). "
                         "Please check the formula.")


def reference_matches(host: str, ion: str | None = None) -> pd.DataFrame:
    if REF is None:
        return pd.DataFrame()
    try:
        red = Composition(host).reduced_formula
    except Exception:
        return pd.DataFrame()
    m = REF[REF["framework_reduced"] == red]
    if ion:
        m = m[m["working_ion"] == ion]
    return m


def parse_batch(text: str) -> list[tuple[str, float | None]]:
    """Read pasted formulas, optionally each followed by its own x.

    Formulas may be separated by new lines, commas, semicolons or spaces. A number written after a
    formula is that formula's x, e.g. "FePO4, Fe2(SO4)3 2, V2O5 1.5". Lines starting with # are ignored.
    Returns a list of (formula, x) pairs; x is None when not given.
    """
    items: list[tuple[str, float | None]] = []
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0]
        for tok in re.split(r"[,;\s]+", line.strip()):
            if not tok:
                continue
            try:
                val = float(tok)
            except ValueError:
                items.append((tok, None))
                continue
            if items and items[-1][1] is None and val > 0:
                items[-1] = (items[-1][0], val)
    return items
