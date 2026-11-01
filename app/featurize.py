"""
featurize.py — turns a host formula + working ion into the exact features used to train the model.
The code is copied from the Week 2 (composition) and Week 2b (oxidation-state) notebooks so that
the app computes features identically to training.
"""
import numpy as np
from pymatgen.core import Composition, Element
PROPS = {
    "X": lambda e: e.X,
    "radius": lambda e: float(e.atomic_radius) if e.atomic_radius is not None else np.nan,
    "Z": lambda e: e.Z,
    "mass": lambda e: float(e.atomic_mass),
    "row": lambda e: e.row,
    "group": lambda e: e.group,
    "mendeleev": lambda e: e.mendeleev_no,
    "ionization": lambda e: e.ionization_energy,
}
_cache = {}
def eprop(sym, name):
    key = (sym, name)
    if key not in _cache:
        try:
            v = PROPS[name](Element(sym)); _cache[key] = np.nan if v is None else float(v)
        except Exception:
            _cache[key] = np.nan
    return _cache[key]

TM = {str(e) for e in Element if e.is_transition_metal}
ANIONS = ["O", "F", "S", "Se", "N", "Cl", "Br"]
POLY = ["P", "Si", "B", "C", "Mo", "W"]

def host_features(formula):
    comp = Composition(formula).fractional_composition
    syms = [str(e) for e in comp.elements]; w = np.array([comp[e] for e in comp.elements])
    f = {}
    for name in PROPS:
        v = np.array([eprop(s, name) for s in syms])
        m = ~np.isnan(v)
        if m.any():
            ww = w[m] / w[m].sum(); vv = v[m]
            mean = (ww * vv).sum()
            f[f"host_{name}_mean"] = mean
            f[f"host_{name}_std"] = np.sqrt((ww * (vv - mean) ** 2).sum())
            f[f"host_{name}_min"], f[f"host_{name}_max"] = vv.min(), vv.max()
        else:
            for s in ("mean", "std", "min", "max"): f[f"host_{name}_{s}"] = np.nan
    f["host_n_elements"] = len(syms)
    f["host_frac_TM"] = sum(comp[e] for e in comp.elements if str(e) in TM)
    for a in ANIONS: f[f"host_frac_{a}"] = comp[a] if a in syms else 0.0
    has_O = "O" in syms
    f["host_polyanion"] = float(has_O and any(p in syms for p in POLY))
    tms = [s for s in syms if s in TM]
    f["host_TM_X_mean"] = np.mean([eprop(s, "X") for s in tms]) if tms else np.nan
    f["host_TM_Z_mean"] = np.mean([eprop(s, "Z") for s in tms]) if tms else np.nan
    return f

# --- Working-ion data -------------------------------------------------------
ION_CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3}
# Standard electrode potentials, V vs SHE (M^z+ + z e- -> M)
E0 = {"Li": -3.04, "Na": -2.71, "K": -2.93, "Rb": -2.98, "Cs": -3.03,
      "Mg": -2.37, "Ca": -2.87, "Zn": -0.76, "Al": -1.66}
TRAINED_IONS = ["Li", "Na", "K", "Mg", "Ca", "Zn"]

def ion_features(ion):
    e = Element(ion); q = ION_CHARGE[ion]
    return {"ion_charge": q, "ion_radius": float(e.ionic_radii.get(q, np.nan)),
            "ion_X": e.X, "ion_ionization": e.ionization_energy, "ion_mass": float(e.atomic_mass)}


# --- Oxidation-state features (Week 2b) -------------------------------------
ANION_Q = {"O": -2, "F": -1, "S": -2, "Se": -2, "Te": -2, "N": -3, "Cl": -1, "Br": -1, "I": -1}
# Elements that form positively charged centres of oxo-anions when oxygen is present
OXO_Q = {"P": 5, "Si": 4, "B": 3, "As": 5, "C": 4, "S": 6, "Se": 6, "N": 5, "Cl": 7, "I": 5, "Te": 6, "Ge": 4}
FIXED_Q = {"H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
           "Zn": 2, "Cd": 2, "Al": 3, "Ga": 3, "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3,
           "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3}
P_REDOX = {"Sb", "Sn", "Bi", "Pb", "Tl", "In"}
OX_RANGE = (-4.0, 8.0)          # physically sensible range; values outside -> unreliable

def _common_max(sym):
    try:
        cs = Element(sym).common_oxidation_states
        return max(cs) if cs else np.nan
    except Exception:
        return np.nan

def redox_state(formula):
    """Average oxidation state of redox-active metals by charge balance; NaN if unreliable."""
    comp = Composition(formula)
    syms = {str(e) for e in comp.elements}
    has_O = "O" in syms
    charge, redox = 0.0, {}
    for e in comp.elements:
        s, n = str(e), comp[e]
        if has_O and s in OXO_Q and s != "O":
            charge += OXO_Q[s] * n
        elif s in ANION_Q:
            charge += ANION_Q[s] * n
        elif s in FIXED_Q:
            charge += FIXED_Q[s] * n
        elif e.is_transition_metal or s in P_REDOX:
            redox[s] = n
        # anything else (elemental C, Si, Ge, P without O ...) contributes 0
    n_redox = sum(redox.values())
    if n_redox == 0:
        return np.nan, np.nan, redox
    avg = -charge / n_redox
    cmax = sum(_common_max(s) * n for s, n in redox.items()) / n_redox
    if not (OX_RANGE[0] <= avg <= OX_RANGE[1]):
        return np.nan, cmax, redox          # unphysical -> let the model treat it as missing
    return avg, cmax, redox

def ox_features(framework, discharged):
    comp = Composition(framework)
    syms = [str(e) for e in comp.elements]
    anion_free = not any(s in ANION_Q for s in syms)
    ox_c, cmax, redox = redox_state(framework)
    ox_d = redox_state(discharged)[0] if isinstance(discharged, str) else np.nan
    change = ox_c - ox_d if not (np.isnan(ox_c) or np.isnan(ox_d)) else np.nan
    if not np.isnan(change) and not (0 < change <= 4):
        change = np.nan                      # more than 4 electrons per metal is not physical
    excess = ox_c - cmax if not np.isnan(ox_c) else np.nan
    return {
        "ox_charged": ox_c,
        "ox_discharged": ox_d,
        "ox_change": change,
        "ox_common_max": cmax,
        "ox_excess": excess,
        "ox_above_max": float(excess > 0.2) if not np.isnan(excess) else 0.0,
        "ox_unreliable": float(len(redox) > 0 and np.isnan(ox_c)),
        "no_redox_metal": float(len(redox) == 0),
        "elemental_host": float(len(syms) == 1),
        "anion_free_host": float(anion_free),
    }

# --- Full feature vector ----------------------------------------------------
def discharged_formula(host, ion, x):
    """Host formula + x working ions per formula unit, e.g. ('FePO4', 'Li', 1) -> 'Li1FePO4'."""
    return f"{ion}{x:g}{Composition(host).formula.replace(' ', '')}"

def frac_ion(host, x):
    """Atomic fraction of the working ion in the discharged material (Materials Project 'fracA_discharge')."""
    n_host = Composition(host).num_atoms
    return x / (x + n_host)

def featurize(host, ion, x, columns):
    """Return one feature row (list) in the exact column order used for training."""
    f = {}
    f.update(host_features(host))
    f.update(ion_features(ion))
    f["frac_ion_discharge"] = frac_ion(host, x)
    f.update(ox_features(Composition(host).formula.replace(" ", ""), discharged_formula(host, ion, x)))
    return [f.get(c, np.nan) for c in columns], f
