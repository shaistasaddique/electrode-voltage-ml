import numpy as np
from pymatgen.core import Composition, Element

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
