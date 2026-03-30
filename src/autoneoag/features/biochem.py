from __future__ import annotations

import math


HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
AROMATIC = set("FWY")
NON_POLAR = set("AVLIMPFWG")


def gravy(peptide: str) -> float:
    return float(sum(HYDRO.get(aa, 0.0) for aa in peptide) / max(len(peptide), 1))


def aromaticity(peptide: str) -> float:
    return sum(aa in AROMATIC for aa in peptide) / max(len(peptide), 1)


def non_polar_ratio(peptide: str) -> float:
    return sum(aa in NON_POLAR for aa in peptide) / max(len(peptide), 1)


def delta_residue_fraction(mut: str, wt: str) -> float:
    mismatches = sum(a != b for a, b in zip(mut, wt))
    mismatches += abs(len(mut) - len(wt))
    return mismatches / max(len(mut), len(wt), 1)


def log_safe_ratio(num: float, den: float) -> float:
    return math.log((num + 1e-6) / (den + 1e-6))

