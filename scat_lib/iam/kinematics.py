"""Scattering-geometry helpers.

Defines three commonly used magnitudes:

- s = sin(theta)/lambda        [Å^-1]  (X-ray crystallography convention)
- g = 2*s = 1/d                [Å^-1]  (electron crystallography convention)
- q = 4*pi*s                   [Å^-1]  (momentum transfer magnitude)

The relations are:
    g = 2*s
    q = 4*pi*s
    s = g/2 = q/(4*pi)

All functions below are pure math (no units inferred).
"""
from __future__ import annotations
import math

def s_from_g(g: float) -> float:
    return 0.5 * g

def g_from_s(s: float) -> float:
    return 2.0 * s

def s_from_q(q: float) -> float:
    return q / (4.0 * math.pi)

def q_from_s(s: float) -> float:
    return 4.0 * math.pi * s

