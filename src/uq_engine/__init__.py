"""
materials-uq-engine
===================
Adaptive Uncertainty Quantification engine for materials science datasets.

Tiered method selection pipeline:
    GUM → Monte Carlo → FSS-PCE → QRNN

Author: Anne Gracia A
"""

from .adaptive_dispatcher import AdaptiveUQDispatcher, UQMethod, DispatchDecision
from .engine import UQEngine, UQResult, UQComponent, UQTypeB, EvaluationMethod

__version__ = "0.1.0"
__author__ = "Anne Gracia A"

__all__ = [
    "AdaptiveUQDispatcher",
    "UQMethod",
    "DispatchDecision",
    "UQEngine",
    "UQResult",
    "UQComponent",
    "UQTypeB",
    "EvaluationMethod",
]
