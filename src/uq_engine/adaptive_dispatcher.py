"""
Adaptive UQ Dispatcher
======================
Routes uncertainty quantification through a tiered method selection pipeline:

    GUM (linear) → Monte Carlo (nonlinear) → FSS-PCE (complex/high-dim) → QRNN (fallback)

Each method is selected automatically based on model linearity diagnostics,
or can be forced explicitly by the user.

Author: Anne Gracia A
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import List, Optional, Callable
from dataclasses import dataclass


class UQMethod(Enum):
    GUM = "GUM"                  # Linear models, analytical propagation
    MONTE_CARLO = "MONTE_CARLO"  # Nonlinear models, sampling-based
    FSS_PCE = "FSS_PCE"          # High-dimensional, sparse surrogate
    QRNN = "QRNN"                # Deep learning fallback for complex/noisy data
    AUTO = "AUTO"                # Let the dispatcher decide


@dataclass
class DispatchDecision:
    selected_method: UQMethod
    reason: str
    linearity_score: float  # 0 = linear, 1 = highly nonlinear
    fallback_used: bool = False


class AdaptiveUQDispatcher:
    """
    Orchestrates the adaptive UQ pipeline.

    Decision logic:
    ---------------
    1. Estimate nonlinearity of the model using finite-difference curvature check.
    2. If linearity_score < LINEAR_THRESHOLD  → use GUM
    3. If linearity_score < NONLINEAR_THRESHOLD → use Monte Carlo
    4. If high-dimensional (n_inputs > DIM_THRESHOLD) → use FSS-PCE
    5. If FSS-PCE fails or data is noisy/irregular → fall back to QRNN
    """

    LINEAR_THRESHOLD = 0.05
    NONLINEAR_THRESHOLD = 0.30
    DIM_THRESHOLD = 4

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _estimate_nonlinearity(
        self,
        func: Callable,
        nominal: dict[str, float],
        std_uncs: dict[str, float],
        n_points: int = 20,
    ) -> float:
        """
        Estimates model nonlinearity using a second-order finite difference curvature index.
        Returns a score between 0 (linear) and 1 (highly nonlinear).
        """
        scores = []
        for name, x0 in nominal.items():
            h = std_uncs.get(name, abs(x0) * 0.01 + 1e-8)
            try:
                y0 = func(**nominal)
                up = {**nominal, name: x0 + h}
                dn = {**nominal, name: x0 - h}
                y_up = func(**up)
                y_dn = func(**dn)
                # Second derivative approximation
                curvature = abs(y_up - 2 * y0 + y_dn) / (h ** 2 + 1e-12)
                first_deriv = abs(y_up - y_dn) / (2 * h + 1e-12)
                score = curvature / (first_deriv + 1e-12)
                scores.append(min(score, 1.0))
            except Exception:
                scores.append(0.5)
        return float(np.mean(scores)) if scores else 0.0

    def decide(
        self,
        func: Callable,
        nominal: dict[str, float],
        std_uncs: dict[str, float],
        force_method: Optional[UQMethod] = None,
    ) -> DispatchDecision:
        """
        Determine which UQ method to use for this problem.
        """
        if force_method and force_method != UQMethod.AUTO:
            return DispatchDecision(
                selected_method=force_method,
                reason=f"User forced method: {force_method.value}",
                linearity_score=-1.0,
            )

        n_inputs = len(nominal)
        linearity_score = self._estimate_nonlinearity(func, nominal, std_uncs)

        if self.verbose:
            print(f"[Dispatcher] Inputs: {n_inputs} | Nonlinearity score: {linearity_score:.4f}")

        if linearity_score < self.LINEAR_THRESHOLD:
            return DispatchDecision(
                selected_method=UQMethod.GUM,
                reason=f"Model is near-linear (score={linearity_score:.4f}). GUM sufficient.",
                linearity_score=linearity_score,
            )
        elif n_inputs > self.DIM_THRESHOLD:
            return DispatchDecision(
                selected_method=UQMethod.FSS_PCE,
                reason=f"High-dimensional input space ({n_inputs} vars). FSS-PCE selected for efficiency.",
                linearity_score=linearity_score,
            )
        elif linearity_score < self.NONLINEAR_THRESHOLD:
            return DispatchDecision(
                selected_method=UQMethod.MONTE_CARLO,
                reason=f"Moderate nonlinearity (score={linearity_score:.4f}). Monte Carlo selected.",
                linearity_score=linearity_score,
            )
        else:
            return DispatchDecision(
                selected_method=UQMethod.FSS_PCE,
                reason=f"High nonlinearity (score={linearity_score:.4f}). FSS-PCE selected.",
                linearity_score=linearity_score,
            )

    def run(
        self,
        func: Callable,
        nominal: dict[str, float],
        std_uncs: dict[str, float],
        data: Optional[np.ndarray] = None,
        force_method: Optional[UQMethod] = None,
        mc_samples: int = 50000,
        seed: int = 42,
    ) -> dict:
        """
        Execute the full adaptive UQ pipeline.

        Parameters
        ----------
        func        : callable model f(**inputs) -> float
        nominal     : dict of nominal input values
        std_uncs    : dict of standard uncertainties per input
        data        : optional raw data array (used for QRNN fallback)
        force_method: override automatic method selection
        mc_samples  : number of samples for MC / PCE
        seed        : random seed for reproducibility

        Returns
        -------
        dict with keys: method, result, decision
        """
        decision = self.decide(func, nominal, std_uncs, force_method)

        if self.verbose:
            print(f"[Dispatcher] Selected: {decision.selected_method.value}")
            print(f"[Dispatcher] Reason  : {decision.reason}")

        try:
            if decision.selected_method == UQMethod.GUM:
                result = self._run_gum(func, nominal, std_uncs)

            elif decision.selected_method == UQMethod.MONTE_CARLO:
                result = self._run_monte_carlo(func, nominal, std_uncs, mc_samples, seed)

            elif decision.selected_method == UQMethod.FSS_PCE:
                result = self._run_fss_pce(func, nominal, std_uncs, mc_samples, seed)

            elif decision.selected_method == UQMethod.QRNN:
                if data is None:
                    raise ValueError("QRNN requires raw data array.")
                result = self._run_qrnn(data)

            return {"method": decision.selected_method.value, "result": result, "decision": decision}

        except Exception as e:
            if self.verbose:
                print(f"[Dispatcher] {decision.selected_method.value} failed: {e}. Falling back to QRNN.")
            if data is not None:
                result = self._run_qrnn(data)
                decision.fallback_used = True
                return {"method": "QRNN (fallback)", "result": result, "decision": decision}
            raise

    # ------------------------------------------------------------------
    # Method executors
    # ------------------------------------------------------------------

    def _run_gum(self, func, nominal, std_uncs) -> dict:
        import math
        y0 = func(**nominal)
        sensitivities = {}
        for name, x0 in nominal.items():
            h = std_uncs.get(name, abs(x0) * 0.01 + 1e-8) * 0.01
            y_up = func(**{**nominal, name: x0 + h})
            y_dn = func(**{**nominal, name: x0 - h})
            sensitivities[name] = (y_up - y_dn) / (2 * h)

        combined_var = sum((sensitivities[n] * std_uncs[n]) ** 2 for n in nominal)
        u_c = math.sqrt(combined_var)
        return {
            "nominal": y0,
            "u_c": u_c,
            "expanded_95": 1.96 * u_c,
            "interval_95": [y0 - 1.96 * u_c, y0 + 1.96 * u_c],
            "sensitivities": sensitivities,
        }

    def _run_monte_carlo(self, func, nominal, std_uncs, n_samples, seed) -> dict:
        rng = np.random.default_rng(seed)
        samples = {
            name: rng.normal(val, std_uncs[name], size=n_samples)
            for name, val in nominal.items()
        }
        outputs = np.array([func(**{n: samples[n][i] for n in nominal}) for i in range(n_samples)])
        return {
            "mean": float(np.mean(outputs)),
            "std": float(np.std(outputs)),
            "lower_95": float(np.percentile(outputs, 2.5)),
            "upper_95": float(np.percentile(outputs, 97.5)),
            "skewness": float(np.mean((outputs - np.mean(outputs)) ** 3) / (np.std(outputs) ** 3 + 1e-12)),
            "samples": outputs,
        }

    def _run_fss_pce(self, func, nominal, std_uncs, n_samples, seed) -> dict:
        """
        Runs FSS-PCE using chaospy for orthogonal polynomial construction.
        Falls back to Monte Carlo if chaospy is unavailable.
        """
        try:
            import chaospy as cp

            distributions = [
                cp.Normal(mu=nominal[n], sigma=std_uncs[n])
                for n in nominal
            ]
            joint = cp.J(*distributions)

            expansion = cp.generate_expansion(order=3, dist=joint, rule="three_terms_recurrence")
            nodes, weights = cp.generate_quadrature(order=3, dist=joint, rule="gaussian")

            evals = np.array([func(**dict(zip(nominal.keys(), nodes[:, i]))) for i in range(nodes.shape[1])])
            approx = cp.fit_quadrature(expansion, nodes, weights, evals)

            mean = float(cp.E(approx, joint))
            std = float(cp.Std(approx, joint))
            sobol = {n: float(cp.Sens_m(approx, joint)[i]) for i, n in enumerate(nominal)}

            return {
                "mean": mean,
                "std": std,
                "lower_95": mean - 1.96 * std,
                "upper_95": mean + 1.96 * std,
                "sobol_indices": sobol,
                "method_detail": "FSS-PCE via chaospy (order=3)",
            }
        except ImportError:
            if self.verbose:
                print("[Dispatcher] chaospy not found. Falling back to Monte Carlo for PCE step.")
            return self._run_monte_carlo(func, nominal, std_uncs, n_samples, seed)

    def _run_qrnn(self, data: np.ndarray) -> dict:
        """Minimal QRNN wrapper for the fallback case."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        data = data[~np.isnan(data)]
        X = torch.tensor(np.linspace(0, 1, len(data)).reshape(-1, 1), dtype=torch.float32)
        Y = torch.tensor(data.reshape(-1, 1), dtype=torch.float32)

        model = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for _ in range(300):
            optimizer.zero_grad()
            preds = model(X)
            quantiles = [0.025, 0.5, 0.975]
            loss = sum(
                torch.max((q - 1) * (Y - preds[:, i:i+1]), q * (Y - preds[:, i:i+1])).mean()
                for i, q in enumerate(quantiles)
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X).numpy()

        return {
            "median": float(np.mean(out[:, 1])),
            "lower_95": float(np.mean(out[:, 0])),
            "upper_95": float(np.mean(out[:, 2])),
            "spread": float(np.mean(out[:, 2]) - np.mean(out[:, 0])),
            "method_detail": "QRNN (Quantile Regression Neural Network)",
        }
