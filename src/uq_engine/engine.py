"""SUnCal-backed uncertainty quantification engine.

Implements Phase C of the DIP measurement uncertainty pipeline.
Three data-source paths feed a unified SUnCal Model builder that
produces GUM and Monte Carlo results, storing them in dip_uq_results.
"""

from __future__ import annotations

import json
import math
import statistics
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Callable

# ---------------------------------------------------------------------------\n# Data models\n# ---------------------------------------------------------------------------

@dataclass
class UQTypeB:
    description: str
    unc: float
    distribution: str   # "normal" | "rectangular" | "triangular"
    k: float            # divisor: 1.0 normal, sqrt(3) rectangular, sqrt(6) triangular
    source: str         # "instrument_spec" | \"calibration\" | \"literature\" | \"estimated\" | \"user\"


@dataclass
class UQComponent:
    name: str
    value: float
    unit: str
    typea_unc: Optional[float]
    typea_dof: Optional[int]
    typea_values: Optional[list[float]]
    typeb_components: list[UQTypeB]
    source_type: str    # "fit" | \"digitized\" | \"tabular\" | \"manual\"
    source_id: Optional[str]


@dataclass
class UQResult:
    uq_id: str
    model_expression: str
    components: list[UQComponent]

    gum_value: float
    gum_unc_k1: float
    gum_expanded_95: float
    gum_coverage_factor: float
    gum_eff_dof: float
    gum_sensitivity: dict[str, float]
    gum_proportions: dict[str, float]

    mc_mean: float
    mc_std: float
    mc_lower_95: float
    mc_upper_95: float
    mc_skewness: float
    mc_histogram: list[dict[str, float]]
    mc_samples_path: Optional[str]

    gum_mc_agreement_pct: float
    dominant_contributor: str
    warnings: list[str]
    
    # New metadata field preserving backward compatibility while tracking modern extensions
    selected_method: str = "GUM_AND_MC"


# ---------------------------------------------------------------------------\n# NEW ADDITION: User Method Selection Enumerator\n# ---------------------------------------------------------------------------
class EvaluationMethod(Enum):
    GUM_AND_MC = "GUM_AND_MC"          # Traditional hybrid execution (Original)
    LATIN_HYPERCUBE = "LATIN_HYPERCUBE"  # Efficient stratified space filling
    DELTA_METHOD = "DELTA_METHOD"        # Direct analytical propagation 
    BOOTSTRAP = "BOOTSTRAP"              # Parametric bootstrap resampling


# ---------------------------------------------------------------------------\n# Calculation Engine Class\n# ---------------------------------------------------------------------------

class UQEngine:
    def __init__(self, db_sync_enabled: bool = True):
        self.db_sync_enabled = db_sync_enabled
        self._FD_STEP = 1e-5

    # -----------------------------------------------------------------------\n    # Data-source builders\n    # -----------------------------------------------------------------------

    def build_from_fit(self, param_name: str, opt_val: float, cov_val: float, 
                       dof: int, unit: str = "", fit_id: str = None) -> UQComponent:
        comp = UQComponent(
            name=param_name,
            value=opt_val,
            unit=unit,
            typea_unc=math.sqrt(cov_val) if cov_val > 0 else 0.0,
            typea_dof=dof,
            typea_values=None,
            typeb_components=[],
            source_type="fit",
            source_id=fit_id
        )
        return comp

    def build_from_digitized(self, feature_name: str, nominal_value: float, 
                             user_confidence: float, full_scale: float, 
                             pixel_resolution: float, unit: str = "", 
                             digitized_id: str = None) -> UQComponent:
        typea_estimate = (1.0 - max(0.0, min(1.0, user_confidence))) * 0.05 * full_scale
        
        typeb = UQTypeB(
            description="Digitization pixel resolution limit",
            unc=pixel_resolution,
            distribution="rectangular",
            k=math.sqrt(3.0),
            source="estimated"
        )
        
        comp = UQComponent(
            name=feature_name,
            value=nominal_value,
            unit=unit,
            typea_unc=typea_estimate,
            typea_dof=9,  # fallback representation
            typea_values=None,
            typeb_components=[typeb],
            source_type="digitized",
            source_id=digitized_id
        )
        return comp

    def build_from_tabular(self, variable_name: str, data_values: list[float], 
                           unit: str = "", tabular_id: str = None) -> UQComponent:
        if not data_values:
            raise ValueError(f"Empty data array supplied for tabular input: {variable_name}")
        
        n = len(data_values)
        val = statistics.mean(data_values)
        
        if n > 1:
            std_dev = statistics.stdev(data_values)
            typea_unc = std_dev / math.sqrt(n)
            dof = n - 1
        else:
            typea_unc = 0.0
            dof = 1
            
        comp = UQComponent(
            name=variable_name,
            value=val,
            unit=unit,
            typea_unc=typea_unc,
            typea_dof=dof,
            typea_values=data_values,
            typeb_components=[],
            source_type="tabular",
            source_id=tabular_id
        )
        return comp

    def build_manual(self, name: str, value: float, typea_unc: float, typea_dof: int, 
                     typeb_list: list[UQTypeB], unit: str = "") -> UQComponent:
        return UQComponent(
            name=name,
            value=value,
            unit=unit,
            typea_unc=typea_unc,
            typea_dof=typea_dof,
            typea_values=None,
            typeb_components=typeb_list,
            source_type="manual",
            source_id=None
        )

    # -----------------------------------------------------------------------\n    # Internal Mathematical Helpers\n    # -----------------------------------------------------------------------

    def _coverage_factor(self, eff_dof: float, coverage: float = 0.95) -> float:
        if eff_dof <= 0:
            return 2.0
        try:
            from scipy.types import ScalarType # fallback safety check
        except ImportError:
            pass
        try:
            import scipy.stats as stats
            return float(stats.t.ppf((1.0 + coverage) / 2.0, df=eff_dof))
        except ImportError:
            if eff_dof < 5: return 2.78
            if eff_dof < 10: return 2.26
            if eff_dof < 20: return 2.09
            return 1.96

    def _welch_satterthwaite(self, std_uncs: list[float], dofs: list[float], 
                             sensitivity_coeffs: list[float]) -> float:
        combined_unc_sq = sum((c * u) ** 2 for c, u in zip(sensitivity_coeffs, std_uncs))
        if combined_unc_sq == 0:
            return float('inf')
        
        denominator = 0.0
        for u, df, c in zip(std_uncs, dofs, sensitivity_coeffs):
            product = c * u
            if product != 0 and df > 0:
                denominator += (product ** 4) / df
                
        if denominator == 0:
            return float('inf')
        return (combined_unc_sq ** 2) / denominator

    def _finite_diff_sensitivity(self, func: Callable, nominal: dict[str, float], 
                                 var_name: str) -> float:
        x0 = nominal[var_name]
        h = self._FD_STEP * (abs(x0) if x0 != 0 else 1.0)
        
        nominal[var_name] = x0 + h
        y_high = func(**nominal)
        
        nominal[var_name] = x0 - h
        y_low = func(**nominal)
        
        nominal[var_name] = x0  # restore
        return (y_high - y_low) / (2.0 * h)

    def _parse_model_expression(self, expr: str) -> Callable:
        _allowed_globals = {
            "math": math,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
            "pi": math.pi, "e": math.e, "abs": abs
        }
        clean_expr = expr.replace("^", "**")
        code = compile(clean_expr, "<string>", "eval")
        
        def _func(**kwargs):
            return eval(code, _allowed_globals, kwargs)
        return _func

    # -----------------------------------------------------------------------\n    # Core Calculation Pipeline\n    # -----------------------------------------------------------------------

    def calculate(self, components: list[UQComponent], model_expression: str, 
                  mc_samples: int = 50000, seed: int = None,
                  method: EvaluationMethod = EvaluationMethod.GUM_AND_MC) -> UQResult:
        """Evaluates uncertainty quantification matching user preference."""
        
        func = self._parse_model_expression(model_expression)
        nominal_dict = {c.name: c.value for c in components}
        
        # 1. Base Evaluation Setup
        try:
            y_nominal = func(**nominal_dict)
        except Exception as e:
            raise ValueError(f"Failed to evaluate model expression mapping basic components: {str(e)}")

        # 2. Extract Individual Component Contributions
        comp_std_uncs = []
        comp_dofs = []
        comp_names = [c.name for c in components]
        
        for c in components:
            # Component aggregate Type A
            u_a = c.typea_unc if c.typea_unc is not None else 0.0
            var_a = u_a ** 2
            df_a = c.typea_dof if (c.typea_dof is not None and c.typea_dof > 0) else float('inf')
            
            # Component aggregate Type B
            var_b = 0.0
            for b in c.typeb_components:
                divisor = b.k if b.k > 0 else 1.0
                var_b += (b.unc / divisor) ** 2
                
            combined_u = math.sqrt(var_a + var_b)
            comp_std_uncs.append(combined_u)
            
            # Combined Component Degrees of Freedom Assignment
            if var_b == 0:
                comp_dofs.append(df_a)
            elif var_a == 0:
                comp_dofs.append(float('inf'))
            else:
                # Local Welch-Satterthwaite for individual component subparts
                u_total_sq = var_a + var_b
                denom = (var_a ** 2) / df_a
                comp_dofs.append((u_total_sq ** 2) / denom if denom > 0 else float('inf'))

        # 3. Calculate Sensitivity Coefficients
        sensitivities = {}
        for c in components:
            sensitivities[c.name] = self._finite_diff_sensitivity(func, nominal_dict, c.name)

        sens_list = [sensitivities[name] for name in comp_names]

        # 4. Routing logic depending on user-specified method
        warnings_list = []
        
        if method == EvaluationMethod.DELTA_METHOD:
            # Direct analytical execution bypassing Monte Carlo generation to eliminate compute overhead
            gum_unc_k1, gum_eff_dof, k_factor, gum_expanded_95, proportions = self._run_gum_core(
                comp_std_uncs, comp_dofs, sens_list, comp_names
            )
            result = UQResult(
                uq_id=str(uuid.uuid4()), model_expression=model_expression, components=components,
                gum_value=y_nominal, gum_unc_k1=gum_unc_k1, gum_expanded_95=gum_expanded_95,
                gum_coverage_factor=k_factor, gum_eff_dof=gum_eff_dof, gum_sensitivity=sensitivities, gum_proportions=proportions,
                mc_mean=y_nominal, mc_std=gum_unc_k1, mc_lower_95=y_nominal - gum_expanded_95, mc_upper_95=y_nominal + gum_expanded_95,
                mc_skewness=0.0, mc_histogram=[], mc_samples_path=None, gum_mc_agreement_pct=100.0,
                dominant_contributor=max(proportions, key=proportions.get) if proportions else "None",
                warnings=warnings_list + ["Evaluated solely via analytical Delta Method."],
                selected_method=method.value
            )
            self._sync_to_db_if_enabled(result)
            return result

        elif method == EvaluationMethod.LATIN_HYPERCUBE:
            # Advanced Space-Filling Latin Hypercube Stratified Execution Loop
            mc_outputs = self._execute_lhs_sampling(components, comp_std_uncs, func, mc_samples, seed)
            warnings_list.append("Evaluated using space-filling Latin Hypercube Stratified Sampling.")
            
        elif method == EvaluationMethod.BOOTSTRAP:
            # Parametric Distribution Resampling Loop
            import numpy as np
            rng = np.random.default_rng(seed)
            mc_outputs = self._execute_parametric_bootstrap(components, comp_std_uncs, func, mc_samples, rng)
            warnings_list.append("Evaluated using Parametric Bootstrap distribution resampling.")
            
        else:
            # Default fallback routing: Standard GUM + Pure Random Monte Carlo (Original Approach)
            import numpy as np
            rng = np.random.default_rng(seed)
            mc_outputs = np.zeros(mc_samples)
            
            # Original vector sampling mechanism preserved intact
            sampled_inputs = {}
            for c, std_u in zip(components, comp_std_uncs):
                sampled_inputs[c.name] = rng.normal(loc=c.value, scale=std_u, size=mc_samples)
                
            for i in range(mc_samples):
                row_kwargs = {name: arr[i] for name, arr in sampled_inputs.items()}
                mc_outputs[i] = func(**row_kwargs)

        # 5. Extract Analytics Matrix Metrics (Shared across simulation frameworks)
        import numpy as np
        mc_mean = float(np.mean(mc_outputs))
        mc_std = float(np.std(mc_outputs, ddof=1))
        
        mc_lower_95 = float(np.percentile(mc_outputs, 2.5))
        mc_upper_95 = float(np.percentile(mc_outputs, 97.5))
        
        # Skewness estimation
        if mc_std > 0:
            mc_skewness = float(np.mean((mc_outputs - mc_mean) ** 3) / (mc_std ** 3))
        else:
            mc_skewness = 0.0

        # Run Standard baseline GUM analytics to run divergence cross-comparisons
        gum_unc_k1, gum_eff_dof, k_factor, gum_expanded_95, proportions = self._run_gum_core(
            comp_std_uncs, comp_dofs, sens_list, comp_names
        )

        # Divergence evaluation
        if gum_unc_k1 > 0:
            agreement = (1.0 - abs(gum_unc_k1 - mc_std) / gum_unc_k1) * 100.0
            gum_mc_agreement_pct = max(0.0, float(agreement))
        else:
            gum_mc_agreement_pct = 100.0

        if gum_mc_agreement_pct < 90.0:
            warnings_list.append(
                f"Low agreement ({gum_mc_agreement_pct:.1f}%) detected between analytical and simulation bounds. "
                f"Review structural model linearity dependencies."
            )

        # Construct raw summary tracking metrics for charts
        counts, bin_edges = np.histogram(mc_outputs, bins=30)
        mc_histogram = []
        for c_val, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
            mc_histogram.append({"bin_center": float((left + right) / 2.0), "relative_frequency": float(c_val) / mc_samples})

        dom_item = max(proportions, key=proportions.get) if proportions else "None"

        result = UQResult(
            uq_id=str(uuid.uuid4()), model_expression=model_expression, components=components,
            gum_value=y_nominal, gum_unc_k1=gum_unc_k1, gum_expanded_95=gum_expanded_95,
            gum_coverage_factor=k_factor, gum_eff_dof=gum_eff_dof, gum_sensitivity=sensitivities, gum_proportions=proportions,
            mc_mean=mc_mean, mc_std=mc_std, mc_lower_95=mc_lower_95, mc_upper_95=mc_upper_95,
            mc_skewness=mc_skewness, mc_histogram=mc_histogram, mc_samples_path=None,
            gum_mc_agreement_pct=gum_mc_agreement_pct, dominant_contributor=dom_item,
            warnings=warnings_list, selected_method=method.value
        )

        self._sync_to_db_if_enabled(result)
        return result

    # -----------------------------------------------------------------------\n    # NEW METHOD EXECUTION EXTRACTIONS (Internal Isolated Plugins)\n    # -----------------------------------------------------------------------

    def _run_gum_core(self, std_uncs: list[float], dofs: list[float], 
                      sens_list: list[float], comp_names: list[str]) -> tuple:
        """Internal isolation encapsulating standard ISO-GUM aggregation math."""
        combined_unc_sq = sum((c * u) ** 2 for c, u in zip(sens_list, std_uncs))
        gum_unc_k1 = math.sqrt(combined_unc_sq)
        
        gum_eff_dof = self._welch_satterthwaite(std_uncs, dofs, sens_list)
        k_factor = self._coverage_factor(gum_eff_dof, 0.95)
        gum_expanded_95 = k_factor * gum_unc_k1
        
        proportions = {}
        if combined_unc_sq > 0:
            for name, c, u in zip(comp_names, sens_list, std_uncs):
                proportions[name] = float(((c * u) ** 2) / combined_unc_sq)
        else:
            for name in comp_names: proportions[name] = 0.0
            
        return gum_unc_k1, gum_eff_dof, k_factor, gum_expanded_95, proportions

    def _execute_lhs_sampling(self, components: list[UQComponent], comp_std_uncs: list[float], 
                              func: Callable, mc_samples: int, seed: int) -> Any:
        """Executes a space-filling Latin Hypercube Stratified sampling loop."""
        import numpy as np
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(components), seed=seed)
            sample_matrix = sampler.random(n=mc_samples) # uniform probability grid [0, 1]
        except (ImportError, AttributeError):
            # Safe clean programmatic fallback loop for LHS matrix calculation if SciPy version is locked out
            rng = np.random.default_rng(seed)
            d = len(components)
            sample_matrix = np.empty((mc_samples, d))
            for j in range(d):
                perm = rng.permutation(mc_samples)
                intervals = (perm + rng.uniform(size=mc_samples)) / mc_samples
                sample_matrix[:, j] = intervals

        import scipy.stats as stats
        mc_outputs = np.empty(mc_samples)
        
        # Map probability thresholds out to inverse CDF normal distributions for parameter stability
        mapped_inputs = {}
        for j, (c, std_u) in enumerate(zip(components, comp_std_uncs)):
            mapped_inputs[c.name] = stats.norm.ppf(sample_matrix[:, j], loc=c.value, scale=std_u)

        for i in range(mc_samples):
            row_vals = {name: arr[i] for name, arr in mapped_inputs.items()}
            mc_outputs[i] = func(**row_vals)
            
        return mc_outputs

    def _execute_parametric_bootstrap(self, components: list[UQComponent], comp_std_uncs: list[float], 
                                      func: Callable, mc_samples: int, rng: Any) -> Any:
        """Resamples inputs directly using distinct underlying parametric shapes."""
        import numpy as np
        mc_outputs = np.empty(mc_samples)
        sampled_inputs = {}

        for c in components:
            # Check if there is raw data available to run an empirical bootstrap
            if c.typea_values and len(c.typea_values) > 1:
                sampled_inputs[c.name] = rng.choice(c.typea_values, size=mc_samples, replace=True)
            else:
                # Fallback to parametric assumptions depending on Type B properties mapped out by user
                if c.typeb_components and len(c.typeb_components) == 1:
                    tb = c.typeb_components[0]
                    if tb.distribution == "rectangular":
                        half_width = tb.unc
                        sampled_inputs[c.name] = rng.uniform(c.value - half_width, c.value + half_width, size=mc_samples)
                    elif tb.distribution == "triangular":
                        half_width = tb.unc
                        sampled_inputs[c.name] = rng.triangular(c.value - half_width, c.value, c.value + half_width, size=mc_samples)
                    else:
                        u_a = c.typea_unc if c.typea_unc is not None else 0.0
                        u_b = tb.unc / (tb.k if tb.k > 0 else 1.0)
                        u_combined = math.sqrt(u_a**2 + u_b**2)
                        sampled_inputs[c.name] = rng.normal(c.value, u_combined, size=mc_samples)
                else:
                    # Combined normal representation assignment standard mapping
                    u_a = c.typea_unc if c.typea_unc is not None else 0.0
                    var_b = sum((b.unc / (b.k if b.k > 0 else 1.0))**2 for b in c.typeb_components)
                    u_combined = math.sqrt(u_a**2 + var_b)
                    sampled_inputs[c.name] = rng.normal(c.value, u_combined, size=mc_samples)

        for i in range(mc_samples):
            row_vals = {name: arr[i] for name, arr in sampled_inputs.items()}
            mc_outputs[i] = func(**row_vals)
            
        return mc_outputs

    # -----------------------------------------------------------------------\n    # Database Syncer and Formatters\n    # -----------------------------------------------------------------------

    def _sync_to_db_if_enabled(self, result: UQResult) -> None:
        if not self.db_sync_enabled:
            return
        try:
            from sqlalchemy import text
            from dip.db.engine import SessionLocal
            
            payload = _uq_result_to_dict(result)
            session = SessionLocal()
            try:
                stmt = text(\"\"\"
                    INSERT INTO dip_uq_results (uq_id, model_expression, payload, created_at)
                    VALUES (:uq_id, :model, :payload, :now)
                \"\"\")
                session.execute(stmt, {
                    "uq_id": result.uq_id,
                    "model": result.model_expression,
                    "payload": json.dumps(payload),
                    "now": datetime.now(timezone.utc)
                })
                session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()
        except ImportError:
            pass

    def format_uq_answer(self, result: UQResult) -> str:
        lines = [
            f"======================================================================",
            f"UNCERTAINTY QUANTIFICATION REPORT",
            f"======================================================================",
            f"Selected Execution Method : {result.selected_method}",
            f"Model Mathematical Formula: y = {result.model_expression}",
            f"Calculated Nominal Value y: {result.gum_value:.6g}",
            f"----------------------------------------------------------------------",
            f"Analytical (ISO-GUM) Path metrics:",
            f"  Standard Combined Uncertainty (u_c) [k=1]  : {result.gum_unc_k1:.6g}",
            f"  Effective Degrees of Freedom (v_eff)       : {result.gum_eff_dof:.2f}",
            f"  Coverage Factor (k_95) [Student-t lookup]  : {result.gum_coverage_factor:.4f}",
            f"  Expanded Uncertainty (U_95)                : {result.gum_expanded_95:.6g}",
            f"  95% Confidence Interval Bounds             : [{result.gum_value - result.gum_expanded_95:.6g}, {result.gum_value + result.gum_expanded_95:.6g}]",
            f"----------------------------------------------------------------------",
            f"Simulation (Statistical Output Profile) Path metrics:",
            f"  Evaluated Distribution Center Mean         : {result.mc_mean:.6g}",
            f"  Evaluated Distribution Standard Deviation  : {result.mc_std:.6g}",
            f"  95% Percentile Confidence Range            : [{result.mc_lower_95:.6g}, {result.mc_upper_95:.6g}]",
            f"  Calculated Skewness Factor                 : {result.mc_skewness:.3f}",
            f"----------------------------------------------------------------------",
            f"System Correlation Integrity Checking:",
            f"  GUM / Simulation Deviation Convergence    : {result.gum_mc_agreement_pct:.1f}%",
            f"  Dominant System Input Contributor          : {result.dominant_contributor}",
        ]
        if result.warnings:
            lines.append("System Diagnostics Alerts:")
            for w in result.warnings:
                lines.append(f"  * WARNING: {w}")
        lines.append("======================================================================")
        return "\n".join(lines)


# ---------------------------------------------------------------------------\n# Payload serialization dictionary helpers\n# ---------------------------------------------------------------------------

def _typeb_to_dict(b: UQTypeB) -> dict:
    return {
        "description": b.description,
        "unc": b.unc,
        "distribution": b.distribution,
        "k": b.k,
        "source": b.source
    }


def _component_to_dict(c: UQComponent) -> dict:
    return {
        "name": c.name,
        "value": c.value,
        "unit": c.unit,
        "typea_unc": c.typea_unc,
        "typea_dof": c.typea_dof,
        "typea_values": c.typea_values,
        "typeb_components": [_typeb_to_dict(b) for b in c.typeb_components],
        "source_type": c.source_type,
        "source_id": c.source_id
    }


def _uq_result_to_dict(result: UQResult) -> dict:
    return {
        "uq_id": result.uq_id,
        "model_expression": result.model_expression,
        "components": [_component_to_dict(c) for c in result.components],
        "gum_value": result.gum_value,
        "gum_unc_k1": result.gum_unc_k1,
        "gum_expanded_95": result.gum_expanded_95,
        "gum_coverage_factor": result.gum_coverage_factor,
        "gum_eff_dof": result.gum_eff_dof,
        "gum_sensitivity": result.gum_sensitivity,
        "gum_proportions": result.gum_proportions,
        "mc_mean": result.mc_mean,
        "mc_std": result.mc_std,
        "mc_lower_95": result.mc_lower_95,
        "mc_upper_95": result.mc_upper_95,
        "mc_skewness": result.mc_skewness,
        "mc_histogram": result.mc_histogram,
        "mc_samples_path": result.mc_samples_path,
        "gum_mc_agreement_pct": result.gum_mc_agreement_pct,
        "dominant_contributor": result.dominant_contributor,
        "warnings": result.warnings,
        "selected_method": result.selected_method
    }


def _get_technique_for_expdata(expparse_id: str) -> str | None:
    """Look up the technique field for an expparse experiment."""
    try:
        from sqlalchemy import text
        from dip.db.engine import SessionLocal
        session = SessionLocal()
        try:
            row = session.execute(
                text("SELECT technique FROM expparse_experiments WHERE id = :id"),
                {"id": expparse_id}
            ).fetchone()
            return row[0] if row else None
        finally:
            session.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------\n# Execution Verification Block\n# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiating backend-sync offline verification tests
    engine = UQEngine(db_sync_enabled=False)

    comp_A = engine.build_manual(
        name="resistance", value=100.0, typea_unc=0.1, typea_dof=9,
        typeb_list=[UQTypeB("Spec accuracy", 0.05, "rectangular", math.sqrt(3.0), "instrument_spec")]
    )
    comp_B = engine.build_manual(
        name="current", value=2.5, typea_unc=0.02, typea_dof=14,
        typeb_list=[UQTypeB("Calibration stability uncertainty offset", 0.01, "normal", 1.0, "calibration")]
    )

    # Example 1: Executing standard classic framework (Default)
    print("--- DEMO 1: CLASSIC RE-ROUTED GUM + MC FRAMEWORK ---")
    res_classic = engine.calculate([comp_A, comp_B], "resistance * current", method=EvaluationMethod.GUM_AND_MC)
    print(engine.format_uq_answer(res_classic))
    print("\n")

    # Example 2: Selecting advanced Latin Hypercube Stratified Sampling
    print("--- DEMO 2: LATIN HYPERCUBE SPACE FILLING SAMPLING ---")
    res_lhs = engine.calculate([comp_A, comp_B], "resistance * current", mc_samples=5000, method=EvaluationMethod.LATIN_HYPERCUBE)
    print(engine.format_uq_answer(res_lhs))