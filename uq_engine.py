"""
UQ Engine — Adaptive Uncertainty Quantification for Materials Datasets
======================================================================
Implements three UQ methods with automatic selection:

    1. GUM       — linear models, analytical propagation (ISO GUM)
    2. Monte Carlo — nonlinear models, sampling-based
    3. QRNN      — deep learning quantile regression (fallback for noisy data)

Auto-selection logic:
    - Compute coefficient of variation (CV) of the data
    - If CV < 0.05 → GUM
    - If CV < 0.20 → Monte Carlo
    - Else         → QRNN

Author: Anne Gracia A
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class UQResult:
    method: str
    target_column: str
    n_samples: int
    mean: float
    std: float
    lower_95: float
    upper_95: float
    cv: float               # coefficient of variation
    skewness: float
    notes: str = ""


# ─────────────────────────────────────────────────────────────────
# Method 1: GUM — analytical uncertainty propagation
# ─────────────────────────────────────────────────────────────────

def run_gum(data: np.ndarray) -> UQResult:
    """
    GUM (Guide to the Expression of Uncertainty in Measurement).
    Treats the data as repeated measurements of a single quantity.
    Computes Type A uncertainty from sample statistics.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    u_mean = std / math.sqrt(n)          # standard uncertainty of the mean

    # Coverage factor for 95% CI using t-distribution
    from scipy import stats
    k = float(stats.t.ppf(0.975, df=n - 1))
    expanded_u = k * u_mean

    cv = std / abs(mean) if mean != 0 else 0.0

    return UQResult(
        method="GUM",
        target_column="",
        n_samples=n,
        mean=mean,
        std=u_mean,
        lower_95=mean - expanded_u,
        upper_95=mean + expanded_u,
        cv=cv,
        skewness=float(stats.skew(data)),
        notes=f"k={k:.3f}, u_mean={u_mean:.4f}, expanded_U95={expanded_u:.4f}"
    )


# ─────────────────────────────────────────────────────────────────
# Method 2: Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────

def run_monte_carlo(data: np.ndarray, n_samples: int = 50000, seed: int = 42) -> UQResult:
    """
    Monte Carlo uncertainty propagation.
    Fits a normal distribution to the data and samples from it.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    cv = std / abs(mean) if mean != 0 else 0.0

    # Sample from fitted distribution
    samples = rng.normal(loc=mean, scale=std, size=n_samples)

    lower_95 = float(np.percentile(samples, 2.5))
    upper_95 = float(np.percentile(samples, 97.5))
    skew = float(stats.skew(data))

    return UQResult(
        method="Monte Carlo",
        target_column="",
        n_samples=len(data),
        mean=mean,
        std=std,
        lower_95=lower_95,
        upper_95=upper_95,
        cv=cv,
        skewness=skew,
        notes=f"n_mc_samples={n_samples}, fitted Normal(μ={mean:.3f}, σ={std:.3f})"
    ), samples


# ─────────────────────────────────────────────────────────────────
# Method 3: QRNN — Quantile Regression Neural Network
# ─────────────────────────────────────────────────────────────────

def run_qrnn(data: np.ndarray, epochs: int = 300) -> UQResult:
    """
    QRNN: Deep learning quantile regression.
    Trains a small neural network to predict [2.5%, 50%, 97.5%] quantiles.
    Best for noisy, nonlinear, or non-Gaussian datasets.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from scipy import stats

    data_clean = data[~np.isnan(data)]
    n = len(data_clean)

    X = torch.tensor(np.linspace(0, 1, n).reshape(-1, 1), dtype=torch.float32)
    Y = torch.tensor(data_clean.reshape(-1, 1), dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 3)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    quantiles = [0.025, 0.5, 0.975]

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = sum(
            torch.max((q - 1) * (Y - preds[:, i:i+1]),
                       q * (Y - preds[:, i:i+1])).mean()
            for i, q in enumerate(quantiles)
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(X).numpy()

    lower_95 = float(np.mean(out[:, 0]))
    median   = float(np.mean(out[:, 1]))
    upper_95 = float(np.mean(out[:, 2]))
    spread   = upper_95 - lower_95
    mean_val = float(np.mean(data_clean))
    std_val  = float(np.std(data_clean, ddof=1))
    cv       = std_val / abs(mean_val) if mean_val != 0 else 0.0

    return UQResult(
        method="QRNN",
        target_column="",
        n_samples=n,
        mean=median,
        std=spread / 3.92,   # approximate std from 95% interval
        lower_95=lower_95,
        upper_95=upper_95,
        cv=cv,
        skewness=float(stats.skew(data_clean)),
        notes=f"epochs={epochs}, spread={spread:.4f}, QRNN predicted median={median:.4f}"
    ), out, data_clean


# ─────────────────────────────────────────────────────────────────
# Auto dispatcher
# ─────────────────────────────────────────────────────────────────

def auto_select_method(data: np.ndarray) -> str:
    """Select UQ method based on coefficient of variation."""
    mean = np.mean(data)
    std  = np.std(data, ddof=1)
    cv   = std / abs(mean) if mean != 0 else 0.0

    if cv < 0.05:
        return "GUM"
    elif cv < 0.20:
        return "Monte Carlo"
    else:
        return "QRNN"


def run_uq(data: np.ndarray, method: str = "AUTO") -> dict:
    """
    Main entry point. Runs UQ on a 1D numpy array.

    Parameters
    ----------
    data   : 1D numpy array of measurement values
    method : "AUTO", "GUM", "Monte Carlo", or "QRNN"

    Returns
    -------
    dict with keys: result, extras (samples/qrnn outputs for plotting)
    """
    data = data[~np.isnan(data)].astype(float)

    if method == "AUTO":
        method = auto_select_method(data)
        print(f"[AutoSelect] CV={np.std(data,ddof=1)/abs(np.mean(data)):.4f} → Method: {method}")

    if method == "GUM":
        result = run_gum(data)
        return {"result": result, "extras": None, "raw_data": data}

    elif method == "Monte Carlo":
        result, samples = run_monte_carlo(data)
        return {"result": result, "extras": samples, "raw_data": data}

    elif method == "QRNN":
        result, qrnn_out, data_clean = run_qrnn(data)
        return {"result": result, "extras": qrnn_out, "raw_data": data_clean}

    else:
        raise ValueError(f"Unknown method: {method}. Choose AUTO, GUM, Monte Carlo, or QRNN.")


# ─────────────────────────────────────────────────────────────────
# Plotly visualizations
# ─────────────────────────────────────────────────────────────────

def plot_results(uq_output: dict, save_path: str = None) -> go.Figure:
    """
    Generate interactive Plotly figure from UQ results.
    Saves as HTML if save_path is provided.
    """
    result = uq_output["result"]
    extras = uq_output["extras"]
    raw    = uq_output["raw_data"]
    method = result.method

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"UQ Result — {result.target_column} ({method})",
            "Uncertainty Summary"
        ],
        column_widths=[0.65, 0.35]
    )

    x_axis = np.arange(len(raw))

    # ── Left panel ──────────────────────────────────────────────
    # Raw data points
    fig.add_trace(go.Scatter(
        x=x_axis, y=raw,
        mode="markers",
        name="Measurements",
        marker=dict(color="#34495E", size=8, opacity=0.8)
    ), row=1, col=1)

    if method == "GUM":
        # Horizontal band for expanded uncertainty
        fig.add_hrect(
            y0=result.lower_95, y1=result.upper_95,
            fillcolor="#2ECC71", opacity=0.15,
            annotation_text="95% CI (GUM)",
            annotation_position="top left",
            row=1, col=1
        )
        fig.add_hline(y=result.mean, line_color="#2ECC71", line_width=2,
                      annotation_text=f"Mean = {result.mean:.3f}", row=1, col=1)

    elif method == "Monte Carlo" and extras is not None:
        fig.add_hline(y=result.mean, line_color="#3498DB", line_width=2,
                      annotation_text=f"Mean = {result.mean:.3f}", row=1, col=1)
        fig.add_hrect(
            y0=result.lower_95, y1=result.upper_95,
            fillcolor="#3498DB", opacity=0.12,
            annotation_text="95% CI (MC)",
            annotation_position="top left",
            row=1, col=1
        )

    elif method == "QRNN" and extras is not None:
        fig.add_trace(go.Scatter(
            x=x_axis, y=extras[:, 1],
            mode="lines", name="QRNN Median (50%)",
            line=dict(color="#2ECC71", width=3)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=extras[:, 2],
            mode="lines", name="Upper Bound (97.5%)",
            line=dict(color="#E74C3C", width=2, dash="dash")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=extras[:, 0],
            mode="lines", name="Lower Bound (2.5%)",
            line=dict(color="#3498DB", width=2, dash="dash"),
            fill="tonexty", fillcolor="rgba(52,152,219,0.08)"
        ), row=1, col=1)

    # ── Right panel: summary bar chart ──────────────────────────
    metrics = ["Mean", "Lower 95%", "Upper 95%", "Std Dev"]
    values  = [result.mean, result.lower_95, result.upper_95, result.std]
    colors  = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12"]

    fig.add_trace(go.Bar(
        x=metrics, y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        name="Statistics"
    ), row=1, col=2)

    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Uncertainty Quantification — {result.target_column} | Method: {method}",
            font=dict(size=16)
        ),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        annotations=[
            dict(
                text=(
                    f"n={result.n_samples}  |  "
                    f"CV={result.cv:.3f}  |  "
                    f"Skewness={result.skewness:.3f}<br>"
                    f"{result.notes}"
                ),
                xref="paper", yref="paper",
                x=0.01, y=-0.12,
                showarrow=False,
                font=dict(size=11, color="#7F8C8D"),
                align="left"
            )
        ]
    )

    fig.update_xaxes(title_text="Measurement Index", row=1, col=1)
    fig.update_yaxes(title_text=result.target_column, row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    if save_path:
        fig.write_html(save_path)
        print(f"[Saved] {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────
# Summary table HTML
# ─────────────────────────────────────────────────────────────────

def plot_summary_table(results: dict, target_col: str, save_path: str = None) -> go.Figure:
    methods   = list(results.keys())
    means     = [f"{results[m].mean:.3f}"     for m in methods]
    stds      = [f"{results[m].std:.3f}"      for m in methods]
    lower_95s = [f"{results[m].lower_95:.3f}" for m in methods]
    upper_95s = [f"{results[m].upper_95:.3f}" for m in methods]
    cvs       = [f"{results[m].cv:.3f}"       for m in methods]
    skews     = [f"{results[m].skewness:.3f}" for m in methods]

    verdicts = []
    for m in methods:
        spread = results[m].upper_95 - results[m].lower_95
        rel    = spread / abs(results[m].mean) if results[m].mean != 0 else 0
        if rel < 0.3:   verdicts.append("✅ Low uncertainty")
        elif rel < 0.6: verdicts.append("⚠️  Moderate uncertainty")
        else:           verdicts.append("🔴 High uncertainty")

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 100, 100, 100, 110, 80, 80, 180],
        header=dict(
            values=["<b>Method</b>","<b>Mean</b>","<b>Std Dev</b>",
                    "<b>Lower 95%</b>","<b>Upper 95%</b>","<b>CV</b>",
                    "<b>Skewness</b>","<b>Verdict</b>"],
            fill_color="#2C3E50",
            font=dict(color="white", size=12),
            align="center", height=35
        ),
        cells=dict(
            values=[methods, means, stds, lower_95s, upper_95s, cvs, skews, verdicts],
            fill_color=[
                ["#D5F5E3","#D6EAF8","#FADBD8"],
                ["white"]*len(methods), ["white"]*len(methods),
                ["white"]*len(methods), ["white"]*len(methods),
                ["white"]*len(methods), ["white"]*len(methods),
                ["#D5F5E3" if "Low" in v else "#FEF9E7" if "Moderate" in v else "#FADBD8" for v in verdicts],
            ],
            font=dict(size=12), align="center", height=32
        )
    )])

    fig.update_layout(
        title=dict(text=f"UQ Summary Table — {target_col}", font=dict(size=15)),
        height=220, margin=dict(t=50, b=20, l=10, r=10)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"[Saved] {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────
# Distribution plot
# ─────────────────────────────────────────────────────────────────

def plot_distribution(data: np.ndarray, target_col: str, save_path: str = None) -> go.Figure:
    from scipy import stats
    mean = float(np.mean(data))
    std  = float(np.std(data, ddof=1))
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))
    x_fit = np.linspace(min(data) - std, max(data) + std, 200)
    y_fit = stats.norm.pdf(x_fit, mean, std)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=10, histnorm="probability density",
        name="Measured Data", marker_color="#3498DB", opacity=0.7))
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
        name="Fitted Normal", line=dict(color="#E74C3C", width=3)))
    fig.add_vline(x=mean, line_color="#2ECC71", line_width=2,
        annotation_text=f"Mean={mean:.3f}", annotation_position="top right")
    fig.add_vline(x=mean - 1.96*std, line_dash="dash", line_color="#F39C12", annotation_text="2.5%")
    fig.add_vline(x=mean + 1.96*std, line_dash="dash", line_color="#F39C12", annotation_text="97.5%")
    fig.update_layout(
        title=dict(text=f"Data Distribution — {target_col}", font=dict(size=15)),
        xaxis_title=target_col, yaxis_title="Probability Density",
        template="plotly_white", height=420,
        annotations=[dict(
            text=f"n={len(data)}  |  Mean={mean:.3f}  |  Std={std:.3f}  |  Skewness={skew:.3f}  |  Kurtosis={kurt:.3f}",
            xref="paper", yref="paper", x=0.5, y=-0.15,
            showarrow=False, font=dict(size=11, color="#7F8C8D"), align="center"
        )]
    )

    if save_path:
        fig.write_html(save_path)
        print(f"[Saved] {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────
# Correlation heatmap
# ─────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, save_path: str = None) -> go.Figure:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix  = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=dict(text=f"Correlation Heatmap — Target: {target_col}", font=dict(size=15)),
        template="plotly_white", height=480,
        xaxis=dict(tickangle=-35)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"[Saved] {save_path}")

    return fig
