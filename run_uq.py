"""
run_uq.py — Test runner for the UQ engine
==========================================
Loads data from /data folder, runs UQ on target column,
saves interactive HTML plots to /output folder.

Usage:
    python run_uq.py
    python run_uq.py --file data/anodization_TOL.csv --target TOL_um
    python run_uq.py --file data/anodization_TOL.csv --target TOL_um --method GUM
"""

import argparse
import os
import pandas as pd
import numpy as np
from uq_engine import run_uq, plot_results, auto_select_method


def main():
    parser = argparse.ArgumentParser(description="Run UQ on a CSV dataset")
    parser.add_argument("--file",   default="data/anodization_TOL.csv", help="Path to CSV file")
    parser.add_argument("--target", default="TOL_um",                   help="Target column name")
    parser.add_argument("--method", default="AUTO",                     help="AUTO, GUM, Monte Carlo, or QRNN")
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────
    if not os.path.exists(args.file):
        print(f"[Error] File not found: {args.file}")
        return

    df = pd.read_csv(args.file)
    print(f"\n{'='*60}")
    print(f"  File    : {args.file}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df)}")
    print(f"{'='*60}")

    if args.target not in df.columns:
        print(f"[Error] Column '{args.target}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return

    data = df[args.target].dropna().values
    print(f"\n  Target  : {args.target}")
    print(f"  n       : {len(data)}")
    print(f"  Mean    : {np.mean(data):.4f}")
    print(f"  Std     : {np.std(data, ddof=1):.4f}")
    print(f"  Min/Max : {np.min(data):.4f} / {np.max(data):.4f}")

    # ── Run UQ ──────────────────────────────────────────────────
    print(f"\n  Running UQ (method={args.method})...")
    output = run_uq(data, method=args.method)
    result = output["result"]
    result.target_column = args.target

    print(f"\n{'─'*60}")
    print(f"  Method        : {result.method}")
    print(f"  Mean          : {result.mean:.4f} μm")
    print(f"  Std Dev       : {result.std:.4f} μm")
    print(f"  95% CI        : [{result.lower_95:.4f}, {result.upper_95:.4f}] μm")
    print(f"  CV            : {result.cv:.4f}")
    print(f"  Skewness      : {result.skewness:.4f}")
    print(f"  Notes         : {result.notes}")
    print(f"{'─'*60}\n")

    # ── Save plots ───────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    method_tag = result.method.lower().replace(" ", "_")
    save_path  = f"output/uq_{args.target}_{method_tag}.html"

    fig = plot_results(output, save_path=save_path)

    # Also run all 3 methods and save comparison
    print("\n  Running all 3 methods for comparison...")
    run_all_methods(data, args.target)

    print(f"\n✅ Done! Open your results:")
    print(f"   output/uq_{args.target}_{method_tag}.html")
    print(f"   output/uq_{args.target}_comparison.html\n")


def run_all_methods(data: np.ndarray, target_col: str):
    """Run GUM, MC, and QRNN and save a comparison HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from uq_engine import run_gum, run_monte_carlo, run_qrnn

    results = {}

    # GUM
    r_gum = run_gum(data)
    r_gum.target_column = target_col
    results["GUM"] = r_gum

    # Monte Carlo
    r_mc, _ = run_monte_carlo(data)
    r_mc.target_column = target_col
    results["Monte Carlo"] = r_mc

    # QRNN
    r_qrnn, _, _ = run_qrnn(data)
    r_qrnn.target_column = target_col
    results["QRNN"] = r_qrnn

    # ── Comparison plot ─────────────────────────────────────────
    fig = go.Figure()

    colors = {"GUM": "#2ECC71", "Monte Carlo": "#3498DB", "QRNN": "#E74C3C"}
    x_pos  = {"GUM": 0, "Monte Carlo": 1, "QRNN": 2}

    for method, result in results.items():
        fig.add_trace(go.Scatter(
            x=[method],
            y=[result.mean],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[result.upper_95 - result.mean],
                arrayminus=[result.mean - result.lower_95],
                color=colors[method],
                thickness=3,
                width=12
            ),
            mode="markers",
            marker=dict(color=colors[method], size=14, symbol="diamond"),
            name=method
        ))

    fig.update_layout(
        title=dict(
            text=f"UQ Method Comparison — {target_col}",
            font=dict(size=16)
        ),
        template="plotly_white",
        xaxis_title="UQ Method",
        yaxis_title=f"{target_col} (μm)",
        height=500,
        showlegend=True,
        annotations=[
            dict(
                text=(
                    f"GUM: {results['GUM'].mean:.3f} [{results['GUM'].lower_95:.3f}, {results['GUM'].upper_95:.3f}]  |  "
                    f"MC: {results['Monte Carlo'].mean:.3f} [{results['Monte Carlo'].lower_95:.3f}, {results['Monte Carlo'].upper_95:.3f}]  |  "
                    f"QRNN: {results['QRNN'].mean:.3f} [{results['QRNN'].lower_95:.3f}, {results['QRNN'].upper_95:.3f}]"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=11, color="#7F8C8D"),
                align="center"
            )
        ]
    )

    save_path = f"output/uq_{target_col}_comparison.html"
    fig.write_html(save_path)
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
