"""
run_uq.py — Test runner for the UQ engine
==========================================
Loads data from /data, runs UQ on target column,
saves all interactive HTML plots to /output.

Usage:
    python run_uq.py
    python run_uq.py --file data/anodization_TOL.csv --target TOL_um
    python run_uq.py --file data/anodization_TOL.csv --target TOL_um --method GUM
"""

import argparse
import os
import pandas as pd
import numpy as np

from uq_engine import (
    run_uq, run_gum, run_monte_carlo, run_qrnn,
    plot_results, plot_summary_table,
    plot_distribution, plot_correlation_heatmap
)


def main():
    parser = argparse.ArgumentParser(description="Run UQ on a CSV dataset")
    parser.add_argument("--file",   default="data/anodization_TOL.csv")
    parser.add_argument("--target", default="TOL_um")
    parser.add_argument("--method", default="AUTO")
    args = parser.parse_args()

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
        print(f"Available: {list(df.columns)}")
        return

    data = df[args.target].dropna().values
    print(f"\n  Target  : {args.target}")
    print(f"  n       : {len(data)}")
    print(f"  Mean    : {np.mean(data):.4f}")
    print(f"  Std     : {np.std(data, ddof=1):.4f}")
    print(f"  CV      : {np.std(data,ddof=1)/abs(np.mean(data)):.4f}")

    os.makedirs("output", exist_ok=True)

    # ── 1. Run selected method ───────────────────────────────────
    print(f"\n  Running UQ (method={args.method})...")
    output = run_uq(data, method=args.method)
    result = output["result"]
    result.target_column = args.target

    print(f"\n{'─'*60}")
    print(f"  Method     : {result.method}")
    print(f"  Mean       : {result.mean:.4f}")
    print(f"  Std Dev    : {result.std:.4f}")
    print(f"  95% CI     : [{result.lower_95:.4f}, {result.upper_95:.4f}]")
    print(f"  CV         : {result.cv:.4f}")
    print(f"  Skewness   : {result.skewness:.4f}")
    print(f"{'─'*60}")

    method_tag = result.method.lower().replace(" ", "_")
    plot_results(output, save_path=f"output/uq_{args.target}_{method_tag}.html")

    # ── 2. Run all 3 methods + summary table ────────────────────
    print("\n  Running all 3 methods for comparison + summary table...")

    r_gum          = run_gum(data);            r_gum.target_column          = args.target
    r_mc, _        = run_monte_carlo(data);    r_mc.target_column           = args.target
    r_qrnn, _, _   = run_qrnn(data);           r_qrnn.target_column         = args.target

    all_results = {"GUM": r_gum, "Monte Carlo": r_mc, "QRNN": r_qrnn}

    # Summary table
    plot_summary_table(
        all_results, args.target,
        save_path=f"output/uq_{args.target}_summary_table.html"
    )

    # Comparison chart
    import plotly.graph_objects as go
    colors = {"GUM": "#2ECC71", "Monte Carlo": "#3498DB", "QRNN": "#E74C3C"}
    fig = go.Figure()
    for m, r in all_results.items():
        fig.add_trace(go.Scatter(
            x=[m], y=[r.mean],
            error_y=dict(
                type="data", symmetric=False,
                array=[r.upper_95 - r.mean],
                arrayminus=[r.mean - r.lower_95],
                color=colors[m], thickness=3, width=12
            ),
            mode="markers",
            marker=dict(color=colors[m], size=14, symbol="diamond"),
            name=m
        ))
    fig.update_layout(
        title=f"UQ Method Comparison — {args.target}",
        template="plotly_white",
        xaxis_title="Method", yaxis_title=f"{args.target} (μm)",
        height=450,
        annotations=[dict(
            text=" | ".join([
                f"{m}: {r.mean:.3f} [{r.lower_95:.3f}, {r.upper_95:.3f}]"
                for m, r in all_results.items()
            ]),
            xref="paper", yref="paper", x=0.5, y=-0.15,
            showarrow=False, font=dict(size=11, color="#7F8C8D"), align="center"
        )]
    )
    fig.write_html(f"output/uq_{args.target}_comparison.html")
    print(f"[Saved] output/uq_{args.target}_comparison.html")

    # ── 3. Distribution plot ─────────────────────────────────────
    plot_distribution(
        data, args.target,
        save_path=f"output/uq_{args.target}_distribution.html"
    )

    # ── 4. Correlation heatmap ───────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["exp_no"], errors="ignore")
    plot_correlation_heatmap(
        numeric_df, args.target,
        save_path=f"output/uq_{args.target}_correlation.html"
    )

    # ── Done ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ All outputs saved to /output:")
    print(f"   uq_{args.target}_{method_tag}.html       ← main UQ result")
    print(f"   uq_{args.target}_summary_table.html     ← method comparison table")
    print(f"   uq_{args.target}_comparison.html        ← CI comparison chart")
    print(f"   uq_{args.target}_distribution.html      ← data distribution")
    print(f"   uq_{args.target}_correlation.html       ← correlation heatmap")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
