# materials-uq-engine

**Adaptive Uncertainty Quantification for Materials Science Datasets**

An open-source Python tool that automatically selects and runs the right UQ method for your experimental data — outputs interactive HTML plots you can open in any browser.

Built from real research in anodization process optimization, corrosion science, and thin-film deposition.

---

## How It Works

The engine measures the coefficient of variation (CV) of your data and auto-selects:

```
CV < 0.05  →  GUM        (ISO analytical, linear data)
CV < 0.20  →  Monte Carlo (nonlinear, moderate spread)
CV ≥ 0.20  →  QRNN       (deep learning, noisy/skewed data)
```

You can also force any method manually.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample data (auto method selection)
python run_uq.py

# Run with specific file, target column, and method
python run_uq.py --file data/anodization_TOL.csv --target TOL_um --method Monte Carlo
```

Results are saved as interactive HTML files in the `/output` folder.

---

## Methods

| Method | Best For | Key Output |
|--------|----------|------------|
| **GUM** | Linear, low-variance data | Expanded uncertainty U95, coverage factor k |
| **Monte Carlo** | Nonlinear, moderate variance | 50k samples, 2.5–97.5 percentile CI |
| **QRNN** | Noisy, skewed, high-variance data | Neural network quantile bounds [2.5%, 50%, 97.5%] |

All methods output the same structure: mean, std, 95% CI, CV, skewness.

---

## Repository Structure

```
materials-uq-engine/
├── uq_engine.py              ← all UQ methods + Plotly visualization
├── run_uq.py                 ← test runner (CLI)
├── data/
│   └── anodization_TOL.csv   ← oxide layer thickness data (Brakes India)
├── output/                   ← generated HTML plots land here
├── requirements.txt
└── README.md
```

---

## Sample Data

`data/anodization_TOL.csv` contains oxide layer thickness (TOL) measurements from anodization process optimization experiments on automotive aluminium plungers — 20 experimental runs varying temperature, electrolyte concentration, current density, and exposure time.

**Target property:** TOL (μm) — oxide layer thickness  
**CV = 0.36** → QRNN selected automatically

---

## Example Output

Running `python run_uq.py` produces two HTML files:

**1. Single method result** (`output/uq_TOL_um_qrnn.html`)
- Raw measurement scatter plot
- QRNN quantile bounds [2.5%, 50%, 97.5%]
- Statistics summary bar chart

**2. Method comparison** (`output/uq_TOL_um_comparison.html`)
- GUM vs Monte Carlo vs QRNN side by side
- Mean ± 95% CI for each method
- Useful for validating method agreement

---

## Real-World Context

This engine was developed as part of the materials characterization pipeline at **[N-ERGY AI Solutions](https://www.nergyai.com/)**, applied to:
- Oxide layer thickness from industrial anodization (Brakes India Pvt. Ltd.)
- Potentiodynamic polarization data from WAAM 316L corrosion studies
- Thin film profilometry from RF sputtering optimization (SSN College of Engineering)

---

## Author

**Anne Gracia A**  
Chemical Engineer & Materials Researcher  
[annegracia.github.io](https://annegracia.github.io) · [LinkedIn](https://linkedin.com/in/anne-gracia-66a53b26a)
