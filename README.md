# Materials UQ Engine

**Adaptive Uncertainty Quantification for Materials Science Datasets**

An open-source Python engine that automatically selects and executes the most appropriate uncertainty quantification method for your experimental data — from simple electrochemical measurements to high-dimensional materials models.

Built from real research experience in thin-film deposition, corrosion science, and AI-assisted materials discovery at [N-ERGY AI Solutions](https://www.nergyai.com/).

---

## Why This Exists

In materials science, uncertainty quantification is often an afterthought — researchers either skip it entirely or apply GUM blindly to nonlinear systems where it breaks down. This engine solves that by automatically diagnosing your model and routing it through the right method.

---

## How It Works: Adaptive Method Selection

The dispatcher estimates model nonlinearity using a finite-difference curvature index and selects the appropriate method:

```
Input data + model
        │
        ▼
┌─────────────────────┐
│  Linearity Check     │  ← finite difference curvature score
└─────────────────────┘
        │
   score < 0.05?  ──── YES ──→  GUM (ISO analytical)
        │
   score < 0.30?  ──── YES ──→  Monte Carlo Simulation
        │
   n_inputs > 4?  ──── YES ──→  FSS-PCE (sparse surrogate)
        │
   PCE fails?     ──── YES ──→  QRNN (deep learning fallback)
```

---

## Methods Implemented

| Method | Best For | Key Feature |
|--------|----------|-------------|
| **GUM** | Linear models, fast analysis | ISO/IEC 98-3 compliant, analytical |
| **Monte Carlo** | Nonlinear models | 50k samples, Latin Hypercube option |
| **FSS-PCE** | High-dimensional, expensive models | Forward Selection Sparse PCE, Sobol indices |
| **QRNN** | Noisy/irregular data, fallback | Quantile Regression Neural Network, PyTorch |

Additional sampling strategies available in the core engine:
- **Latin Hypercube Sampling (LHS)** — efficient space-filling MC
- **Delta Method** — analytical Taylor series, zero sampling cost
- **Parametric Bootstrap** — empirical resampling from raw data

---

## Installation

```bash
git clone https://github.com/annegracia/materials-uq-engine.git
cd materials-uq-engine
pip install -r requirements.txt
```

For FSS-PCE support:
```bash
pip install chaospy
```

---

## Quick Start

```python
from src.uq_engine.adaptive_dispatcher import AdaptiveUQDispatcher
import math

# Define your measurement model
def corrosion_model(E_corr, b_a, b_c):
    eta = -0.45 - E_corr
    return abs(math.exp(2.303 * eta / b_a) - math.exp(-2.303 * eta / b_c)) * 10

# Nominal values and uncertainties from your experiment
nominal  = {"E_corr": -0.312, "b_a": 0.082, "b_c": 0.095}
std_uncs = {"E_corr": 0.008,  "b_a": 0.005, "b_c": 0.006}

# Run — method selected automatically
dispatcher = AdaptiveUQDispatcher()
output = dispatcher.run(func=corrosion_model, nominal=nominal, std_uncs=std_uncs)

print(f"Method : {output['method']}")
print(f"Mean   : {output['result']['mean']:.4f} μA/cm²")
print(f"95% CI : {output['result']['lower_95']:.4f} – {output['result']['upper_95']:.4f}")
```

Output:
```
[Dispatcher] Inputs: 3 | Nonlinearity score: 0.2341
[Dispatcher] Selected: MONTE_CARLO
[Dispatcher] Reason  : Moderate nonlinearity (score=0.2341). Monte Carlo selected.

Method : MONTE_CARLO
Mean   : 8.4312 μA/cm²
95% CI : 6.1204 – 11.0847
```

---

## Examples

### Corrosion Science — Butler-Volmer UQ
```bash
python examples/corrosion_uq_example.py
```
UQ on potentiodynamic polarization measurements of WAAM 316L stainless steel in chloride environment. Identifies which electrochemical parameter (E_corr, Tafel slopes) dominates measurement uncertainty.

### Thin Film Deposition — RF Sputtering UQ
```bash
python examples/thin_film_uq_example.py
```
UQ on Al thin film thickness from RF sputtering process parameters (power, time, pressure). Validates whether deposited films fall within the ±10 nm specification for Metal-Induced Crystallization.

---

## Repository Structure

```
materials-uq-engine/
│
├── src/uq_engine/
│   ├── __init__.py
│   ├── adaptive_dispatcher.py   ← core routing logic
│   ├── engine.py                ← SUnCal-backed GUM + MC + LHS engine
│   └── qrnn.py                  ← QRNN deep learning fallback
│
├── examples/
│   ├── corrosion_uq_example.py  ← Butler-Volmer / WAAM 316L
│   └── thin_film_uq_example.py  ← RF sputtering / Al films
│
├── tests/
│   └── test_dispatcher.py
│
├── data/sample/                 ← sample CSV datasets
├── docs/images/                 ← output plots
├── requirements.txt
└── README.md
```

---

## Scientific Background

The FSS-PCE implementation is based on:
> Liu & Choe (2023). *Data-driven sparse polynomial chaos expansion for models with dependent inputs.* Journal of Nuclear and Safety Security Engineering. [DOI: 10.1016/j.jnlssr.2023.08.003](https://doi.org/10.1016/j.jnlssr.2023.08.003)

The adaptive architecture integrates:
- **SUnCal** (Sandia National Labs) for GUM/MC baseline
- **chaospy** for polynomial chaos basis construction
- **PyTorch** QRNN for deep learning uncertainty bounds

---

## Real-World Applications

This engine was developed as part of the materials characterization pipeline at **N-ERGY AI Solutions**, applied to:

- Electrochemical impedance spectroscopy (EIS) data from anodization experiments
- Potentiodynamic polarization curves from WAAM 316L corrosion studies
- Thin film profilometry data from RF sputtering optimization

---

## Author

**Anne Gracia A**  
Chemical Engineer & Materials Researcher  
N-ERGY AI Solutions | SSN College of Engineering, Chennai  
[annegracia.github.io](https://annegracia.github.io) · [LinkedIn](https://linkedin.com/in/anne-gracia-66a53b26a)

---

## License

MIT License — free to use, modify, and build on.
