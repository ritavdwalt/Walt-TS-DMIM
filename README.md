# Walt TS-DMIM: van der Walt's Time-Series Delta Moment-Independent Measure

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A highly optimized, object-oriented Python framework for executing Global Sensitivity Analysis (GSA) across complex, asynchronous time-series data. 

**Walt TS-DMIM** adapts the density-based Delta Moment-Independent Measure (DMIM) for temporal data, allowing researchers to extract true physical causality, identify time-lags, and filter out statistical noise in dynamic systems (like building thermodynamics and Industrial IoT networks).

---

## 1. The "Why": Solving Time-Series GSA

Standard sensitivity analysis methods operate on static, synchronous snapshots of data. When applied to time-series data, they struggle with temporal delays, intermittent sensor dropouts, and autocorrelation (where variables are correlated with their own past states). 

**This framework introduces three core innovations:**
1. **$O(1)$ Relational Time-Shifting:** Bypasses computationally heavy `datetime` manipulations by flattening time into continuous integer steps, allowing for massive-scale historical lag sweeping.
2. **Variance Preservation:** Natively handles duplicate sensor readings and missing data without artificially averaging or interpolating the target variable, preserving the true thermodynamic density distributions required for DMIM.
3. **Topological Diagnostics:** Automatically extracts the "Causal Basin" of an event, separating genuine physical influence from background statistical noise.

---

## 2. Installation

Install the package directly from GitHub. Because this library relies on the latest density-estimation algorithms, it automatically installs the unreleased `main` branch of the SALib engine.

```bash
pip install git+[https://github.com/ritavdwalt/Walt-TS-DMIM.git](https://github.com/ritavdwalt/Walt-TS-DMIM.git)
```

**Core Dependencies:**
* `pandas` >= 1.3.0
* `numpy` >= 1.20.0
* `scipy` >= 1.7.0
* `matplotlib` >= 3.4.0
* `SALib` (Sourced directly from GitHub main)

---

## 3. Crucial Data Formatting Rules

To utilize the ultra-fast relational `.join()` engine, you **must** format your inputs according to these strict rules before passing them to the `TemporalAnalyzer`.

### Rule 1: The Continuous Integer Timeline (Weather / Inputs)
Your input data (e.g., weather variables) cannot use standard Pandas `datetime` indexes. It must be mapped to a strictly continuous integer column named `timestep`. 
* **No missing steps:** If a sensor failed for an hour, the integer step must still exist in the dataframe (fill the data values with `0` or `NaN`).
* **No duplicates:** The input timeline represents the objective passage of time; it must be unique.

### Rule 2: The Asynchronous Target Data (Indoor / Outputs)
Your target dataset (e.g., indoor room temperature) must also use the matching integer `timestep` column, but **it is highly flexible:**
* **Duplicates are allowed:** If three sensors fired at timestep `50`, leave all three rows in the dataset. The engine will map the single weather state to all three independent readings simultaneously.
* **Gaps are allowed:** If your target sensors went offline, simply omit those rows. The engine only computes intersections where target data physically exists.

**Example Format:**

| `timestep` | Input: Air Temp | Input: Solar | Target: Room Temp | Note |
|:---:|:---:|:---:|:---:|:---|
| `1` | 20.5 | 800 | 22.1 | Standard reading |
| `2` | 20.6 | 810 | *Missing Row* | Target sensor offline (Allowed) |
| `3` | 20.6 | 805 | 22.4 | Standard reading |
| `3` | 20.6 | 805 | 22.8 | Duplicate target reading (Allowed) |

---

## 4. Understanding the Diagnostic Profile

The framework outputs high-resolution diagnostic grids. Instead of forcing you to guess which peaks are real, the engine automatically extracts topological metrics to define the physical boundaries of the relationship.

* **Peak Lag:** The exact index step where the input variable exerts its maximum thermodynamic influence on the target. *(Multiply this by your logging interval to get the true physical time delay).*
* **The Causal Basin:** The mathematical "Window of Influence." It identifies exactly when the variable begins to measurably affect the system, and exactly when its influence dissipates back into the baseline noise floor.
* **Cumulative AUC (Area Under Curve):** The total integrated energy of the influence across the entire Causal Basin. 
* **Prominence Ratio (The Confidence Score):** The signal-to-noise ratio of the extracted peak. The engine uses a baseline threshold of $\approx 1.15$. If a peak falls below this ratio, the library will flag it as a false positive, indicating that the sensitivity is practically indistinguishable from the background variance of the dataset.

---

## Quickstart Tutorial
For a complete, copy-pasteable pipeline—including synthetic data generation and visual plotting—please refer to the interactive Jupyter Notebook located in `examples/tutorial.ipynb`.