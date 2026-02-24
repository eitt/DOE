# DOE – Interactive Design of Experiments

An illustrative web-based application for Design of Experiments (DOE) analysis built with Streamlit. The application enables interactive exploration of factorial designs, fractional factorial designs, one-way ANOVA, conjoint analysis, and regression modeling, with real-time visualization and statistical output.

---

## Overview

This project provides an educational and exploratory environment for understanding experimental design and statistical inference. Users can define factors, adjust model coefficients, generate synthetic data, and immediately inspect statistical and graphical outputs.

---

## Features

### Factorial Design Analysis

* Two-level factorial designs (2 factors, 2 levels)
* Three-factor factorial designs (3 factors, 3 levels)
* Customizable factor and level names
* Interactive data generation with user-defined coefficients
* Boxplots, surface plots, and regression modeling

### Advanced Designs and Analysis

* Fractional factorial designs (2^(k−p)):

  * Resolution III, IV, and V designs
  * 3D cube visualizations
  * Aliasing structure inspection

* Choice-based conjoint analysis:

  * Interactive choice tasks
  * Part-worth utility estimation using logistic regression

* Post-hoc analysis (Tukey HSD):

  * Pairwise comparisons
  * Grouping summaries in Minitab-style format

### Effect Significance Visualizations

* Pareto plots for identifying dominant main and interaction effects
* Daniel (normal) plots for distinguishing active effects from noise

### One-Way ANOVA

* Three-level factor design (15 replications per level)
* Customizable factor and level names
* Boxplot visualization of group effects
* ANOVA table and linear regression summary
* Graphical representation of SST, SSTR, and SSE

### Statistical Outputs and Export

* Regression equations in readable mathematical form
* ANOVA summary tables and OLS model results
* CSV export for generated datasets and Tukey HSD reports

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/doe.git
cd doe
```

### 2. Install dependencies

Use Python 3.10 or 3.11.

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run mainapp.py
```

---

## Usage Guide

### Factorial Designs

1. Navigate to the **Factorial Design** section.
2. Define factor names (e.g., Temperature, Pressure, Thinner).
3. Specify factor levels (e.g., Low, Medium, High).
4. Adjust model coefficients to simulate different effect structures.
5. Inspect visual outputs (boxplots, surface plots, Pareto plots, Daniel plots) and regression summaries.

### One-Way ANOVA and Post-Hoc Analysis

1. Open the **One-Way ANOVA** section.
2. Define the factor name (e.g., Material Type).
3. Rename levels (e.g., Plastic, Metal, Wood).
4. Set level coefficients to control effect magnitude.
5. Review the ANOVA table, regression output, and Tukey HSD groupings.

### Fractional Factorial and Conjoint Analysis

* Fractional factorial:
  Select the design configuration to review required runs, aliasing structure, and 3D factor space representation.

* Conjoint analysis:
  Complete the interactive choice tasks. The application generates a dataset and estimates logistic regression coefficients with corresponding odds ratios.

### Data Export

All generated datasets and post-hoc comparison tables can be exported as CSV files directly from the interface.

### Streamlit deployment stability tips (to avoid install failures)

- Use a supported Python runtime (`python-3.11`) in `runtime.txt`.
- Avoid unnecessary heavy packages (for this app, `matplotlib` is not required).
- If deployment starts failing after package updates, clear app cache and redeploy.
- Keep package bounds compatible (example: `statsmodels>=0.14,<1` with `numpy>=1.24,<3`).

---

## Deployment on Streamlit Cloud

To deploy on Streamlit Cloud:

1. Keep `requirements.txt` minimal and use bounded version ranges (e.g., `streamlit>=1.33,<2`).
2. Push the repository to GitHub.
3. Create a new application in Streamlit Cloud.
4. Set the entry point to `mainapp.py`.
5. Deploy the application.

### Deployment Stability Recommendations

* Use a supported runtime (e.g., `python-3.11` in `runtime.txt`).
* Avoid unnecessary heavy dependencies.
* Clear cache and redeploy if dependency conflicts arise.
* Maintain compatible version ranges (e.g., `statsmodels>=0.14,<1` with `numpy>=1.24,<3`).

---

## Future Enhancements

* Response Surface Methodology (RSM) extensions.
* Support for user-uploaded experimental datasets.

---

## Contact

Developed by Leonardo H. Talero-Sarmiento

Profile: [https://apolo.unab.edu.co/en/persons/leonardo-talero](https://apolo.unab.edu.co/en/persons/leonardo-talero)

