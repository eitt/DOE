# **DOE - Interactive Design of Experiments**
🚀 **An illustrative web-based application for Design of Experiments (DOE) analysis, built with Streamlit.** This tool enables users to explore **factorial designs, fractional factorial designs, one-way ANOVA, conjoint analysis, and regression analysis** interactively, with real-time data visualization and statistical modeling.

## 📌 **Features**
✅ **Factorial Design Analysis**
- **Two-level factorial designs** (2 factors, 2 levels)
- **Three-factor factorial designs** (3 factors, 3 levels)
- **Fully customizable factor names and levels**
- **Interactive data generation with user-defined coefficients**
- **Boxplots, surface plots, and regression models**

✅ **Advanced Designs & Analysis**
- **Fractional Factorial Designs ($2^{k-p}$):** Explore Resolution III, IV, and V designs with 3D cube visualizations and aliasing structures.
- **Choice-Based Conjoint Analysis:** Interactive choice tasks to estimate part-worth utilities using Logistic Regression (Logit).
- **Post-hoc Analysis (Tukey HSD):** Filtered pairwise comparisons and Minitab-style grouping summaries.

✅ **Effect Significance Visualizations**
- **Pareto Plots:** Visually identify the most significant main effects and interactions.
- **Daniel Plots (Normal Plots):** Distinguish real factor impacts from random noise.

✅ **One-Way ANOVA Analysis**
- **Three-level factor design (15 replications per level)**
- **Customizable factor names and level names**
- **Boxplot visualization of factor effects**
- **ANOVA table & linear regression summary**
- **Graphical representation of Sum of Squares (SST, SSTR, SSE)**

✅ **Statistical Outputs & Exports**
- **Regression equations displayed in clean format**
- **ANOVA summary table and OLS regression model results**
- **Download generated datasets and full Tukey HSD reports in CSV format**

---

## 📂 **Installation**
To run this Streamlit app locally, follow these steps:

### 1️⃣ **Clone this repository**
```bash
git clone [https://github.com/yourusername/doe.git](https://github.com/yourusername/doe.git)
cd doe

```

### 2️⃣ **Install required dependencies**

Make sure you have Python **3.10 or 3.11** installed, then run:

```bash
pip install -r requirements.txt

```

### 3️⃣ **Run the app**

```bash
streamlit run mainapp.py

```

---

## 🛠 **How to Use the App**

### **📊 Factorial Designs**

1. **Navigate to the Factorial Design section**.
2. **Define factor names** (e.g., Temperature, Pressure, Thinner).
3. **Set levels (e.g., Low, Medium, High)**.
4. **Adjust coefficients to explore different effects**.
5. **Visualize results** (boxplots, surface plots, Pareto/Daniel plots, regression model).

### **📈 One-Way ANOVA & Post-Hoc**

1. **Go to the One-Way ANOVA section**.
2. **Enter a custom factor name** (e.g., Material Type).
3. **Rename levels (e.g., Plastic, Metal, Wood)**.
4. **Adjust coefficients to set the effect of each level**.
5. **View the ANOVA table, regression model, and Tukey HSD groupings**.

### **📉 Fractional & Conjoint**

* **Fractional Factorial:** Select your design (, , etc.) to view the required runs, the aliasing structure, and the 3D factor space.
* **Conjoint Analysis:** Complete the interactive A/B choices to generate a dataset, then view the Logistic Regression odds ratios.

### **📥 Download Data**

* Every dataset generated in the app, as well as full Post-Hoc comparison tables, **can be exported as CSV files**.

---

## 🚀 **Deployment on Streamlit Cloud**

To deploy this app smoothly on **Streamlit Cloud**, follow these steps:

1. **Keep `requirements.txt` lightweight and bounded:** include only required runtime packages and use version ranges (for example `streamlit>=1.33,<2`) instead of hard pins for every package. This reduces dependency-resolution failures and long install times on Streamlit Cloud.
2. **Push your repository to GitHub**.
3. **Go to [Streamlit Cloud](https://share.streamlit.io/).**
4. **Create a new app** and connect it to your GitHub repo.
5. **Set the entry point as `mainapp.py`.**
6. **Deploy and share your interactive DOE tool! 🎉**

### ✅ Streamlit deployment stability tips (to avoid install failures)

- Use a supported Python runtime (`python-3.11`) in `runtime.txt`.
- Avoid unnecessary heavy packages (for this app, `matplotlib` is not required).
- If deployment starts failing after package updates, clear app cache and redeploy.
- Keep package bounds compatible (example: `statsmodels>=0.14,<1` with `numpy>=1.24,<3`).

---

## 📝 **Future Enhancements**

🔹 **Response surface methodology (RSM)** optimizations.

🔹 **Custom data input support** for analyzing real-world user experiments.

For any questions, open an **issue** or reach out to us.

---

## 📬 **Contact**

💡 **Developed by:** Leonardo H. Talero-Sarmiento

🌐 **Profile:** [View Apolo Profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)

---

🎯 **Start exploring the power of Design of Experiments with this interactive tool! 🚀**

```
