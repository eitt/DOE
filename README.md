# **DOE - Interactive Design of Experiments**
🚀 **A illsutrative web-based application for Design of Experiments (DOE) analysis, built with Streamlit.**  
This tool enables users to explore **factorial designs, one-way ANOVA, and regression analysis** interactively, with real-time data visualization and statistical modeling.

## 📌 **Features**
✅ **Factorial Design Analysis**
- **Two-level factorial designs** (2 factors, 2 levels)
- **Three-factor factorial designs** (3 factors, 3 levels)
- **Fully customizable factor names and levels**
- **Interactive data generation with user-defined coefficients**
- **Boxplots, surface plots, and regression models**

✅ **One-Way ANOVA Analysis**
- **Three-level factor design (15 replications per level)**
- **Customizable factor names and level names**
- **Boxplot visualization of factor effects**
- **ANOVA table & linear regression summary**
- **Graphical representation of Sum of Squares (SST, SSTR, SSE)**

✅ **Statistical Outputs**
- **Regression equations displayed in LaTeX format**
- **ANOVA summary table**
- **OLS regression model results**
- **Download generated datasets in CSV format**

✅ **Interactive Data Visualization**
- **3D scatter and surface plots for factorial designs**
- **Boxplots with labeled levels**
- **Pie charts for Sum of Squares decomposition (SST, SSTR, SSE)**

---

## 📂 **Installation**
To run this Streamlit app locally, follow these steps:

### 1️⃣ **Clone this repository**
```bash
git clone https://github.com/yourusername/doe-interactive.git
cd doe-interactive
```

### 2️⃣ **Install required dependencies**
Make sure you have Python **>=3.8** installed, then run:
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
5. **Visualize results** (boxplots, surface plots, regression model).

### **📈 One-Way ANOVA**
1. **Go to the One-Way ANOVA section**.
2. **Enter a custom factor name** (e.g., Material Type).
3. **Rename levels (e.g., Plastic, Metal, Wood)**.
4. **Adjust coefficients to set the effect of each level**.
5. **View the ANOVA table, regression model, and sum of squares decomposition**.

### **📥 Download Data**
- Every dataset generated in the app **can be exported as a CSV file**.


---

## 🚀 **Deployment**
To deploy this app on **Streamlit Cloud**, follow these steps:
1. **Push your repository to GitHub**.
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**.
3. **Create a new app** and connect it to your GitHub repo.
4. **Set the entry point as `mainapp.py`**.
5. **Deploy and share your interactive DOE tool! 🎉**

---

## 📝 **Future Enhancements**
🔹 **More experimental designs** (e.g., fractional factorial, response surface methodology).  
🔹 **Custom data input support** for real-world experiments.  



For any questions, open an **issue** or reach out to us.

---


## 📬 **Contact**
💡 **Developed by:** Leonardo H. Talero-Sarmiento  
🌐 **LinkedIn:** [Leonardo Talero](https://www.linkedin.com/in/leonardo-talero-sarmiento/)  
📧 **Email:** ltalero@unab.edu.co  

---

🎯 **Start exploring the power of Design of Experiments with this interactive tool! 🚀**
