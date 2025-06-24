<a name="00-ViDoe-ANOVA-One-Way"></a>
# Using ViDoE to Understand One-Way ANOVA

One-Way ANOVA (Analysis of Variance) is one of the most fundamental tools in experimental design. It is used when we want to compare the means of three or more independent groups (or levels of a factor) to determine if the observed differences in sample means are statistically significant. In practice, this method is key when analyzing how different conditions (treatments, materials, times, etc.) affect a given outcome.

However, interpreting ANOVA results from purely numerical outputs—such as F-values, p-values, and sum of squares—can be challenging for students. That is why **ViDoE** includes a dedicated, interactive module for One-Way ANOVA that helps users understand how treatment effects and residuals contribute to variation in the data.

## Educational Utilities in ViDoE

The ViDoE platform allows learners to:

- Customize the **name of the factor** and its **three levels** (e.g., Temperature: Low, Medium, High).
- Modify the **treatment effects** using sliders, which dynamically update the dataset and visualizations.
- Observe how random noise influences the outcome variable (`Y`).
- Generate and download the experimental dataset in `.csv` format for offline analysis.
- Visualize the results with a **boxplot**, an **ANOVA table**, and a **linear regression summary**.
- See a clear representation of the **regression equation** and the **components of variation** using pie charts.

## Key Figures

### 📊 **Figure 2.1 - Boxplot of Y across Factor Levels**
![Fig2.1 - Boxplot of Y across Levels](/workspaces/DOE/e-book/book/Fig2_1_Boxplot_Y_by_Level.png)

- **Description**: This figure shows the distribution of the response variable `Y` for each factor level. It helps visualize differences in central tendency and variation among groups.

### 📈 **Figure 2.2 - Regression Equation Visualization**
![Fig2.2 - Regression Equation](/workspaces/DOE/e-book/book/Fig2_2_Regression_Equation.png)

- **Description**: Displays the linear regression model derived from the data using indicator variables. This aids in understanding how treatment effects modify the outcome.

### 🧮 **Figure 2.3 - Sum of Squares Decomposition**
![Fig2.3 - Sum of Squares Pie](/workspaces/DOE/e-book/book/Fig2_3_Sum_of_Squares_Pie.png)

- **Description**: Two pie charts showing:
  1. Total Sum of Squares (SST)
  2. Partitioning of SST into Treatment (SSTr) and Error (SSE)

## Suggested Classroom Activity

Students can follow these steps directly within the ViDoE app:

1. Rename the factor (e.g., “Method”) and assign new level names (e.g., “A”, “B”, “C”).
2. Use sliders to assign meaningful treatment effects for Levels B and C.
3. Download the dataset and perform a manual ANOVA in R or Python for comparison.
4. Interpret the results: p-value, F-statistic, and regression coefficients.
5. Visualize the decomposition of variation (SST, SSTr, SSE) using the pie charts.
6. Formulate conclusions: Is the effect of the factor statistically significant?

## Interpretation Support

The module provides both **statistical tables** and **visual guidance** to support interpretation:

- The **ANOVA table** displays degrees of freedom, sum of squares, F-statistic, and p-value.
- The **regression summary** (OLS model) shows estimates and confidence intervals.
- The **LaTeX-style equation** reinforces symbolic thinking.
- The **boxplot and pie charts** simplify abstract variance decomposition.

