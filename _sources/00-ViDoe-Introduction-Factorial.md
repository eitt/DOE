# Exploring Two-Level Factorial Designs with ViDoE

Factorial designs are one of the most efficient and informative experimental strategies when studying the effect of multiple variables. Unlike one-factor-at-a-time approaches, factorial designs explore main effects and interactions between two or more factors simultaneously, allowing for a richer understanding of cause-effect relationships.

ViDoE includes a dynamic module to interactively explore full factorial designs with two factors at two levels each (2×2). This setup enables users to model not only the independent effects of each factor but also the interaction term, which is often critical in real-world experimentation.

## Educational Utilities in ViDoE

The factorial module in ViDoE allows learners to:

- Define custom names for the two factors (e.g., "Temperature" and "Pressure").
- Adjust treatment effects for each factor and the interaction term using sliders.
- Generate datasets with replications and random noise.
- Visualize effects through boxplots and 2D scatter plots.
- Fit linear regression models and visualize the corresponding equations.
- Explore surface plots that illustrate interaction effects.

By interacting with this module, students can build an intuitive understanding of how factors combine to influence an outcome—something often obscured in static, textbook examples.

## Key Figures

### Figure 3.1 - Factor Space
![Fig3.1 - Factor Space](/workspaces/DOE/e-book/book/Fig3_1_Factor_Space.png)

**Description**: A scatter plot showing the combinations of the two factors (each with two levels), with the response `Y` represented by color or marker size. This helps visualize the experimental domain.

### Figure 3.2 - Boxplot of Y Grouped by Factor
![Fig3.2 - Boxplot Y by Factor](/workspaces/DOE/e-book/book/Fig3_2_Boxplot_Y_by_Factor.png)

**Description**: Boxplots of the response variable `Y` grouped by one or both factors, showing main effect trends and hinting at interaction presence.

### Figure 3.3 - Surface Plot of Y
![Fig3.3 - Surface Plot Y](/workspaces/DOE/e-book/book/Fig3_3_Surface_Plot_Y.png)

**Description**: A 3D surface plot representing the response variable `Y` across the full factorial combination space. Useful for visualizing interaction curvature.

### Figure 3.4 - Regression Equation
![Fig3.4 - Regression Equation](/workspaces/DOE/e-book/book/Fig3_4_Regression_Equation.png)

**Description**: Visual representation of the estimated linear regression model with coefficients for each main and interaction effect, based on the simulated data.

## Suggested Classroom Activity

1. Rename the two factors (e.g., "Speed" and "Pressure") and define their levels.
2. Use sliders to assign different effects to Factor A, Factor B, and their interaction.
3. Generate the dataset and download it for analysis.
4. Visualize the factor space to confirm design balance and replication.
5. Create boxplots for each factor and their interaction.
6. Fit a linear model and interpret coefficients—focus on whether the interaction is statistically and practically significant.
7. Present the results using visual support and explain how each factor influences the outcome.

## Interpretation Support

The factorial module in ViDoE helps learners interpret:

- Main effects: how each factor independently affects the outcome.
- Interaction effects: how the effect of one factor depends on the level of another.
- Model fit: R², p-values, and coefficient estimates for model evaluation.
- Graphical patterns: curvature and interaction presence visualized via surface plots.

