import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import streamlit as st
from itertools import combinations

# Define numeric mappings
FACTOR_VALUES = {'low': -1, 'medium': 0, 'high': 1}
TWO_LEVEL_MAPPING = {'low': -1, 'high': 1}

def create_factorial_dataframe(levels, numeric_mapping, replications=2):
    """
    Generates a factorial design dataframe with numeric mappings.

    Args:
        levels (dict): Dictionary where keys are factor names and values are lists of levels.
        numeric_mapping (dict): Mapping from factor levels to numeric values.
        replications (int): Number of replications.

    Returns:
        pd.DataFrame: Dataframe containing the factorial design with numeric mappings.
    """
    df = pd.DataFrame(list(itertools.product(*levels.values())) * replications, columns=levels.keys())

    # Apply numeric mapping dynamically
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(numeric_mapping)

    return df

def compute_response(df, coefficients, factor_names):
    """
    Computes response variable Y dynamically for any factorial design.

    Args:
        df (pd.DataFrame): Input dataframe.
        coefficients (list): Model coefficients.
        factor_names (dict): Mapping of factor names.

    Returns:
        pd.DataFrame: Dataframe with computed Y values.
    """
    noise = np.random.normal(0, 1, len(df))

    # Ensure factor columns exist before renaming
    rename_mapping = {f"{factor}_num": f"{factor_names[factor]}_num" for factor in factor_names if f"{factor}_num" in df.columns}
    df = df.rename(columns=rename_mapping)

    # Compute main effects
    y = coefficients[0] + sum(
        coefficients[i] * df[f'{factor_names[factor]}_num']
        for i, factor in enumerate(factor_names, start=1)
    )

    # Compute interaction effects
    interaction_index = len(factor_names) + 1
    for i, (f1, f2) in enumerate(combinations(factor_names, 2)):
        y += coefficients[interaction_index + i] * df[f'{factor_names[f1]}_num'] * df[f'{factor_names[f2]}_num']

    df['Y'] = y + noise
    return df

def fit_factorial_model(df, factor_names):
    """
    Fits an OLS model for factorial designs dynamically.

    Args:
        df (pd.DataFrame): Input dataframe.
        factor_names (dict): Mapping of factor names.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Fitted model results.
    """
    factors = list(factor_names.keys())

    # Construct formula dynamically
    formula = "Y ~ " + " + ".join([f"Q('{factor_names[f]}_num')" for f in factors]) + \
              " + " + " + ".join([f"Q('{factor_names[f1]}_num'):Q('{factor_names[f2]}_num')"
                                  for f1, f2 in combinations(factors, 2)])

    return smf.ols(formula, data=df).fit()

def print_equation(results, factor_names):
    """
    Constructs and returns a formatted regression equation from model results.
    """
    coefs = results.params
    terms = ["Intercept"] + [f"{factor_names[f]}_num" for f in factor_names] + \
            [f"{factor_names[f1]}_num:{factor_names[f2]}_num" for f1, f2 in combinations(factor_names, 2)]
    return "Y = " + " + ".join(f"{coefs[i]:.2f} * {term}" for i, term in enumerate(terms))

def plot_2d(df, factor_names):
    """
    Generates a 2D scatter plot for a two-level factorial design.
    """
    fig = go.Figure(data=go.Scatter(
        x=df[f'{factor_names["FactorA"]}_num'], 
        y=df[f'{factor_names["FactorB"]}_num'], 
        mode='markers', 
        marker=dict(
            size=8,
            color=df['Y'],  
            colorscale='Viridis',  
            colorbar=dict(title='Y')
        )
    ))

    fig.update_layout(
        xaxis_title=factor_names["FactorA"],
        yaxis_title=factor_names["FactorB"],
        title='2D Plot for Two-Level Factorial Design'
    )
    st.plotly_chart(fig)

def plot_3d(df, factor_names):
    """
    Generates a 3D scatter plot of the factorial design.
    """
    fig = go.Figure(data=go.Scatter3d(
        x=df[f'{factor_names["Temperature"]}_num'], 
        y=df[f'{factor_names["Pressure"]}_num'], 
        z=df[f'{factor_names["Thinner"]}_num'],
        mode='markers',
        marker=dict(size=10, color=df['Y'], colorbar=dict(title='Y'), opacity=0.8)
    ))

    fig.update_layout(scene=dict(
        xaxis_title=factor_names["Temperature"],
        yaxis_title=factor_names["Pressure"],
        zaxis_title=factor_names["Thinner"]
    ))
    
    st.plotly_chart(fig)

def plot_surface(df, factor1, factor2):
    """
    Generates a 3D surface plot for factorial design.
    """
    pivot_table = df.pivot_table(values='Y', index=f'{factor1}_num', columns=f'{factor2}_num')

    fig = go.Figure(data=[go.Surface(z=pivot_table.values, x=pivot_table.index, y=pivot_table.columns)])

    fig.update_layout(scene=dict(
        xaxis_title=factor1,
        yaxis_title=factor2,
        zaxis_title='Y'
    ))
    
    st.plotly_chart(fig)

def plot_surface_twolevels(df, factor_names):
    """
    Generates a 3D surface plot for a two-level factorial design.
    """
    pivot_df = df.pivot_table(values='Y', index=f"{factor_names['FactorA']}_num", columns=f"{factor_names['FactorB']}_num")

    fig = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.index, y=pivot_df.columns)])

    fig.update_layout(
        title='Surface Plot for Two-Level Factorial Design',
        width=800, height=500,
        scene=dict(
            xaxis_title=factor_names['FactorA'],
            yaxis_title=factor_names['FactorB'],
            zaxis_title='Y'
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )

    st.plotly_chart(fig)

def plot_boxplot(df, groupby, factor_names):
    """
    Generates a boxplot grouped by a factor or an interaction term.
    """
    fig = go.Figure()

    # Handle interaction term dynamically
    if "*" in groupby:
        factors = groupby.split(" * ")

        # Ensure factors exist in factor_names before using them
        if factors[0] not in factor_names.values() or factors[1] not in factor_names.values():
            st.error(f"Error: One of the selected factors '{factors[0]}' or '{factors[1]}' does not exist.")
            return

        # Find corresponding column names in df
        col1 = [key for key, value in factor_names.items() if value == factors[0]]
        col2 = [key for key, value in factor_names.items() if value == factors[1]]

        if not col1 or not col2:
            st.error(f"Error: Could not find corresponding columns for '{factors[0]}' and '{factors[1]}' in DataFrame.")
            return
        
        col1 = f'{factor_names[col1[0]]}_num'
        col2 = f'{factor_names[col2[0]]}_num'

        if col1 not in df.columns or col2 not in df.columns:
            st.error(f"Error: Columns '{col1}' or '{col2}' not found in DataFrame.")
            return

        # Create interaction column dynamically
        df['Interaction'] = df[col1].astype(str) + " * " + df[col2].astype(str)
        groupby = 'Interaction'
    else:
        # Ensure correct factor name mapping to `_num` columns
        groupby_mapped = {v: f"{v}_num" for v in factor_names.values()}
        groupby = groupby_mapped.get(groupby, groupby)

    # Verify that `groupby` exists before running `.groupby()`
    if groupby not in df.columns:
        st.error(f"Error: The selected column '{groupby}' does not exist in the dataset.")
        return

    # Group and plot the data
    for name, group in df.groupby(groupby):
        fig.add_trace(go.Box(y=group['Y'], name=str(name), boxmean=True))

    fig.update_layout(xaxis_title=groupby, yaxis_title='Y')
    st.plotly_chart(fig)


def three_factorial():
    """
    Streamlit app for three-factor factorial design visualization and analysis.
    """
    st.title('Factorial Design Visualization')
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    # User-defined custom factor names
    factor_names = {
        "Temperature": st.sidebar.text_input("Enter a custom name for Temperature", "Temperature"),
        "Pressure": st.sidebar.text_input("Enter a custom name for Pressure", "Pressure"),
        "Thinner": st.sidebar.text_input("Enter a custom name for Thinner", "Thinner"),
    }

    # Coefficients Input
    coefficients = [
        st.sidebar.slider('Intercept (Coefficient 0)', -100.0, 100.0, 0.0)
    ] + [
        st.sidebar.slider(f'Coefficient {i+1} ({factor_names[factor]})', -100.0, 100.0, 0.0)
        for i, factor in enumerate(["Temperature", "Pressure", "Thinner"])
    ] + [
        st.sidebar.slider(f'Coefficient {i+4} ({factor_names[f1]} * {factor_names[f2]})', -100.0, 100.0, 0.0)
        for i, (f1, f2) in enumerate(combinations(["Temperature", "Pressure", "Thinner"], 2))
    ]

    # Define factor levels
    levels = {factor: ['low', 'medium', 'high'] for factor in factor_names.keys()}

    # Create DataFrame
    df = create_factorial_dataframe(levels, FACTOR_VALUES, replications=2)

    # Ensure correct column renaming before computation
    rename_mapping = {f"{factor}_num": f"{factor_names[factor]}_num" for factor in factor_names}
    df.rename(columns=rename_mapping, inplace=True)

    # Compute Y
    df = compute_response(df, coefficients, factor_names)

    # Display DataFrame
    st.subheader('Generated Data')
    st.dataframe(df)

    # CSV Download Button
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="data.csv", mime="text/csv")

    # 3D Scatter Plot
    st.subheader('Factors Space')
    plot_3d(df, factor_names)

    # Boxplot Analysis
    st.subheader('Analysis of Y based on Variability Source')
    st.subheader('Box Plot')

    # 🔥 **Fix: Ensure interaction terms are correctly mapped**
    original_factors = list(factor_names.keys())  # Extract original factor names
    renamed_factors = list(factor_names.values())  # Extract renamed factor names

    # Create `groupby_options` using renamed factor names
    groupby_options = renamed_factors + [
        f"{factor_names[f1]} * {factor_names[f2]}" for f1, f2 in combinations(original_factors, 2)
    ]

    groupby = st.selectbox('Group by', groupby_options, index=0)

    # Ensure correct column selection
    plot_boxplot(df, groupby, factor_names)

    # Surface Plot Selection
    st.subheader('Surface Plot')
    factor1 = st.selectbox('Select First Factor', renamed_factors, index=0)
    factor2 = st.selectbox('Select Second Factor', renamed_factors, index=1)
    
    # Ensure valid surface plot selection
    if factor1 != factor2:
        plot_surface(df, factor1, factor2)
    else:
        st.error("Please select two different factors for the surface plot.")

    # Model Fitting
    st.subheader('Model Fitting')
    results = fit_factorial_model(df, factor_names)
    st.latex(print_equation(results, factor_names))
    st.text(results.summary())


def factorial_twolevels():
    """
    Streamlit app to interactively explore two-factor, two-level factorial designs.
    """
    st.title("Introduction to Factorial Designs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("A factorial design is an experimental design that studies the effects of two or more factors, each with multiple levels, on a dependent variable. In a factorial design, the factors are manipulated independently of each other, and the levels of each factor are crossed with the levels of the other factors. This allows researchers to study the main effects of each factor, as well as the interactions between the factors.")
    st.markdown("Factorial designs are more efficient than one-factor-at-a-time designs, because they allow researchers to study the effects of multiple factors with the same number of participants. Additionally, factorial designs can help researchers to identify interactions between factors, which can be important for understanding the effects of the factors on the dependent variable.")
    st.markdown("There are two main types of factorial designs: full factorial designs and fractional factorial designs. Full factorial designs include all possible combinations of levels for each factor. Fractional factorial designs are a subset of full factorial designs that include only a subset of the possible combinations of levels. Fractional factorial designs are less efficient than full factorial designs, but they can be used when resources are limited.")
    
    st.subheader('Two Factors and Two Levels')

    # Define two-level factor values
    two_level_factor_values = {'FactorA': ['low', 'high'], 'FactorB': ['low', 'high']}

    # Custom Factor Names from Sidebar
    two_level_factor_names = {
        "FactorA": st.sidebar.text_input("Enter a custom name for Factor A", "FactorA"),
        "FactorB": st.sidebar.text_input("Enter a custom name for Factor B", "FactorB")
    }

    # Coefficients Input
    coefficients = [
        st.sidebar.slider('Intercept (Coefficient 0)', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'{two_level_factor_names["FactorA"]} Coefficient', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'{two_level_factor_names["FactorB"]} Coefficient', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'Interaction ({two_level_factor_names["FactorA"]} * {two_level_factor_names["FactorB"]})', -100.0, 100.0, 0.0)
    ]

    # Create DataFrame
    df = create_factorial_dataframe(two_level_factor_values, TWO_LEVEL_MAPPING, replications=3)

    # Ensure correct column renaming before computation
    rename_mapping = {f"{factor}_num": f"{two_level_factor_names[factor]}_num" for factor in two_level_factor_values}
    df.rename(columns=rename_mapping, inplace=True)

    # Compute Y
    df = compute_response(df, coefficients, two_level_factor_names)

    # Display DataFrame
    st.subheader('Generated Data')
    st.dataframe(df)

    # CSV Download Button
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="data.csv", mime="text/csv")

    # 2D Scatter Plot
    st.subheader('Factors Space')
    plot_2d(df, two_level_factor_names)

    # Boxplot Analysis
    st.subheader('Analysis of Y based on Variability Source')
    st.subheader('Box Plot')

    # Dynamically Generate Grouping Options for Boxplots
    groupby_options = list(two_level_factor_names.values()) + [f"{two_level_factor_names['FactorA']} * {two_level_factor_names['FactorB']}"]
    groupby = st.selectbox('Group by', groupby_options, index=0)

    # Ensure correct mapping before passing to plot_boxplot
    plot_boxplot(df, groupby, two_level_factor_names)

    # Surface Plot Selection
    st.subheader('Surface Plot')
    plot_surface_twolevels(df, two_level_factor_names)

    # Model Fitting
    st.subheader('Model Fitting')
    results = fit_factorial_model(df, two_level_factor_names)
    st.latex(print_equation(results, two_level_factor_names))
    st.text(results.summary())

import statsmodels.api as sm
import statsmodels.formula.api as smf

def anova_oneway():
    """
    Streamlit app to demonstrate One-Way ANOVA with one factor, three levels, and 15 replications per level.
    """
    st.title("One-Way ANOVA: Analysis of Variance with One Factor")
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("""
    One-Way ANOVA is a statistical method used to compare the means of three or more independent groups 
    to determine if there is a statistically significant difference between them. It helps to test 
    whether the population means of the different levels of a factor are equal.

    ### **Key Features:**
    - **Factor with Three Levels**
    - **15 Replications per Level**
    - **Boxplot for Visualization**
    - **Linear Regression for Statistical Analysis**
    - **Dataset Download Option**
    """)

    st.subheader('One Factor with Three Levels')

    # Custom Factor Name from Sidebar
    factor_name = st.sidebar.text_input("Enter a custom name for the factor", "Factor")

    # Custom Level Names from Sidebar
    level_names = {
        "low": st.sidebar.text_input("Enter a custom name for Level 1", "Low"),
        "medium": st.sidebar.text_input("Enter a custom name for Level 2", "Medium"),
        "high": st.sidebar.text_input("Enter a custom name for Level 3", "High")
    }

    # Coefficients Input
    coefficients = [
        st.sidebar.slider('Intercept (Baseline Level)', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Effect of {level_names["medium"]}', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Effect of {level_names["high"]}', -10.0, 10.0, 0.0)
    ]

    # Define factor levels and replications
    levels = ["low", "medium", "high"]
    replications = 15  # Each level appears 15 times

    # Create DataFrame
    df = pd.DataFrame({
        factor_name: np.repeat(levels, replications)  # Repeat each level 15 times
    })

    # Assign numeric values to factor levels
    level_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df[f"{factor_name}_num"] = df[factor_name].map(level_mapping)

    # Rename levels to user-defined names
    df[factor_name] = df[factor_name].map(level_names)

    # Compute Response Variable (Y)
    noise = np.random.normal(0, 2, len(df))  # Normally distributed random noise
    df["Y"] = (coefficients[0]  # Intercept
               + coefficients[1] * (df[factor_name] == level_names["medium"])  # Effect of Medium
               + coefficients[2] * (df[factor_name] == level_names["high"])  # Effect of High
               + noise)

    # Display DataFrame
    st.subheader('Generated Data')
    st.dataframe(df)

    # CSV Download Button
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="anova_data.csv", mime="text/csv")

    # Boxplot Analysis
    st.subheader('Box Plot for Factor Levels')
    fig = go.Figure()
    for level in df[factor_name].unique():
        fig.add_trace(go.Box(y=df[df[factor_name] == level]["Y"], name=level, boxmean=True))

    fig.update_layout(xaxis_title=factor_name, yaxis_title="Y", title="Distribution of Y across Levels")
    st.plotly_chart(fig)

    # One-Way ANOVA using Linear Regression
    st.subheader('Statistical Analysis: One-Way ANOVA')

    formula = f"Y ~ C({factor_name})"
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Display ANOVA table
    st.write("### **ANOVA Table:**")
    st.write(anova_table)

    # Regression Summary
    st.write("### **Linear Regression Model Summary:**")
    st.text(model.summary())


def Analysis():
    """
    Tips to Analyze the Statistical Outputs.
    """
    st.title("Tips to Analyze the Statistical Outputs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.subheader('Analysis of Y Based on a Source of Variability')

    st.subheader('Box Plot')
    
    st.markdown("### *Why Are Box Plots by Group So Useful in Data Analysis?*")
    st.markdown("""
    Creating a box plot for different groups is a powerful way to visually explore and compare data distributions across categories. This simple yet effective visualization can reveal patterns, differences, and anomalies that might be hidden in raw numbers. Here’s why box plots are incredibly useful and what insights they can provide:
    """)
    
    st.markdown("## 1. Comparing Distributions Across Groups")
    st.markdown("""
    Box plots allow us to compare the spread and central tendency of data across different categories at a glance.
    
    👉 **Example:** Imagine you're analyzing **student test scores** across multiple schools. A box plot can quickly show whether some schools have consistently higher scores, if others have a wider range of performance, or if certain schools have unusually low or high scores compared to the rest.
    
    👉 **Another example:** If you're studying **salaries across different industries**, a box plot can instantly highlight whether some fields (like tech) have a broader pay range while others (like education) have more uniform salaries.
    """)
    
    st.markdown("## 2. Spotting Outliers")
    st.markdown("""
    Outliers—data points that fall far outside the typical range—are clearly visible in box plots as dots beyond the "whiskers." These can signal interesting insights or potential data issues.
    
    👉 **Example:** If you're analyzing **delivery times** for different shipping companies, and one company consistently has extreme delays (represented by outliers), this could indicate operational inefficiencies.
    
    👉 **Another example:** If you're studying **home prices** in different neighborhoods, outliers might reveal luxury properties or undervalued homes that merit deeper investigation.
    """)
    
    st.markdown("## 3. Understanding Skewness and Symmetry")
    st.markdown("""
    A box plot isn’t just about the median (the middle value); it also shows how data is distributed. If the median is off-center within the box, it indicates skewness.
    
    👉 **Example:** Suppose you're analyzing **commute times** for employees in different cities. If the median commute time in one city is much closer to the lower quartile, it suggests that most people have short commutes, but a few endure very long ones—perhaps due to traffic congestion or urban sprawl.
    
    Recognizing skewness helps in making fair comparisons and drawing better conclusions.
    """)
    
    st.markdown("## 4. Detecting Shifts and Trends Over Time")
    st.markdown("""
    If you compare box plots over different time periods, you can see how things change.
    
    👉 **Example:** A retail business tracking **monthly customer spending** can use box plots to reveal seasonal trends—perhaps spending spikes in **December** (holiday shopping) and drops in **January**.
    
    👉 **Another example:** If you're analyzing **stock market performance**, box plots can show how price volatility changes from one quarter to the next.
    """)
    
    st.markdown("## 5. Exploring Relationships Between Variables")
    st.markdown("""
    Box plots help uncover connections between categorical and continuous variables.
    
    👉 **Example:** Suppose you're studying the impact of **different diet plans on weight loss**. By grouping individuals based on their diet type and plotting their weight loss results, a box plot could quickly reveal which diets tend to lead to the most consistent or extreme results.
    
    👉 **Another example:** In a **manufacturing setting**, box plots can help compare **defect rates across production shifts**, identifying if a certain shift tends to produce more defective items than others.
    """)
    
    st.subheader('# How to Read an OLS Regression Output')

    st.markdown("""
    When running an **Ordinary Least Squares (OLS) regression**, the output provides key statistics to evaluate the model’s performance and understand the relationship between variables. Below is a breakdown of the main components and what they mean.
    """)
    
    st.markdown("## 1. Model Summary")
    st.markdown("""
    This section gives an overview of the regression model:
    
    - **Dep. Variable:** The dependent variable (Y) being predicted.
    - **Model:** Specifies the regression type (OLS - Ordinary Least Squares).
    - **Method:** Indicates how the model estimates coefficients (Least Squares).
    - **No. Observations:** The number of data points used in the model.
    - **Df Model:** The degrees of freedom for the predictors (number of independent variables).
    - **Df Residuals:** The degrees of freedom remaining after estimating the model.
    - **Covariance Type:** Describes how standard errors are calculated (e.g., "nonrobust" or "robust").
    """)
    
    st.markdown("## 2. Model Fit Statistics")
    st.markdown("""
    These metrics indicate how well the model explains the variability in the dependent variable:
    
    - **R-squared (R²):** Measures how much variation in Y is explained by the model. Ranges from 0 to 1, with higher values indicating a better fit.
    - **Adjusted R-squared:** Similar to R² but adjusted for the number of predictors. Useful when comparing models with different numbers of variables.
    - **F-statistic:** Tests whether at least one predictor is statistically significant. A higher value suggests a more meaningful model.
    - **Prob (F-statistic):** The p-value for the F-test. A low value (e.g., < 0.05) suggests the model is statistically significant.
    - **Log-Likelihood:** Measures model fit; higher values indicate a better model.
    - **AIC / BIC:** Model selection criteria (lower values indicate a better fit with fewer parameters).
    """)
    
    st.markdown("## 3. Understanding R-Squared (R²)")
    st.markdown("""
    - **What it tells you:** R² represents the percentage of variation in the dependent variable explained by the independent variables.
    - **High R²:** The model explains most of the variability in Y.
    - **Low R²:** The model does not explain much variation, meaning the predictors may not be strong enough.
    
    🔍 **Important:** A **high R² does not always mean a good model**—overfitting can occur if too many variables are used.
    """)
    
    st.markdown("## 4. Checking Model Significance")
    st.markdown("""
    - **F-statistic & Prob (F-statistic):**  
      - If the p-value is **low** (e.g., < 0.05), at least one predictor significantly contributes to the model.  
      - If the p-value is **high**, the model may not be useful.
    - **Adjusted R²:** If negative or decreasing as more variables are added, it suggests that additional predictors are not improving the model.
    """)
    
    st.markdown("## 5. Next Steps: Improving the Model")
    st.markdown("""
    If the model has **poor fit** (low R², high p-values), consider:
    ✅ Checking for **missing or irrelevant variables**  
    ✅ Transforming variables (e.g., log transformation for skewed data)  
    ✅ Using **interaction terms** or polynomial regression  
    ✅ Trying **different models** (e.g., Ridge, Lasso, Decision Trees)  
    """)
    
    st.markdown("## 🔍 Key Takeaways")
    st.markdown("""
    - **R² tells you how much variance the model explains, but it’s not the only factor to consider.**  
    - **F-statistic and its p-value help determine if the model is significant.**  
    - **Adjusted R² helps evaluate if additional predictors improve the model.**  
    - **Low performance suggests revising predictors, transforming variables, or considering other modeling approaches.**  
    
    🚀 **Understanding these metrics helps in making better, data-driven decisions!**
    """)


# Navigation System
pages = {
    "Anova One-way - Introduction": anova_oneway,
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial Designs with Three Factors and Three Levels": three_factorial,
    "Tips to Analyze the Statistical Outputs":Analysis
}

st.title('Navigation')
selection = st.radio("Go to", list(pages.keys()))
pages[selection]()

