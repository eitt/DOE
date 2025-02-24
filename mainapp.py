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
    noise = np.random.normal(0, 2, len(df))

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
        st.sidebar.slider('Intercept (Coefficient 0)', -10.0, 10.0, 0.0)
    ] + [
        st.sidebar.slider(f'Coefficient {i+1} ({factor_names[factor]})', -10.0, 10.0, 0.0)
        for i, factor in enumerate(["Temperature", "Pressure", "Thinner"])
    ] + [
        st.sidebar.slider(f'Coefficient {i+4} ({factor_names[f1]} * {factor_names[f2]})', -10.0, 10.0, 0.0)
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
    st.markdown("""
    A factorial design is an experimental design that studies the effects of two or more factors, 
    each with multiple levels, on a dependent variable. It allows researchers to study main effects 
    and interactions efficiently.
    """)

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
        st.sidebar.slider('Intercept (Coefficient 0)', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'{two_level_factor_names["FactorA"]} Coefficient', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'{two_level_factor_names["FactorB"]} Coefficient', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Interaction ({two_level_factor_names["FactorA"]} * {two_level_factor_names["FactorB"]})', -10.0, 10.0, 0.0)
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


# Navigation System
pages = {
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial Designs with Three Factors and Three Levels": three_factorial
}

st.title('Navigation')
selection = st.radio("Go to", list(pages.keys()))
pages[selection]()

