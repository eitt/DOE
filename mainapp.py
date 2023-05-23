import streamlit as st
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations


# Define levels for each factor
levels = {
    'Temperature': ['low', 'medium', 'high'],
    'Pressure': ['low', 'medium', 'high'],
    'Thinner':  ['low', 'medium', 'high']
}

# Default factor names
factor_names = {
    "Temperature": "Temperature",
    "Pressure": "Pressure",
    "Thinner": "Thinner"
}

# Assign new names
factor_names = {
    "Temperature": st.sidebar.text_input("Enter a custom name for Temperature", "Temperature"),
    "Pressure": st.sidebar.text_input("Enter a custom name for Pressure", "Pressure"),
    "Thinner": st.sidebar.text_input("Enter a custom name for Thinner", "Thinner"),
}

def create_dataframe(replications=2):
    # Generate all combinations of factor levels
    combinations = list(itertools.product(*levels.values()))

    # Repeat the combinations for the number of replications
    combinations = combinations * replications

    # Create a DataFrame with the factorial design
    df = pd.DataFrame(combinations, columns=levels.keys())

    # Map factor levels to numeric values
    mappings = {'low': -1, 'medium': 0, 'high': 1}
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(mappings)

    # Rename columns with custom factor names
    # Renaming dataframe columns
    # Rename the columns in df
    df.rename(columns={old: new for old, new in factor_names.items()}, inplace=True)


    return df
def add_numeric_columns(df, factor_names):
    """
    Create numeric columns for each factor in the dataframe.
    """

    # Numeric columns for factors
    for factor, new_name in factor_names.items():
        df[f'{new_name}_num'] = df[new_name].map(factor_values[factor])

    return df


def compute_y(df, coefficients, factor_names):
    noise = np.random.normal(0, 1, len(df))  # 0 is the mean of the normal distribution, and 1 is the standard deviation
    df['Y'] = (coefficients[0]
               + coefficients[1]*df[f'{factor_names["Temperature"]}_num']
               + coefficients[2]*df[f'{factor_names["Pressure"]}_num']
               + coefficients[3]*df[f'{factor_names["Thinner"]}_num']
               + coefficients[4]*df[f'{factor_names["Temperature"]}_num']*df[f'{factor_names["Pressure"]}_num']
               + coefficients[5]*df[f'{factor_names["Temperature"]}_num']*df[f'{factor_names["Thinner"]}_num']
               + coefficients[6]*df[f'{factor_names["Pressure"]}_num']*df[f'{factor_names["Thinner"]}_num']
               + noise)
    return df



def plot_3d(df):
    fig = go.Figure(data=go.Scatter3d(
        x=df[f'{factor_names["Temperature"]}_num'], 
        y=df[f'{factor_names["Pressure"]}_num'], 
        z=df[f'{factor_names["Thinner"]}_num'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Y'], 
            colorbar=dict(title='Y'), 
            opacity=0.8
        ))
    )

    fig.update_layout(scene = dict(
        xaxis_title=factor_names["Temperature"],
        yaxis_title=factor_names["Pressure"],
        zaxis_title=factor_names["Thinner"]
    ))
    st.plotly_chart(fig)
    

def plot_surface(df, factor1, factor2):
    # Create a pivot table for the surface plot
    pivot_table = df.pivot_table(values='Y', index=f'{factor1}_num', columns=f'{factor2}_num')

    # Create a 3D plot
    fig = go.Figure(data=[go.Surface(z=pivot_table.values, x=pivot_table.index, y=pivot_table.columns)])

    # Add labels
    fig.update_layout(scene = dict(
        xaxis_title=factor1,
        yaxis_title=factor2,
        zaxis_title='Y'
    ))
    st.plotly_chart(fig)


def plot_boxplot(df, groupby):
    fig = go.Figure()
    
    # Check if interaction term is selected
    if "*" in groupby:
        factors = groupby.split("*")
        xaxis_title = f"{factors[0]} * {factors[1]}"
        df['Interaction'] = df[f'{factors[0]}'].astype(str) + " * " + df[f'{factors[1]}'].astype(str)
        groupby = 'Interaction'
    elif groupby in factor_names.values():
        xaxis_title = groupby
        groupby = f'{groupby}_num' 
    else:
        xaxis_title = groupby

    # Group by the custom name
    groups = df.groupby(groupby)

    for name, group in groups:
        fig.add_trace(go.Box(y=group['Y'], name=str(name), boxmean=True))

    # Add labels
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Y',
    )
    st.plotly_chart(fig)


def fit_model(df, factor_names):
    formula = f"Y ~ Q('{factor_names['Temperature']}_num') + Q('{factor_names['Pressure']}_num') + Q('{factor_names['Thinner']}_num') + \
                Q('{factor_names['Temperature']}_num'):Q('{factor_names['Pressure']}_num') + \
                Q('{factor_names['Temperature']}_num'):Q('{factor_names['Thinner']}_num') + \
                Q('{factor_names['Pressure']}_num'):Q('{factor_names['Thinner']}_num')"
    model = smf.ols(formula, data=df)
    results = model.fit()
    return results
    
def print_equation(results, factor_names):
    coefs = results.params
    coef_str = [f'{coef:.2f}' if not np.isclose(coef, 1) else '' for coef in coefs]

    equation = f'Y = {coef_str[0]}'
    terms = ['Intercept', f"{factor_names['Temperature']}_num", f"{factor_names['Pressure']}_num", f"{factor_names['Thinner']}_num",
             f"{factor_names['Temperature']}_num:{factor_names['Pressure']}_num",
             f"{factor_names['Temperature']}_num:{factor_names['Thinner']}_num",
             f"{factor_names['Pressure']}_num:{factor_names['Thinner']}_num"]

    for coef, term in zip(coef_str[1:], terms[1:]):
        equation += f' + {coef}*{term}'

    return equation



def homepage():
    st.title('Factorial Design Visualization')
    st.markdown("By **Leonardo Talero**")
    st.markdown("A factorial design is an experimental design that studies the effects of two or more factors, each with multiple levels, on a dependent variable. In a factorial design, the factors are manipulated independently of each other, and the levels of each factor are crossed with the levels of the other factors. This allows researchers to study the main effects of each factor, as well as the interactions between the factors.")
    st.markdown("Factorial designs are more efficient than one-factor-at-a-time designs, because they allow researchers to study the effects of multiple factors with the same number of participants. Additionally, factorial designs can help researchers to identify interactions between factors, which can be important for understanding the effects of the factors on the dependent variable.")
    st.markdown("There are two main types of factorial designs: full factorial designs and fractional factorial designs. Full factorial designs include all possible combinations of levels for each factor. Fractional factorial designs are a subset of full factorial designs that include only a subset of the possible combinations of levels. Fractional factorial designs are less efficient than full factorial designs, but they can be used when resources are limited.")
    
    coefficients = [
        st.sidebar.slider(f'Coefficient 0 (Intercept)', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 1 ({factor_names["Temperature"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 2 ({factor_names["Pressure"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 3 ({factor_names["Thinner"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 4 ({factor_names["Temperature"]}*{factor_names["Pressure"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 5 ({factor_names["Temperature"]}*{factor_names["Thinner"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 6 ({factor_names["Pressure"]}*{factor_names["Thinner"]})', -10.0, 10.0, 0.0),
    ]
    
    df = create_dataframe()
    # rename the columns in the DataFrame to match the user's custom names
    df.rename(columns={f"{factor}_num": f"{factor_names[factor]}_num" for factor in ["Temperature", "Pressure", "Thinner"]}, inplace=True)

    # Compute Y
    df = compute_y(df, coefficients, factor_names)
    st.subheader('Dataframe')
    st.dataframe(df)
    
    csv = df.to_csv(index=False)
    st.download_button(
        "Download data as CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv")

    st.subheader('Factors space')
    plot_3d(df)
    st.subheader('Analysis of Y based on a source of variability')
    st.subheader('Box plot')
    groupby_options = list(factor_names.values()) + [f'{i}*{j}' for i, j in combinations(factor_names.values(), 2)]
    groupby = st.selectbox('Group by', groupby_options, index=0)

    if groupby == f'{factor_names["Temperature"]}*{factor_names["Pressure"]}':
        df['Interaction'] = df[f'{factor_names["Temperature"]}_num'].astype(str) + '*' + df[f'{factor_names["Pressure"]}_num'].astype(str)
    elif groupby == f'{factor_names["Temperature"]}*{factor_names["Thinner"]}':
        df['Interaction'] = df[f'{factor_names["Temperature"]}_num'].astype(str) + '*' + df[f'{factor_names["Thinner"]}_num'].astype(str)
    elif groupby == f'{factor_names["Pressure"]}*{factor_names["Thinner"]}':
        df['Interaction'] = df[f'{factor_names["Pressure"]}_num'].astype(str) + '*' + df[f'{factor_names["Thinner"]}_num'].astype(str)
    else:
        df['Interaction'] = df[f'{groupby}_num']

    plot_boxplot(df, 'Interaction')


    st.subheader('Surface plot')

    # And in the plot_surface function call
    factors = list(factor_names.values())
    factor1 = st.selectbox('Select first factor', factors, index=0)
    factor2 = st.selectbox('Select second factor', factors, index=1)
    plot_surface(df, factor1, factor2)

    st.subheader('Model fitting')

    # Similarly, when you call the fit_model and print_equation functions
    results = fit_model(df, factor_names)
    st.latex(print_equation(results, factor_names))

    st.text(results.summary())


homepage()


