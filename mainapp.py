# loading packages
import streamlit as st
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Define levels for each factor
levels = {
    'Temperature': ['low', 'medium', 'high'],
    'Pressure': ['low', 'medium', 'high'],
    'Thinner':  ['low', 'medium', 'high']
}

def create_dataframe():
    # Generate all combinations of factor levels
    combinations = list(itertools.product(*levels.values()))

    # Create a DataFrame with the factorial design
    df = pd.DataFrame(combinations, columns=levels.keys())

    # Map factor levels to numeric values
    mappings = {'low': -1, 'medium': 0, 'high': 1}
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(mappings)

    return df

def compute_y(df, coefficients):
    # Compute Y using itertools.product to generate pairwise interactions
    factors = [f'{factor}_num' for factor in levels.keys()]
    interactions = list(itertools.combinations(factors, 2))
    df['Y'] = coefficients[0] + np.random.normal(size=len(df))
    
    for coef, factor in zip(coefficients[1:4], factors):
        df['Y'] += coef * df[factor]
        
    for coef, interaction in zip(coefficients[4:], interactions):
        df['Y'] += coef * df[interaction[0]] * df[interaction[1]]

    return df

def plot_3d(df):
    # Create a 3D plot
    fig = go.Figure(data=go.Scatter3d(
        x=df['Temperature_num'], 
        y=df['Pressure_num'], 
        z=df['Thinner_num'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Y'],  
            colorbar=dict(title='Y'),              
            opacity=0.8
        )
    ))

    # Add labels
    fig.update_layout(scene = dict(
        xaxis_title='Temperature',
        yaxis_title='Pressure',
        zaxis_title='Thinner'
    ))
    st.plotly_chart(fig)

def plot_surface(df, factor1, factor2):
    # Create a pivot table for the surface plot
    pivot_table = df.pivot_table(values='Y', index=factor1, columns=factor2)
    
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
    # Create a box plot for each group
    fig = go.Figure()

    groups = df.groupby(groupby)

    for name, group in groups:
        fig.add_trace(go.Box(y=group['Y'], name=str(name), boxmean=True))

    # Add labels
    fig.update_layout(
        xaxis_title=groupby,
        yaxis_title='Y',
    )

    st.plotly_chart(fig)

def fit_model(df):
    # Fit a linear regression model
    model = smf.ols('Y ~ Temperature_num + Pressure_num + Thinner_num + Temperature_num:Pressure_num + Temperature_num:Thinner_num + Pressure_num:Thinner_num', data=df)
    results = model.fit()

    return results

def print_equation(results):
    params = results.params
    equation = f"Y = {params[0]:.2f} + {params[1]:.2f} * Temperature + {params[2]:.2f} * Pressure + {params[3]:.2f} * Thinner + {params[4]:.2f} * Temperature:Pressure + {params[5]:.2f} * Temperature:Thinner + {params[6]:.2f} * Pressure:Thinner + e"
    return equation


def homepage():
    st.title('Factorial Design Visualization')
    st.markdown("By **Leonardo Talero**")
    st.markdown("A factorial design is an experimental design that studies the effects of two or more factors, each with multiple levels, on a dependent variable. In a factorial design, the factors are manipulated independently of each other, and the levels of each factor are crossed with the levels of the other factors. This allows researchers to study the main effects of each factor, as well as the interactions between the factors.")
    st.markdown("Factorial designs are more efficient than one-factor-at-a-time designs, because they allow researchers to study the effects of multiple factors with the same number of participants. Additionally, factorial designs can help researchers to identify interactions between factors, which can be important for understanding the effects of the factors on the dependent variable.")
    st.markdown("There are two main types of factorial designs: full factorial designs and fractional factorial designs. Full factorial designs include all possible combinations of levels for each factor. Fractional factorial designs are a subset of full factorial designs that include only a subset of the possible combinations of levels. Fractional factorial designs are less efficient than full factorial designs, but they can be used when resources are limited.")

    descriptions = ['Intercept', 'Temperature', 'Pressure', 'Thinner', 
                    'Temperature*Pressure', 'Temperature*Thinner', 'Pressure*Thinner']
    coefficients = [st.sidebar.slider(f'Coefficient {i} ({desc})', 1.5, 10.0, 0.0) 
                    for i, desc in enumerate(descriptions)]

    df = create_dataframe()
    df = compute_y(df, coefficients)
    st.subheader('Dataframe')
    st.dataframe(df)
    st.subheader('Factors space')
    plot_3d(df)
    st.subheader('Analysis of Y based on a source of varaibility')
    st.subheader('Box plot')
    groupby_options = ['Temperature', 'Pressure', 'Thinner', 'Temperature*Pressure', 'Temperature*Thinner', 'Pressure*Thinner']
    groupby = st.selectbox('Group by', groupby_options, index=0)

    if groupby == 'Temperature*Pressure':
        df['Interaction'] = df['Temperature'] + '*' + df['Pressure']
    elif groupby == 'Temperature*Thinner':
        df['Interaction'] = df['Temperature'] + '*' + df['Thinner']
    elif groupby == 'Pressure*Thinner':
        df['Interaction'] = df['Pressure'] + '*' + df['Thinner']
    else:
        df['Interaction'] = df[groupby]

    plot_boxplot(df, 'Interaction')


    st.subheader('Surface plot')
    factors = [ 'Temperature', 'Pressure','Thinner']
    factor1 = st.selectbox('Select first factor', factors, index=0)

    factor2 = st.selectbox('Select second factor', factors, index=1)

    plot_surface(df, factor1, factor2)

    st.subheader('Model fitting')
    
    results = fit_model(df)
    st.latex(print_equation(results))

    st.text(results.summary())

homepage()

