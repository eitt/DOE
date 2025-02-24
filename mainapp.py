import streamlit as st
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
import statsmodels.formula.api as smf

# Define levels for factors
levels = {factor: ['low', 'medium', 'high'] for factor in ['Temperature', 'Pressure', 'Thinner']}
mappings = {'low': -1, 'medium': 0, 'high': 1}

def create_dataframe(replications=2):
    """Generates a factorial design dataframe."""
    combinations = list(itertools.product(*levels.values()))
    df = pd.DataFrame(combinations * replications, columns=levels.keys())
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(mappings)
    return df

def compute_y(df, coefficients):
    """Computes response variable Y based on given coefficients."""
    noise = np.random.normal(0, 2, len(df))
    df['Y'] = (coefficients[0] + sum(coefficients[i+1] * df[f'{factor}_num'] for i, factor in enumerate(levels))
                + sum(coefficients[i+4] * df[f'{f1}_num'] * df[f'{f2}_num'] for i, (f1, f2) in enumerate(itertools.combinations(levels, 2)))
                + noise)
    return df

def update_equation(coefficients):
    """Generates the regression equation dynamically based on slider values."""
    terms = ["Intercept"] + [f'{factor}_num' for factor in levels] + [f'{f1}_num:{f2}_num' for f1, f2 in itertools.combinations(levels, 2)]
    equation = "Y = " + " + ".join(f"{coeff:.2f} * {term}" for coeff, term in zip(coefficients, terms))
    return equation

def plot_3d(df):
    """Plots a 3D scatter plot of factors vs. Y."""
    fig = go.Figure(data=go.Scatter3d(
        x=df['Temperature_num'], y=df['Pressure_num'], z=df['Thinner_num'],
        mode='markers', marker=dict(size=10, color=df['Y'], opacity=0.8)
    ))
    st.plotly_chart(fig)

def plot_boxplot(df, groupby):
    """Generates a boxplot grouped by a selected factor."""
    fig = go.Figure([go.Box(y=df['Y'], x=df[groupby], boxmean=True)])
    st.plotly_chart(fig)

def Three_factorial():
    st.title('Factorial Design Visualization')
    factor_names = {factor: st.sidebar.text_input(f'Custom name for {factor}', factor) for factor in levels}
    coefficients = [st.sidebar.slider(f'Coefficient {i}', -10.0, 10.0, 0.0) for i in range(7)]
    
    df = create_dataframe()
    df.rename(columns={f"{factor}_num": f"{factor_names[factor]}_num" for factor in levels}, inplace=True)
    df = compute_y(df, coefficients)
    st.dataframe(df)
    
    st.download_button("Download CSV", df.to_csv(index=False), "data.csv", "text/csv")
    
    st.subheader('Model Equation')
    st.latex(update_equation(coefficients))
    
    st.subheader('Factors Space')
    plot_3d(df)
    
    st.subheader('Box Plot')
    groupby = st.selectbox('Group by', list(factor_names.values()))
    plot_boxplot(df, groupby)
    
    st.subheader('Model Fitting')
    formula = "Y ~ " + " + ".join([f'Q("{factor}_num")' for factor in levels]) + " + " + " + ".join([f'Q("{f1}_num"):Q("{f2}_num")' for f1, f2 in itertools.combinations(levels, 2)])
    results = smf.ols(formula, data=df).fit()
    st.text(results.summary())

def factorial_twolevels():
    st.title('Two-Level Factorial Design')
    factor_names = {'FactorA': 'Factor A', 'FactorB': 'Factor B'}
    coefficients = [st.sidebar.slider(f'Coefficient {i}', -10.0, 10.0, 0.0) for i in range(4)]
    
    df = pd.DataFrame(itertools.product(['low', 'high'], repeat=2), columns=factor_names.keys())
    mapping = {'low': -1, 'high': 1}
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(mapping)
    
    noise = np.random.normal(0, 2, len(df))
    df['Y'] = (coefficients[0] + sum(coefficients[i+1] * df[f'{factor}_num'] for i, factor in enumerate(factor_names))
                + coefficients[3] * df['FactorA_num'] * df['FactorB_num'] + noise)
    
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "two_level_data.csv", "text/csv")
    
    st.subheader('Model Equation')
    equation = f'Y = {coefficients[0]:.2f} + {coefficients[1]:.2f} * FactorA_num + {coefficients[2]:.2f} * FactorB_num + {coefficients[3]:.2f} * FactorA_num * FactorB_num'
    st.latex(equation)
    
    st.subheader('Box Plot')
    plot_boxplot(df, 'FactorA_num')
    
    st.subheader('Model Fitting')
    formula = "Y ~ FactorA_num + FactorB_num + FactorA_num:FactorB_num"
    results = smf.ols(formula, data=df).fit()
    st.text(results.summary())

st.sidebar.title('Navigation')
pages = {"Three-Level Factorial Design": Three_factorial, "Two-Level Factorial Design": factorial_twolevels}
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
