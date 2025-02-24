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
    df = pd.DataFrame(itertools.product(*levels.values()) * replications, columns=levels.keys())
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

def plot_3d(df):
    """Plots a 3D scatter plot of factors vs. Y."""
    fig = go.Figure(data=go.Scatter3d(
        x=df['Temperature_num'], y=df['Pressure_num'], z=df['Thinner_num'],
        mode='markers', marker=dict(size=10, color=df['Y'], opacity=0.8)
    ))
    st.plotly_chart(fig)

def plot_boxplot(df, groupby):
    """Generates a boxplot grouped by a selected factor."""
    df['Group'] = df[groupby].astype(str) if "*" not in groupby else df[groupby.split("*")[0]] + " * " + df[groupby.split("*")[1]]
    fig = go.Figure([go.Box(y=df['Y'], x=df['Group'], boxmean=True)])
    st.plotly_chart(fig)

def fit_model(df):
    """Fits an OLS regression model."""
    formula = "Y ~ " + " + ".join([f'Q("{factor}_num")' for factor in levels]) + " + " + " + ".join([f'Q("{f1}_num"):Q("{f2}_num")' for f1, f2 in itertools.combinations(levels, 2)])
    return smf.ols(formula, data=df).fit()

def Three_factorial():
    st.title('Factorial Design Visualization')
    factor_names = {factor: st.sidebar.text_input(f'Custom name for {factor}', factor) for factor in levels}
    coefficients = [st.sidebar.slider(f'Coefficient {i}', -10.0, 10.0, 0.0) for i in range(7)]
    
    df = create_dataframe()
    df.rename(columns={f"{factor}_num": f"{factor_names[factor]}_num" for factor in levels}, inplace=True)
    df = compute_y(df, coefficients)
    st.dataframe(df)
    
    st.download_button("Download CSV", df.to_csv(index=False), "data.csv", "text/csv")
    st.subheader('Factors space')
    plot_3d(df)
    
    st.subheader('Box plot')
    groupby = st.selectbox('Group by', list(factor_names.values()) + [f'{i}*{j}' for i, j in itertools.combinations(factor_names.values(), 2)])
    plot_boxplot(df, groupby)
    
    st.subheader('Model fitting')
    results = fit_model(df)
    st.text(results.summary())

st.sidebar.title('Navigation')
if st.sidebar.radio("Go to", ["Factorial Design"]):
    Three_factorial()
