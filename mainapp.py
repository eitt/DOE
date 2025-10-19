# =========================
# Factorial/ANOVA Streamlit App (clean & robust) — Py3.9 safe
# =========================
import itertools
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

# -------------------------
# Globals / constants
# -------------------------
FACTOR_VALUES_3 = {'low': -1, 'medium': 0, 'high': 1}
FACTOR_VALUES_2 = {'low': -1, 'high': 1}


# -------------------------
# Helpers
# -------------------------
def _ensure_unique_ordered_keys(d):
    """Return the keys of a dict in insertion order as a list (explicit)."""
    return list(d.keys())


def create_factorial_dataframe(levels, numeric_mapping, replications=2, random_state=None):
    """
    Generates a factorial design dataframe with numeric mappings for any number of factors.
    `levels` is a dict: {factor_key: [level_str, ...]}
    """
    _ = np.random.default_rng(seed=random_state)  # kept for future use
    grid = list(itertools.product(*levels.values()))
    df = pd.DataFrame(grid * replications, columns=levels.keys())
    # numeric maps
    for col in df.columns:
        if set(df[col].unique()).issubset(set(numeric_mapping.keys())):
            df[f'{col}_num'] = df[col].map(numeric_mapping)
        else:
            # If custom labels are used, map by position
            lvl_list = list(levels[col])
            pos_map = {lvl: numeric_mapping[lvl] for lvl in lvl_list if lvl in numeric_mapping}
            df[f'{col}_num'] = df[col].map(pos_map)
    return df


def compute_response(df, coefficients, factor_name_map, noise_sd=0.5, random_state=None):
    """
    Compute Y = b0 + sum(bi*Xi) + sum(bij*Xi*Xj) + noise
    factor_name_map: {"internal_key": "Custom Name"}
    """
    rng = np.random.default_rng(seed=random_state)
    keys = _ensure_unique_ordered_keys(factor_name_map)

    # Guarantee numeric columns exist with the CUSTOM name suffix
    for k in keys:
        src = f"{k}_num"
        dst = f"{factor_name_map[k]}_num"
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Build y from coefficients
    needed = 1 + len(keys) + len(list(combinations(keys, 2)))
    if len(coefficients) != needed:
        st.warning(f"Coefficient count mismatch: expected {needed}, got {len(coefficients)}. Truncating/Extending with zeros.")
        if len(coefficients) < needed:
            coefficients = coefficients + [0.0] * (needed - len(coefficients))
        else:
            coefficients = coefficients[:needed]

    # Intercept
    y = np.full(len(df), coefficients[0], dtype=float)

    # Main effects
    for i, k in enumerate(keys, start=1):
        col = f"{factor_name_map[k]}_num"
        if col not in df.columns:
            st.error(f"Missing column '{col}' required for main effect.")
            return df
        y += coefficients[i] * df[col].to_numpy(dtype=float)

    # Interactions
    base = 1 + len(keys)
    for j, (k1, k2) in enumerate(combinations(keys, 2)):
        c1 = f"{factor_name_map[k1]}_num"
        c2 = f"{factor_name_map[k2]}_num"
        if c1 not in df.columns or c2 not in df.columns:
            st.error(f"Missing columns '{c1}' or '{c2}' required for interaction.")
            return df
        y += coefficients[base + j] * (df[c1].to_numpy(dtype=float) * df[c2].to_numpy(dtype=float))

    # Noise
    y = y + rng.normal(0, noise_sd, len(df))
    df['Y'] = y
    return df


def fit_factorial_model(df, factor_name_map):
    """OLS with main effects + 2-way interactions on *_num columns (custom names)."""
    keys = _ensure_unique_ordered_keys(factor_name_map)
    mains = [f"Q('{factor_name_map[k]}_num')" for k in keys]
    inters = [f"Q('{factor_name_map[a]}_num'):Q('{factor_name_map[b]}_num')" for a, b in combinations(keys, 2)]
    formula = "Y ~ " + " + ".join(mains + inters)
    return smf.ols(formula, data=df).fit()


def format_equation(results, factor_name_map):
    """
    Human-friendly equation using custom names (no LaTeX special chars).
    """
    keys = _ensure_unique_ordered_keys(factor_name_map)
    coefs = results.params
    parts = []
    # Intercept
    if "Intercept" in coefs.index:
        parts.append(f"{coefs['Intercept']:.3f}")
    # Mains
    for k in keys:
        nm = f"Q('{factor_name_map[k]}_num')"
        if nm in coefs.index:
            parts.append(f"{coefs[nm]:+.3f}·{factor_name_map[k]}")
    # Interactions
    for a, b in combinations(keys, 2):
        nm = f"Q('{factor_name_map[a]}_num'):Q('{factor_name_map[b]}_num')"
        if nm in coefs.index:
            parts.append(f"{coefs[nm]:+.3f}·{factor_name_map[a]}·{factor_name_map[b]}")
    return "Y = " + " ".join(parts)


def plot_2d(df, factor_name_map):
    """2D scatter heat by Y for two-level design."""
    try:
        xname = list(factor_name_map.values())[0]
        yname = list(factor_name_map.values())[1]
        fig = go.Figure(data=go.Scatter(
            x=df[f'{xname}_num'],
            y=df[f'{yname}_num'],
            mode='markers',
            marker=dict(size=10, color=df['Y'], colorbar=dict(title='Y'))
        ))
        fig.update_layout(
            xaxis_title=xname,
            yaxis_title=yname,
            title='2D Factor Space (colored by Y)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"2D plot error: {e}")


def plot_3d(df, factor_name_map):
    """3D scatter for 3-factor design."""
    try:
        v = list(factor_name_map.values())
        x, y, z = v[0], v[1], v[2]
        fig = go.Figure(data=go.Scatter3d(
            x=df[f'{x}_num'],
            y=df[f'{y}_num'],
            z=df[f'{z}_num'],
            mode='markers',
            marker=dict(size=8, color=df['Y'], colorbar=dict(title='Y'), opacity=0.85)
        ))
        fig.update_layout(
            scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
            title='3D Factor Space (colored by Y)',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"3D plot error: {e}")


def plot_surface(df, factor1_custom, factor2_custom):
    """3D surface Y over two *_num axes (requires a grid)."""
    idx = f'{factor1_custom}_num'
    col = f'{factor2_custom}_num'
    if idx not in df.columns or col not in df.columns:
        st.error("Selected factors not found for surface plot.")
        return
    pivot = df.pivot_table(values='Y', index=idx, columns=col, aggfunc='mean')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        st.warning("Not enough grid points to draw a surface. Increase replications/levels.")
    fig = go.Figure(data=[go.Surface(z=pivot.values, x=pivot.index, y=pivot.columns)])
    fig.update_layout(scene=dict(xaxis_title=factor1_custom, yaxis_title=factor2_custom, zaxis_title='Y'),
                      title=f'Surface: {factor1_custom} vs {factor2_custom}', height=550)
    st.plotly_chart(fig, use_container_width=True)


def plot_boxplot(df, groupby_label, factor_name_map):
    """
    Boxplots by a factor (custom name) or an interaction "A * B".
    """
    fig = go.Figure()
    # Interaction?
    if " * " in groupby_label:
        a_label, b_label = groupby_label.split(" * ")
        a_col = f"{a_label}_num"
        b_col = f"{b_label}_num"
        if a_col not in df.columns or b_col not in df.columns:
            st.error(f"Columns '{a_col}' or '{b_col}' not found.")
            return
        inter_col = f"Interaction_{a_label}_{b_label}"
        df[inter_col] = df[a_col].astype(str) + " * " + df[b_col].astype(str)
        gcol = inter_col
    else:
        gcol = f"{groupby_label}_num"
        if gcol not in df.columns:
            st.error(f"Column '{gcol}' not found.")
            return

    for name, group in df.groupby(gcol):
        fig.add_trace(go.Box(y=group['Y'], name=str(name), boxmean=True))

    fig.update_layout(xaxis_title=groupby_label, yaxis_title='Y', title='Boxplot by Group', height=500)
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Pages
# -------------------------
def three_factorial():
    st.title('Factorial Designs with Three Factors and Three Levels')
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    # Sidebar controls
    st.sidebar.header("Simulation controls")
    replications = st.sidebar.slider("Replications per run", 1, 10, 2, 1)
    noise_sd = st.sidebar.slider("Noise σ", 0.0, 5.0, 0.5, 0.1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    # Custom names
    st.sidebar.header("Custom factor names")
    factor_name_map = {
        "Temperature": st.sidebar.text_input("Name for Temperature", "Temperature"),
        "Pressure": st.sidebar.text_input("Name for Pressure", "Pressure"),
        "Thinner": st.sidebar.text_input("Name for Thinner", "Thinner"),
    }

    # Coefficients: 1 + 3 mains + 3 interactions = 7
    st.sidebar.header("Model coefficients")
    coef = [st.sidebar.slider('Intercept', -100.0, 100.0, 0.0)]
    for k in _ensure_unique_ordered_keys(factor_name_map):
        coef.append(st.sidebar.slider(f'Main effect: {factor_name_map[k]}', -100.0, 100.0, 0.0))
    for a, b in combinations(_ensure_unique_ordered_keys(factor_name_map), 2):
        coef.append(st.sidebar.slider(f'Interaction: {factor_name_map[a]} × {factor_name_map[b]}', -100.0, 100.0, 0.0))

    # Build data
    levels = {k: ['low', 'medium', 'high'] for k in factor_name_map}
    df = create_factorial_dataframe(levels, FACTOR_VALUES_3, replications=replications, random_state=seed or None)
    df = compute_response(df, coef, factor_name_map, noise_sd=noise_sd, random_state=seed or None)

    st.subheader('Generated Data')
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="three_factorial_data.csv", mime="text/csv")

    st.subheader('Factors Space')
    plot_3d(df, factor_name_map)

    st.subheader('Analysis of Y based on Variability Source — Box Plot')
    # Grouping options use custom labels
    custom_labels = list(factor_name_map.values())
    groupby_options = custom_labels + [f"{fa} * {fb}" for fa, fb in combinations(custom_labels, 2)]
    groupby_label = st.selectbox('Group by', groupby_options, index=0)
    plot_boxplot(df, groupby_label, factor_name_map)

    st.subheader('Surface Plot')
    f1 = st.selectbox('First Factor', custom_labels, index=0)
    f2 = st.selectbox('Second Factor', custom_labels, index=1)
    if f1 != f2:
        plot_surface(df, f1, f2)
    else:
        st.info("Select two different factors for the surface.")

    st.subheader('Model Fitting')
    results = fit_factorial_model(df, factor_name_map)
    st.code(format_equation(results, factor_name_map))
    # >>> Print text summary (classic)
    st.text(results.summary())


def factorial_twolevels():
    st.title("Introduction to Factorial Designs (2 factors × 2 levels)")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    # Sidebar controls
    st.sidebar.header("Simulation controls")
    replications = st.sidebar.slider("Replications per run", 1, 15, 3, 1)
    noise_sd = st.sidebar.slider("Noise σ", 0.0, 5.0, 0.5, 0.1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    st.sidebar.header("Custom factor names")
    factor_map_2 = {
        "FactorA": st.sidebar.text_input("Name for Factor A", "FactorA"),
        "FactorB": st.sidebar.text_input("Name for Factor B", "FactorB")
    }

    st.sidebar.header("Model coefficients")
    # 1 + 2 mains + 1 interaction = 4
    coef = [
        st.sidebar.slider('Intercept', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'Main effect: {factor_map_2["FactorA"]}', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'Main effect: {factor_map_2["FactorB"]}', -100.0, 100.0, 0.0),
        st.sidebar.slider(f'Interaction: {factor_map_2["FactorA"]} × {factor_map_2["FactorB"]}', -100.0, 100.0, 0.0)
    ]

    levels = {'FactorA': ['low', 'high'], 'FactorB': ['low', 'high']}
    df = create_factorial_dataframe(levels, FACTOR_VALUES_2, replications=replications, random_state=seed or None)

    # Rename numeric columns to custom
    for k, v in factor_map_2.items():
        if f"{k}_num" in df.columns:
            df.rename(columns={f"{k}_num": f"{v}_num"}, inplace=True)

    df = compute_response(df, coef, factor_map_2, noise_sd=noise_sd, random_state=seed or None)

    st.subheader('Generated Data')
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="twolevel_factorial_data.csv", mime="text/csv")

    st.subheader('Factors Space')
    plot_2d(df, factor_map_2)

    st.subheader('Analysis of Y based on Variability Source — Box Plot')
    custom_labels = list(factor_map_2.values())
    groupby_options = custom_labels + [f"{custom_labels[0]} * {custom_labels[1]}"]
    groupby_label = st.selectbox('Group by', groupby_options, index=0)
    plot_boxplot(df, groupby_label, factor_map_2)

    st.subheader('Surface Plot')
    # For 2x2, show the single surface
    plot_surface(df, custom_labels[0], custom_labels[1])

    st.subheader('Model Fitting')
    results = fit_factorial_model(df, factor_map_2)
    st.code(format_equation(results, factor_map_2))
    # >>> Print text summary (classic)
    st.text(results.summary())


def anova_oneway():
    st.title("One-Way ANOVA: One Factor, Three Levels")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    # Controls
    st.sidebar.header("Simulation controls")
    replications = st.sidebar.slider("Replications per level", 3, 50, 15, 1)
    noise_sd = st.sidebar.slider("Noise σ", 0.0, 5.0, 0.5, 0.1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    # Names
    factor_name = st.sidebar.text_input("Factor name", "Factor")
    level_names = {
        "low": st.sidebar.text_input("Level 1 label", "Low"),
        "medium": st.sidebar.text_input("Level 2 label", "Medium"),
        "high": st.sidebar.text_input("Level 3 label", "High")
    }

    # Coeffs: baseline + (medium) + (high)
    st.sidebar.header("Effects (relative to baseline level)")
    coef_intercept = st.sidebar.slider('Baseline mean', -100.0, 100.0, 0.0)
    coef_medium = st.sidebar.slider(f'Effect of {level_names["medium"]}', -100.0, 100.0, 0.0)
    coef_high = st.sidebar.slider(f'Effect of {level_names["high"]}', -100.0, 100.0, 0.0)

    # Build data
    rng = np.random.default_rng(seed=seed or None)
    raw_levels = np.repeat(['low', 'medium', 'high'], replications)
    df = pd.DataFrame({factor_name: raw_levels})
    # Relabel for display
    map_show = {'low': level_names['low'], 'medium': level_names['medium'], 'high': level_names['high']}
    df[factor_name] = df[factor_name].map(map_show)
    # Generate Y
    means = df[factor_name].map({level_names['low']: coef_intercept,
                                 level_names['medium']: coef_intercept + coef_medium,
                                 level_names['high']: coef_intercept + coef_high})
    df['Y'] = means + rng.normal(0, noise_sd, len(df))

    st.subheader('Generated Data')
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="anova_data.csv", mime="text/csv")

    st.subheader('Box Plot for Factor Levels')
    fig = go.Figure()
    for level in df[factor_name].unique():
        fig.add_trace(go.Box(y=df.loc[df[factor_name] == level, "Y"], name=level, boxmean=True))
    fig.update_layout(xaxis_title=factor_name, yaxis_title="Y", title="Distribution of Y across Levels", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Analysis: OLS and classic printed summary
    model = smf.ols(f"Y ~ C({factor_name})", data=df).fit()
    # >>> Print text summary (classic) — no new tables
    st.subheader("Linear Regression Model Summary")
    st.text(model.summary())

    # Variance Decomposition (Plotly pies)
    st.subheader("Variance Decomposition (SST vs. SSTr & SSE)")
    anova_table = sm.stats.anova_lm(model, typ=2)
    sstr = float(anova_table['sum_sq'].iloc[0])
    sse  = float(anova_table['sum_sq'].iloc[1])
    sst  = sstr + sse

    pies = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                         subplot_titles=("Total Sum of Squares (SST)", "Treatment vs Error (SSTr vs SSE)"))

    # Left: SST (single slice)
    pies.add_trace(
        go.Pie(labels=["SST"], values=[sst], textinfo='label+percent'),
        row=1, col=1
    )
    # Right: SSTR vs SSE
    pies.add_trace(
        go.Pie(labels=["SSTr", "SSE"], values=[sstr, sse], textinfo='label+percent'),
        row=1, col=2
    )

    pies.update_layout(height=500, showlegend=False)
    st.plotly_chart(pies, use_container_width=True)


def Analysis():
    st.title("Tips to Analyze the Statistical Outputs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.subheader('Analysis of Y Based on a Source of Variability — Box Plot')
    st.markdown("""
    Box plots by group provide a compact view of distributions: medians, spread, skewness, and outliers.
    They are excellent for quickly spotting differences across categories, shifts over time, and potential anomalies.
    """)
    st.subheader('How to Read an OLS Regression Output')
    st.markdown("""
    Focus on **R²/Adj. R²** (fit), **F-test** (overall significance), **coefficients & p-values** (variable significance),
    and **AIC/BIC** (model parsimony). Poor fit? Consider missing variables, transforms, interactions, or alternative models.
    """)


# -------------------------
# Navigation
# -------------------------
PAGES = {
    "Anova One-way - Introduction": anova_oneway,
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial Designs with Three Factors and Three Levels": three_factorial,
    "Tips to Analyze the Statistical Outputs": Analysis,
}

st.title('Navigation')
choice = st.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
