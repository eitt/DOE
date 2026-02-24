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
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------
# Globals / constants
# -------------------------
FACTOR_VALUES_3 = {'low': -1, 'medium': 0, 'high': 1}
FACTOR_VALUES_2 = {'low': -1, 'high': 1}


# -------------------------
# Helper utilities
# -------------------------
def _ensure_unique_ordered_keys(d):
    """Return the keys of a dict in insertion order as a list (explicit)."""
    return list(d.keys())


def _is_finite_array(a):
    """Return True if ndarray-like contains only finite values."""
    try:
        arr = np.asarray(a, dtype=float)
        return np.isfinite(arr).all()
    except Exception:
        return False


def _safe_plot(fig):
    """Wrap st.plotly_chart with minimal guard."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plot rendering error: {e}")

def render_model_summary(model_or_summary, title="Model Summary"):
    """Render statsmodels summary in monospace style similar to print(model.summary())."""
    st.markdown(f"**{title}**")
    try:
        summary_txt = str(model_or_summary.summary())
    except Exception:
        summary_txt = str(model_or_summary)
    st.code(summary_txt, language="text")


def _clean_term_names(terms_list):
    """Cleans statsmodels regression terms for pretty plotting."""
    cleaned = []
    for term in terms_list:
        # e.g., "Q('Temp_num'):Q('Pressure_num')" -> "Temp x Pressure"
        # e.g., "Q('Temp_num')" -> "Temp"
        t = term.replace("Q('", "").replace("_num')", "").replace("'):", " x ").replace("'", "")
        cleaned.append(t)
    return cleaned


def plot_pareto(results):
    """
    Generates a Pareto plot of effect magnitudes, coloring by significance.
    """
    try:
        effects = results.params.drop('Intercept')
        p_values = results.pvalues.loc[effects.index]
        
        df = pd.DataFrame({
            'abs_effect': effects.abs(),
            'p_value': p_values
        }).sort_values('abs_effect', ascending=False)
        
        df['is_significant'] = df['p_value'] < 0.05
        df['color'] = df['is_significant'].map({True: 'blue', False: 'grey'})
        df['term_clean'] = _clean_term_names(df.index)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['term_clean'],
            y=df['abs_effect'],
            marker_color=df['color'],
            text=df['abs_effect'].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Pareto Plot of Effect Magnitudes',
            xaxis_title='Factor / Interaction',
            yaxis_title='Absolute Effect (Coefficient)',
            xaxis=dict(categoryorder='total descending'), # Sort bars by y-value
            showlegend=False,
            height=500
        )
        _safe_plot(fig)
    except Exception as e:
        st.warning(f"Could not generate Pareto Plot: {e}")


def plot_daniel(results):
    """
    Generates a Normal Probability Plot (Daniel Plot) of effects.
    """
    try:
        effects = results.params.drop('Intercept')
        terms = _clean_term_names(effects.index)
        
        # Get plot data and fit line from scipy.stats.probplot
        (osm, osr), (slope, intercept, r) = stats.probplot(effects, dist='norm', fit=True)
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=osm,
            y=osr,
            mode='markers+text',
            text=terms,
            textposition='top right',
            marker_color='blue',
            name='Effects'
        ))
        
        # Add the fitted line
        line_x = np.array([np.min(osm), np.max(osm)])
        line_y = slope * line_x + intercept
        
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode='lines',
            line=dict(dash='dash', color='red'),
            name='Fit Line (Insignificant Effects)'
        ))
        
        fig.update_layout(
            title='Normal Plot of Effects (Daniel Plot)',
            xaxis_title='Theoretical Quantiles (z-scores)',
            yaxis_title='Ordered Effect (Coefficient)',
            showlegend=True,
            height=500
        )
        _safe_plot(fig)
    
    except Exception as e:
        st.warning(f"Could not generate Daniel Plot: {e}")


def create_factorial_dataframe(levels, numeric_mapping, replications=2, random_state=None):
    """
    Generates a factorial design dataframe with numeric mappings for any number of factors.
    `levels` is a dict: {factor_key: [level_str, ...]}
    """
    _ = np.random.default_rng(seed=random_state)  # reserved for future stochastic use
    grid = list(itertools.product(*levels.values()))
    df = pd.DataFrame(grid * replications, columns=levels.keys())
    # numeric maps
    for col in df.columns:
        if set(df[col].unique()).issubset(set(numeric_mapping.keys())):
            df[f'{col}_num'] = df[col].map(numeric_mapping)
        else:
            # Fallback: map by position if someone passed custom labels
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

    # Ensure *_num columns are named with the CUSTOM label
    for k in keys:
        src = f"{k}_num"
        dst = f"{factor_name_map[k]}_num"
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Build y from coefficients
    needed = 1 + len(keys) + len(list(combinations(keys, 2)))
    if len(coefficients) != needed:
        st.warning(f"Coefficient count mismatch: expected {needed}, got {len(coefficients)}. "
                   f"Truncating/Extending with zeros.")
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
    """Human-friendly equation using custom names (no LaTeX special chars)."""
    keys = _ensure_unique_ordered_keys(factor_name_map)
    coefs = results.params
    parts = []
    if "Intercept" in coefs.index:
        parts.append(f"{coefs['Intercept']:.3f}")
    for k in keys:
        nm = f"Q('{factor_name_map[k]}_num')"
        if nm in coefs.index:
            parts.append(f"{coefs[nm]:+.3f}·{factor_name_map[k]}")
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
        x_vals = list(np.asarray(df[f'{xname}_num']).ravel())
        y_vals = list(np.asarray(df[f'{yname}_num']).ravel())
        y_color = list(np.asarray(df['Y']).ravel())
        if not (_is_finite_array(x_vals) and _is_finite_array(y_vals) and _is_finite_array(y_color)):
            st.error("Non-finite values in 2D plot.")
            return
        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(size=10, color=y_color, colorbar=dict(title='Y'))
        ))
        fig.update_layout(
            xaxis_title=xname,
            yaxis_title=yname,
            title='2D Factor Space (colored by Y)',
            height=500
        )
        _safe_plot(fig)
    except Exception as e:
        st.error(f"2D plot error: {e}")


def plot_3d(df, factor_name_map):
    """3D scatter for 3-factor design."""
    try:
        v = list(factor_name_map.values())
        x, y, z = v[0], v[1], v[2]
        xv = list(np.asarray(df[f'{x}_num']).ravel())
        yv = list(np.asarray(df[f'{y}_num']).ravel())
        zv = list(np.asarray(df[f'{z}_num']).ravel())
        cvals = list(np.asarray(df['Y']).ravel())
        if not (_is_finite_array(xv) and _is_finite_array(yv) and _is_finite_array(zv) and _is_finite_array(cvals)):
            st.error("Non-finite values in 3D plot.")
            return
        fig = go.Figure(data=go.Scatter3d(
            x=xv, y=yv, z=zv,
            mode='markers',
            marker=dict(size=8, color=cvals, colorbar=dict(title='Y'), opacity=0.85)
        ))
        fig.update_layout(
            scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z),
            title='3D Factor Space (colored by Y)',
            height=600
        )
        _safe_plot(fig)
    except Exception as e:
        st.error(f"3D plot error: {e}")


def plot_surface(df, factor1_custom, factor2_custom):
    """3D surface Y over two *_num axes (requires a complete grid)."""
    idx = f'{factor1_custom}_num'
    col = f'{factor2_custom}_num'
    if idx not in df.columns or col not in df.columns:
        st.error("Selected factors not found for surface plot.")
        return

    pivot = (
        df.pivot_table(values='Y', index=idx, columns=col, aggfunc='mean')
          .sort_index(axis=0)
          .sort_index(axis=1)
    )

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        st.warning("Not enough grid points to draw a surface. Increase replications/levels.")
        return
    if pivot.isna().any().any():
        st.error("Surface has missing cells (NaN). Increase replications or ensure all level combinations exist.")
        return

    try:
        x = np.asarray(pivot.index, dtype=float)
        y = np.asarray(pivot.columns, dtype=float)
        Z = np.asarray(pivot.values, dtype=float)
        if not (_is_finite_array(x) and _is_finite_array(y) and _is_finite_array(Z)):
            st.error("Surface contains non-finite values.")
            return
    except Exception:
        st.error("Surface axes must be numeric. Ensure factors are coded as numbers (e.g., -1, 0, 1).")
        return

    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
    fig.update_layout(
        scene=dict(
            xaxis_title=factor1_custom,
            yaxis_title=factor2_custom,
            zaxis_title='Y'
        ),
        title=f'Surface: {factor1_custom} vs {factor2_custom}',
        height=550
    )
    _safe_plot(fig)


def plot_boxplot(df, groupby_label, factor_name_map):
    """
    Boxplots by a factor (custom name) or interactions:
    supports "A", "A * B", and "A * B * C" where labels are the CUSTOM factor names.
    We construct an explicit group label for each row so interaction levels don't collapse.
    """
    parts = [p.strip() for p in groupby_label.split('*') if p.strip()]

    def col_from_label(lbl):
        return f"{lbl}_num"

    try:
        if len(parts) == 1:
            a = parts[0]
            a_col = col_from_label(a)
            if a_col not in df.columns:
                st.error(f"Column '{a_col}' not found in DataFrame.")
                return
            group_series = df[a_col].astype(str)

        elif len(parts) == 2:
            a, b = parts
            a_col, b_col = col_from_label(a), col_from_label(b)
            for c in (a_col, b_col):
                if c not in df.columns:
                    st.error(f"Column '{c}' not found in DataFrame.")
                    return
            group_series = (df[a_col].astype(str) + " * " + df[b_col].astype(str))

        elif len(parts) == 3:
            a, b, c = parts
            a_col, b_col, c_col = col_from_label(a), col_from_label(b), col_from_label(c)
            for col in (a_col, b_col, c_col):
                if col not in df.columns:
                    st.error(f"Column '{col}' not found in DataFrame.")
                    return
            group_series = (
                df[a_col].astype(str) + " * " + df[b_col].astype(str) + " * " + df[c_col].astype(str)
            )
        else:
            st.error("Only up to 3-way interactions are supported.")
            return
    except Exception as e:
        st.error(f"Failed to build interaction groups: {e}")
        return

    cats = pd.Categorical(group_series, categories=sorted(group_series.unique()), ordered=True)
    group_series = pd.Series(cats, index=df.index, name="__group__")

    fig = go.Figure()
    for cat in cats.categories:
        mask = (group_series == cat)
        yvals = list(np.asarray(df.loc[mask, 'Y']).ravel())
        if len(yvals) == 0:
            continue
        if not _is_finite_array(yvals):
            st.error(f"Non-finite values in boxplot group '{cat}'.")
            return
        fig.add_trace(go.Box(y=yvals, name=str(cat), boxmean=True, showlegend=False))

    fig.update_layout(
        xaxis_title=groupby_label,
        yaxis_title='Y',
        title='Boxplot by Group',
        height=520,
        boxmode='group',
        xaxis=dict(categoryorder='array', categoryarray=list(cats.categories))
    )
    _safe_plot(fig)

def _format_tukey_summary_for_display(tukey_result, data_df, group_col, response_col='Y'):
    """
    Processes Tukey HSD results to generate a grouped summary table.
    Similar to the example: Age Group, N, Mean, Grouping.
    """
    # 1. Get means and N for each group
    group_stats = data_df.groupby(group_col)[response_col].agg(['mean', 'count']).reset_index()
    group_stats.rename(columns={'mean': 'Mean', 'count': 'N', group_col: 'Group'}, inplace=True)
    group_stats['Mean'] = group_stats['Mean'].round(2)
    
    # 2. Extract significant differences
    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                            columns=tukey_result._results_table.data[0])
    
    # Filter only significant differences
    significant_diffs = tukey_df[tukey_df['reject']].copy()

    # 3. Assign grouping letters based on significant differences
    # This logic is complex and aims to replicate Minitab's grouping
    
    # Sort groups by mean, descending. This is the order we'll process.
    sorted_groups = group_stats.sort_values('Mean', ascending=False)['Group'].tolist()
    
    # Dictionary to hold the letters for each group
    group_letters = {group: [] for group in sorted_groups}
    
    current_letter_index = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # This list will hold sets of groups. Each set is a "clique" that gets a letter.
    letter_groups = []

    for i, group1 in enumerate(sorted_groups):
        # Check if this group is already part of a letter group
        is_grouped = any(group1 in letter_group for letter_group in letter_groups)
        
        if not is_grouped:
            # Start a new letter group
            new_letter_group = {group1}
            
            # Look at remaining groups to see if they can join
            for group2 in sorted_groups[i+1:]:
                
                # Check if group2 is non-significant with ALL members of the current new_letter_group
                is_non_significant_with_all = True
                for member in new_letter_group:
                    is_diff = not significant_diffs[
                        ((significant_diffs['group1'] == group2) & (significant_diffs['group2'] == member)) |
                        ((significant_diffs['group1'] == member) & (significant_diffs['group2'] == group2))
                    ].empty
                    
                    if is_diff:
                        is_non_significant_with_all = False
                        break
                
                if is_non_significant_with_all:
                    new_letter_group.add(group2)

            letter_groups.append(new_letter_group)

    # Now assign the letters based on the cliques
    letter_groups.sort(key=lambda g: min(sorted_groups.index(m) for m in g))
    
    for letter_group in letter_groups:
        letter = letters[current_letter_index]
        for group in letter_group:
            group_letters[group].append(letter)
        current_letter_index += 1

    # Format the final table
    group_stats['Grouping'] = group_stats['Group'].map(lambda g: "".join(group_letters[g]))
    
    return group_stats.sort_values('Mean', ascending=False).reset_index(drop=True)


# -------------------------
# Pages
# -------------------------
def three_factorial():
    st.title('Factorial Designs with Three Factors and Three Levels')
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    # Sidebar controls
    st.sidebar.header("Simulation controls")

    st.sidebar.markdown("**Slider Range Controls**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        coef_min = st.number_input("Coefficients Min", value=-10.0, step=1.0)
        coef_step = st.number_input("Coefficients Step", value=0.5, step=0.1, min_value=0.01, format="%.2f")
    with c2:
        coef_max = st.number_input("Coefficients Max", value=10.0, step=1.0)
        noise_max = st.number_input("Noise σ Max", value=5.0, min_value=0.1, step=0.5)
    st.sidebar.markdown("---")

    replications = st.sidebar.slider("Replications per run", 1, 10, 2, 1)
    noise_sd = st.sidebar.slider("Noise σ", 0.0, noise_max, 0.5, 0.1)
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
    coef = [st.sidebar.slider('Intercept', coef_min, coef_max, 0.0, step=coef_step)]
    for k in _ensure_unique_ordered_keys(factor_name_map):
        coef.append(st.sidebar.slider(f'Main effect: {factor_name_map[k]}', coef_min, coef_max, 0.0, step=coef_step))
    for a, b in combinations(_ensure_unique_ordered_keys(factor_name_map), 2):
        coef.append(st.sidebar.slider(f'Interaction: {factor_name_map[a]} × {factor_name_map[b]}', coef_min, coef_max, 0.0, step=coef_step))

    # Build data
    levels = {k: ['low', 'medium', 'high'] for k in factor_name_map}
    df = create_factorial_dataframe(levels, FACTOR_VALUES_3, replications=replications, random_state=seed or None)
    df = compute_response(df, coef, factor_name_map, noise_sd=noise_sd, random_state=seed or None)

    # Rename categorical columns to custom names for display, download, and categorical models
    categorical_cols_to_rename = {k: v for k, v in factor_name_map.items() if k in df.columns}
    df.rename(columns=categorical_cols_to_rename, inplace=True)
    custom_names = list(factor_name_map.values())

    st.subheader('Generated Data')
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="three_factorial_data.csv", mime="text/csv")

    st.subheader('Factors Space')
    plot_3d(df, factor_name_map)

    st.subheader('Analysis of Y based on Variability Source — Box Plot')
    groupby_options = (
        custom_names
        + [f"{fa} * {fb}" for fa, fb in combinations(custom_names, 2)]
        + [f"{custom_names[0]} * {custom_names[1]} * {custom_names[2]}"]
    )
    groupby_label = st.selectbox('Group by', groupby_options, index=0)
    plot_boxplot(df, groupby_label, factor_name_map)

    st.subheader('Surface Plot')
    f1 = st.selectbox('First Factor', custom_names, index=0)
    f2 = st.selectbox('Second Factor', custom_names, index=1)
    if f1 != f2:
        plot_surface(df, f1, f2)
    else:
        st.info("Select two different factors for the surface.")

    st.subheader('Model Fitting (Response Surface)')
    results = fit_factorial_model(df, factor_name_map)
    st.code(format_equation(results, factor_name_map))
    render_model_summary(results, "OLS Regression Results")

    st.subheader("Effect Significance Plots")
    st.markdown("""
    These plots help visualize the relative importance of each factor and interaction from the regression model.

    - **Pareto Plot:** Sorts effects from largest to smallest absolute magnitude. Bars shaded **blue** are statistically significant ($p < 0.05$), while **grey** bars are not. This helps quickly identify the "vital few" factors that have the largest impact on the response.
    - **Daniel Plot (Normal Plot):** Checks which effects are significant. Insignificant effects (pure noise) will tend to fall along the **red line**. Significant effects (real factor impacts) will "pop off" this line, appearing as outliers. This is a visual way to separate real signals from random noise.
    """)
    c1, c2 = st.columns(2)
    with c1:
        plot_pareto(results)
    with c2:
        plot_daniel(results)

    st.subheader("ANOVA Table (Categorical)")
    st.markdown("This model treats factors as **categories** (e.g., 'low', 'medium', 'high') to partition variance, "
                "which differs from the regression model above that uses **numeric** codes (-1, 0, 1).")
    try:
        # Use C() and Q() to handle custom names with spaces
        f1, f2, f3 = [f"C(Q('{name}'))" for name in custom_names]
        formula_anova = f"Y ~ {f1} * {f2} * {f3}"
        model_anova = smf.ols(formula_anova, data=df).fit()
        anova_table = sm.stats.anova_lm(model_anova, typ=2)
        
        # Clean up index names for display
        anova_table.index = anova_table.index.str.replace("C\(Q\('", "", regex=True).str.replace("'\)\)", "", regex=True)
        
        st.dataframe(anova_table)
    except Exception as e:
        st.error(f"Could not generate ANOVA table: {e}")


    # --- Post-hoc analysis (Tukey HSD) ---
    st.subheader("Post-hoc Analysis (Tukey HSD)")

    st.markdown("**Main Effects**")
    for fac in custom_names: # Use custom names
        st.caption(f"Pairwise comparisons for {fac}")
        tuk = pairwise_tukeyhsd(endog=df["Y"], groups=df[fac], alpha=0.05)
        
        # Create full dataframe for filtering and download
        tuk_df_full = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
        
        # Show only significant differences
        st.dataframe(tuk_df_full[tuk_df_full['reject'] == True])
        
        # Show grouped summary
        tukey_grouped_df = _format_tukey_summary_for_display(tuk, df, fac)
        st.dataframe(tukey_grouped_df)
        st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

        st.download_button(
            label=f"Download Full {fac} Pairwise Report",
            data=tuk_df_full.to_csv(index=False).encode("utf-8"),
            file_name=f"{fac}_tukey_full_report.csv",
            mime="text/csv",
            key=f"download_tukey_full_{fac}"
        )


    st.markdown("**Two-way Interaction Cells**")
    for fa, fb in combinations(custom_names, 2): # Use custom names
        interaction_label = f"{fa} * {fb}"
        st.caption(f"Cells for {interaction_label}")
        groups = df[[fa, fb]].astype(str).agg(' * '.join, axis=1)
        tuk = pairwise_tukeyhsd(endog=df["Y"], groups=groups, alpha=0.05)
        
        # Create full dataframe for filtering and download
        tuk_df_full = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])

        # Show only significant differences
        st.dataframe(tuk_df_full[tuk_df_full['reject'] == True])
        
        # Create a temporary dataframe with combined groups for _format_tukey_summary_for_display
        temp_df_interaction = df.copy()
        temp_df_interaction['__interaction_groups__'] = groups
        
        tukey_grouped_df = _format_tukey_summary_for_display(tuk, temp_df_interaction, '__interaction_groups__')
        st.dataframe(tukey_grouped_df)
        st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

        st.download_button(
            label=f"Download Full {interaction_label} Pairwise Report",
            data=tuk_df_full.to_csv(index=False).encode("utf-8"),
            file_name=f"{interaction_label.replace(' * ', '_')}_tukey_full_report.csv",
            mime="text/csv",
            key=f"download_tukey_full_{interaction_label}"
        )


    st.markdown("**Three-way Interaction Cells (A × B × C)**")
    interaction_label_3way = f"{custom_names[0]} * {custom_names[1]} * {custom_names[2]}"
    groups_3 = df[custom_names].astype(str).agg(' * '.join, axis=1) # Use custom names
    tuk3 = pairwise_tukeyhsd(endog=df["Y"], groups=groups_3, alpha=0.05)

    # Create full dataframe for filtering and download
    tuk_df_full_3way = pd.DataFrame(data=tuk3._results_table.data[1:], columns=tuk3._results_table.data[0])
    
    # Show only significant differences
    st.dataframe(tuk_df_full_3way[tuk_df_full_3way['reject'] == True])

    temp_df_3way_interaction = df.copy()
    temp_df_3way_interaction['__3way_interaction_groups__'] = groups_3
    
    tukey_grouped_df = _format_tukey_summary_for_display(tuk3, temp_df_3way_interaction, '__3way_interaction_groups__')
    st.dataframe(tukey_grouped_df)
    st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")
    
    st.download_button(
        label=f"Download Full {interaction_label_3way} Pairwise Report",
        data=tuk_df_full_3way.to_csv(index=False).encode("utf-8"),
        file_name=f"{interaction_label_3way.replace(' * ', '_')}_tukey_full_report.csv",
        mime="text/csv",
        key=f"download_tukey_full_{interaction_label_3way}"
    )


def factorial_twolevels():
    st.title("Introduction to Factorial Designs (2 factors × 2 levels)")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    st.sidebar.header("Simulation controls")

    st.sidebar.markdown("**Slider Range Controls**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        coef_min = st.number_input("Coefficients Min", value=-10.0, step=1.0)
        coef_step = st.number_input("Coefficients Step", value=0.5, step=0.1, min_value=0.01, format="%.2f")
    with c2:
        coef_max = st.number_input("Coefficients Max", value=10.0, step=1.0)
        noise_max = st.number_input("Noise σ Max", value=5.0, min_value=0.1, step=0.5)
    st.sidebar.markdown("---")

    replications = st.sidebar.slider("Replications per run", 1, 15, 3, 1)
    noise_sd = st.sidebar.slider("Noise σ", 0.0, noise_max, 0.5, 0.1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    st.sidebar.header("Custom factor names")
    factor_map_2 = {
        "FactorA": st.sidebar.text_input("Name for Factor A", "FactorA"),
        "FactorB": st.sidebar.text_input("Name for Factor B", "FactorB")
    }

    st.sidebar.header("Model coefficients")
    coef = [
        st.sidebar.slider('Intercept', coef_min, coef_max, 0.0, step=coef_step),
        st.sidebar.slider(f'Main effect: {factor_map_2["FactorA"]}', coef_min, coef_max, 0.0, step=coef_step),
        st.sidebar.slider(f'Main effect: {factor_map_2["FactorB"]}', coef_min, coef_max, 0.0, step=coef_step),
        st.sidebar.slider(f'Interaction: {factor_map_2["FactorA"]} × {factor_map_2["FactorB"]}', coef_min, coef_max, 0.0, step=coef_step)
    ]

    levels = {'FactorA': ['low', 'high'], 'FactorB': ['low', 'high']}
    df = create_factorial_dataframe(levels, FACTOR_VALUES_2, replications=replications, random_state=seed or None)

    # Rename numeric columns to custom (this is for _num columns, handled by compute_response)
    for k, v in factor_map_2.items():
        if f"{k}_num" in df.columns:
            df.rename(columns={f"{k}_num": f"{v}_num"}, inplace=True)

    df = compute_response(df, coef, factor_map_2, noise_sd=noise_sd, random_state=seed or None)

    # Rename categorical columns to custom names for display, download, and categorical models
    categorical_cols_to_rename = {k: v for k, v in factor_map_2.items() if k in df.columns}
    df.rename(columns=categorical_cols_to_rename, inplace=True)
    custom_names = list(factor_map_2.values())

    st.subheader('Generated Data')
    st.dataframe(df)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="twolevel_factorial_data.csv", mime="text/csv")

    st.subheader('Factors Space')
    plot_2d(df, factor_map_2)

    st.subheader('Analysis of Y based on Variability Source — Box Plot')
    groupby_options = custom_names + [f"{custom_names[0]} * {custom_names[1]}"]
    groupby_label = st.selectbox('Group by', groupby_options, index=0)
    plot_boxplot(df, groupby_label, factor_map_2)

    st.subheader('Surface Plot')
    plot_surface(df, custom_names[0], custom_names[1])

    st.subheader('Model Fitting (Response Surface)')
    results = fit_factorial_model(df, factor_map_2)
    st.code(format_equation(results, factor_map_2))
    render_model_summary(results, "OLS Regression Results")

    st.subheader("Effect Significance Plots")
    st.markdown("""
    These plots help visualize the relative importance of each factor and interaction from the regression model.

    - **Pareto Plot:** Sorts effects from largest to smallest absolute magnitude. Bars shaded **blue** are statistically significant ($p < 0.05$), while **grey** bars are not. This helps quickly identify the "vital few" factors that have the largest impact on the response.
    - **Daniel Plot (Normal Plot):** Checks which effects are significant. Insignificant effects (pure noise) will tend to fall along the **red line**. Significant effects (real factor impacts) will "pop off" this line, appearing as outliers. This is a visual way to separate real signals from random noise.
    """)
    c1, c2 = st.columns(2)
    with c1:
        plot_pareto(results)
    with c2:
        plot_daniel(results)

    st.subheader("ANOVA Table (Categorical)")
    st.markdown("This model treats factors as **categories** (e.g., 'low', 'high') to partition variance, "
                "which differs from the regression model above that uses **numeric** codes (-1, 1).")
    try:
        # Use C() and Q() to handle custom names with spaces
        f1, f2 = [f"C(Q('{name}'))" for name in custom_names]
        formula_anova = f"Y ~ {f1} * {f2}"
        model_anova = smf.ols(formula_anova, data=df).fit()
        anova_table = sm.stats.anova_lm(model_anova, typ=2)

        # Clean up index names for display
        anova_table.index = anova_table.index.str.replace("C\(Q\('", "", regex=True).str.replace("'\)\)", "", regex=True)

        st.dataframe(anova_table)
    except Exception as e:
        st.error(f"Could not generate ANOVA table: {e}")

    # --- Post-hoc analysis (Tukey HSD) ---
    st.subheader("Post-hoc Analysis (Tukey HSD)")
    
    st.markdown("**Main Effects**")
    for fac in custom_names: # Use custom names
        st.caption(f"Pairwise comparisons for {fac}")
        tuk = pairwise_tukeyhsd(endog=df["Y"], groups=df[fac], alpha=0.05)
        
        # Create full dataframe for filtering and download
        tuk_df_full = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
        
        # Show only significant differences
        st.dataframe(tuk_df_full[tuk_df_full['reject'] == True])

        # Show grouped summary
        tukey_grouped_df = _format_tukey_summary_for_display(tuk, df, fac)
        st.dataframe(tukey_grouped_df)
        st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

        st.download_button(
            label=f"Download Full {fac} Pairwise Report",
            data=tuk_df_full.to_csv(index=False).encode("utf-8"),
            file_name=f"{fac}_tukey_full_report.csv",
            mime="text/csv",
            key=f"download_tukey_full_{fac}"
        )


    st.markdown("**Interaction Cells (A × B)**")
    interaction_label = f"{custom_names[0]} * {custom_names[1]}"
    groups = df[custom_names].astype(str).agg(' * '.join, axis=1) # Use custom names
    tuk_ab = pairwise_tukeyhsd(endog=df["Y"], groups=groups, alpha=0.05)
    
    # Create full dataframe for filtering and download
    tuk_df_full_ab = pd.DataFrame(data=tuk_ab._results_table.data[1:], columns=tuk_ab._results_table.data[0])
    
    # Show only significant differences
    st.dataframe(tuk_df_full_ab[tuk_df_full_ab['reject'] == True])

    temp_df_interaction = df.copy()
    temp_df_interaction['__interaction_groups__'] = groups
    
    tukey_grouped_df = _format_tukey_summary_for_display(tuk_ab, temp_df_interaction, '__interaction_groups__')
    st.dataframe(tukey_grouped_df)
    st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

    st.download_button(
        label=f"Download Full {interaction_label} Pairwise Report",
        data=tuk_df_full_ab.to_csv(index=False).encode("utf-8"),
        file_name=f"{interaction_label.replace(' * ', '_')}_tukey_full_report.csv",
        mime="text/csv",
        key=f"download_tukey_full_{interaction_label}"
    )


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

    st.sidebar.header("Effects (relative to baseline level)")
    coef_intercept = st.sidebar.slider('Baseline mean', -100.0, 100.0, 0.0)
    coef_medium = st.sidebar.slider(f'Effect of {level_names["medium"]}', -100.0, 100.0, 0.0)
    coef_high = st.sidebar.slider(f'Effect of {level_names["high"]}', -100.0, 100.0, 0.0)

    # Build data
    rng = np.random.default_rng(seed=seed or None)
    raw_levels = np.repeat(['low', 'medium', 'high'], replications)
    df = pd.DataFrame({factor_name: raw_levels})
    map_show = {'low': level_names['low'], 'medium': level_names['medium'], 'high': level_names['high']}
    df[factor_name] = df[factor_name].map(map_show)
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
        vals = list(np.asarray(df.loc[df[factor_name] == level, "Y"]).ravel())
        if not _is_finite_array(vals):
            st.error("Non-finite values in box plot.")
            return
        fig.add_trace(go.Box(y=vals, name=level, boxmean=True, showlegend=False))
    fig.update_layout(xaxis_title=factor_name, yaxis_title="Y", title="Distribution of Y across Levels", height=500)
    _safe_plot(fig)

    # OLS summary
    model = smf.ols(f"Y ~ C(Q('{factor_name}'))", data=df).fit() # Added Q() for safety
    st.subheader("Linear Regression Model Summary")
    render_model_summary(model, "OLS Regression Results")

    # Variance Decomposition pies
    st.subheader("Variance Decomposition (SST vs. SSTr & SSE)")
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Clean up index
    anova_table.index = anova_table.index.str.replace("C\(Q\('", "", regex=True).str.replace("'\)\)", "", regex=True)

    try:
        resid_row = anova_table.index.str.contains("Residual", case=False, regex=False)
        if not resid_row.any():
            raise ValueError("Residual row not found in ANOVA table.")
        sse = float(anova_table.loc[resid_row, 'sum_sq'].iloc[0])
    except Exception:
        sse = float(np.sum(np.square(model.resid)))

    y = np.asarray(df['Y'], dtype=float)
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    sstr = max(sst - sse, 0.0)

    if not all(map(np.isfinite, [sst, sstr, sse])) or sst <= 0:
        st.info("Variance pies unavailable (non-finite or zero SST).")
    else:
        pies = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=("Total Sum of Squares (SST)", "Treatment vs Error (SSTr vs SSE)")
        )
        pies.add_trace(go.Pie(labels=["SST"], values=[sst], textinfo='label+percent', hole=0.4),
                       row=1, col=1)
        if sstr > 0 or sse > 0:
            pies.add_trace(go.Pie(labels=["SSTr", "SSE"], values=[max(sstr, 0.0), max(sse, 0.0)],
                                  textinfo='label+percent', hole=0.4),
                           row=1, col=2)
        pies.update_layout(height=500, showlegend=False)
        _safe_plot(pies)

    # --- Post-hoc analysis (Tukey HSD) ---
    st.subheader("Post-hoc Analysis (Tukey HSD)")
    st.markdown(f"**Pairwise comparisons for {factor_name}**")
    tuk = pairwise_tukeyhsd(endog=df["Y"], groups=df[factor_name], alpha=0.05)
    
    # Create full dataframe for filtering and download
    tuk_df_full = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    
    # Show only significant differences
    st.caption("Significant Pairwise Differences (p < 0.05):")
    st.dataframe(tuk_df_full[tuk_df_full['reject'] == True])
    
    # Show grouped summary
    tukey_grouped_df = _format_tukey_summary_for_display(tuk, df, factor_name)
    st.markdown(f"**Grouping Summary for {factor_name}**")
    st.dataframe(tukey_grouped_df)
    st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

    st.download_button(
        label=f"Download Full {factor_name} Pairwise Report",
        data=tuk_df_full.to_csv(index=False).encode("utf-8"),
        file_name=f"{factor_name}_tukey_full_report.csv",
        mime="text/csv",
        key=f"download_tukey_full_{factor_name}"
    )


def Analysis():
    st.title("Tips to Analyze the Statistical Outputs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

    st.markdown("""
    This page demonstrates a **complete analysis workflow** using a realistic industrial engineering scenario:
    balancing **cycle times** across three machines (A/B/C) to improve line balance and throughput.
    We: (1) simulate data, (2) fit an OLS model, (3) examine ANOVA and **Tukey HSD** post-hoc,
    (4) check assumptions (normality & homoscedasticity), and (5) guide interpretation step-by-step.
    """)

    st.sidebar.header("Scenario & Simulation Controls")
    n_per_machine = st.sidebar.slider("Replications per machine", 10, 200, 40, 5)
    mean_A = st.sidebar.number_input("Mean cycle time — Machine A (sec)", 42.0, value=45.0, step=0.5)
    mean_B = st.sidebar.number_input("Mean cycle time — Machine B (sec)", 42.0, value=47.5, step=0.5)
    mean_C = st.sidebar.number_input("Mean cycle time — Machine C (sec)", 42.0, value=50.0, step=0.5)
    sd_A = st.sidebar.slider("Std. dev. — Machine A", 0.1, 10.0, 2.0, 0.1)
    sd_B = st.sidebar.slider("Std. dev. — Machine B", 0.1, 10.0, 2.2, 0.1)
    sd_C = st.sidebar.slider("Std. dev. — Machine C", 0.1, 10.0, 2.4, 0.1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    rng = np.random.default_rng(seed=seed or None)
    df = pd.DataFrame({
        "Machine": np.repeat(["A", "B", "C"], n_per_machine),
        "CycleTime": np.concatenate([
            rng.normal(mean_A, sd_A, n_per_machine),
            rng.normal(mean_B, sd_B, n_per_machine),
            rng.normal(mean_C, sd_C, n_per_machine),
        ])
    })

    st.subheader("Generated Data (Cycle Times by Machine)")
    st.dataframe(df.head(10))
    st.download_button("Download simulated dataset (CSV)",
                       data=df.to_csv(index=False),
                       file_name="IE_cycle_times_simulated.csv",
                       mime="text/csv")

    st.subheader("Distribution by Machine (Box Plot)")
    fig_box = go.Figure()
    for m in ["A", "B", "C"]:
        vals = list(np.asarray(df.loc[df["Machine"] == m, "CycleTime"]).ravel())
        if not _is_finite_array(vals):
            st.error("Non-finite values in box plot.")
            return
        fig_box.add_trace(go.Box(y=vals, name=f"Machine {m}", boxmean=True, showlegend=False))
    fig_box.update_layout(xaxis_title="Machine",
                          yaxis_title="Cycle Time (seconds)",
                          height=450,
                          title="Cycle Time Distribution across Machines")
    _safe_plot(fig_box)

    st.subheader("OLS Model: CycleTime ~ C(Machine)")
    model = smf.ols("CycleTime ~ C(Machine)", data=df).fit()
    render_model_summary(model, "OLS Regression Results")

    st.subheader("ANOVA (One-Way) — Text Output")
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    st.text(anova_tbl.to_string())

    eta_sq = np.nan
    try:
        resid = model.resid
        sse = float(np.sum(np.square(resid)))
        y = np.asarray(df['CycleTime'], dtype=float)
        ybar = float(np.mean(y))
        sst = float(np.sum((y - ybar) ** 2))
        sstr = max(sst - sse, 0.0)
        eta_sq = sstr / sst if sst > 0 else np.nan
        st.markdown(f"**Effect size (η²)**: {eta_sq:.3f}  — proportion of total variance explained by Machine.")
    except Exception:
        st.info("Could not compute η² from ANOVA table.")

    st.subheader("Post-hoc Comparisons: Tukey HSD")
    tukey = pairwise_tukeyhsd(endog=df["CycleTime"], groups=df["Machine"], alpha=0.05)
    
    # Create full dataframe for filtering and download
    tuk_df_full_analysis = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # Show only significant differences
    st.caption("Significant Pairwise Differences (p < 0.05):")
    st.dataframe(tuk_df_full_analysis[tuk_df_full_analysis['reject'] == True])

    # Special case for Analysis page, which uses 'CycleTime' and 'Machine'
    tukey_grouped_df_analysis = _format_tukey_summary_for_display(tukey, df, 'Machine', response_col='CycleTime')
    st.markdown("**Grouping Summary for Machine**")
    st.dataframe(tukey_grouped_df_analysis)
    st.caption("ℹ️ **How to read this table:** Groups (levels) that **share a letter** in the 'Grouping' column are **not** significantly different from each other.")

    st.download_button(
        label=f"Download Full Machine Pairwise Report",
        data=tuk_df_full_analysis.to_csv(index=False).encode("utf-8"),
        file_name=f"Machine_tukey_full_report.csv",
        mime="text/csv",
        key=f"download_tukey_full_Machine_analysis"
    )

    st.subheader("Model Assumptions")

    resid = model.resid
    sh_W, sh_p = stats.shapiro(resid)
    st.markdown(f"**Shapiro–Wilk (residuals)**: W = {sh_W:.3f}, p = {sh_p:.4f} "
                f"→ {'Fail to reject normality' if sh_p >= 0.05 else 'Potential non-normality'}")

    A_vals = df.loc[df["Machine"] == "A", "CycleTime"]
    B_vals = df.loc[df["Machine"] == "B", "CycleTime"]
    C_vals = df.loc[df["Machine"] == "C", "CycleTime"]
    lev_W, lev_p = stats.levene(A_vals, B_vals, C_vals, center='median')
    st.markdown(f"**Levene (homogeneity)**: W = {lev_W:.3f}, p = {lev_p:.4f} "
                f"→ {'Variances appear equal' if lev_p >= 0.05 else 'Variances may differ'}")

    st.subheader("Residual Diagnostics")

    fitted = model.fittedvalues
    fig_rvf = go.Figure()
    fig_rvf.add_trace(go.Scatter(x=list(np.asarray(fitted).ravel()),
                                  y=list(np.asarray(resid).ravel()),
                                  mode='markers', name='Residuals'))
    x_min = float(np.min(fitted))
    x_max = float(np.max(fitted))
    fig_rvf.add_shape(
        type="line",
        x0=x_min, x1=x_max, y0=0, y1=0,
        xref="x", yref="y",
        line=dict(dash="dash")
    )
    fig_rvf.update_layout(xaxis_title="Fitted values",
                          yaxis_title="Residuals",
                          height=420,
                          title="Residuals vs Fitted")
    _safe_plot(fig_rvf)

    osm, osr = stats.probplot(resid, dist="norm", sparams=(), fit=False)
    qq_x = np.array(osm, dtype=float)
    qq_y = np.array(osr, dtype=float)
    if np.allclose(np.std(qq_y), 0):
        line_y = np.full_like(qq_x, fill_value=qq_y[0], dtype=float)
    else:
        lr = stats.linregress(qq_x, qq_y)
        line_y = lr.intercept + lr.slope * qq_x

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=qq_x, y=qq_y, mode='markers', name='Residuals'))
    fig_qq.add_trace(go.Scatter(x=qq_x, y=line_y, mode='lines', name='Reference line'))
    fig_qq.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Residuals",
        height=420,
        title="Q–Q Plot of Residuals"
    )
    _safe_plot(fig_qq)

    st.subheader("How to Interpret These Results (Step-by-Step)")
    st.markdown(f"""
1) **Context** — We compare mean **cycle times** across three machines (A/B/C). Lower and more uniform cycle times
   support better **line balance** and throughput.

2) **Visual screening (box plot)** — Look for clear differences in medians and spread. If Machine C shows higher
   median and similar spread, it likely **bottlenecks** the line.

3) **Model fit (OLS summary)** — Focus on **R²/Adj. R²** (explained variance), and the **F-statistic p-value**:
   if p < 0.05, there is evidence that at least one machine's mean differs.

4) **ANOVA** — The `C(Machine)` row tests equality of means. If the **p-value** is < 0.05, proceed with post-hoc.

5) **Effect size (η²)** — Here η² ≈ {eta_sq:.3f} if computed. Values near 0.01/0.06/0.14 are often interpreted as
   small/medium/large (rule-of-thumb), but use domain judgment.

6) **Post-hoc (Tukey HSD)** — Pairs with `reject = True` differ significantly.
   Use these to identify which machines are **statistically slower** (e.g., C slower than A, B). The **Grouping Summary** table
   is the easiest way to see this: machines that *do not share a letter* are significantly different.

7) **Assumptions** — Shapiro–Wilk tests residual normality (we want p ≥ 0.05). Levene assesses equal variances
   (we want p ≥ 0.05). If violated, consider **transformations** (e.g., log) or **robust/ Welch ANOVA**.

8) **Diagnostics** — The residuals-vs-fitted plot should look **random** around zero (no patterns).
   The Q–Q plot should be roughly linear (normal residuals).

9) **Actionable conclusion** — If Machine C is significantly slower (e.g., it's in group 'A' and Machine A is in group 'B'), prioritize:
   - **SMED/Setup reduction** or **micro-motion** improvements on C
   - **Preventive maintenance** if downtime adds to cycle time variance
   - **Work redistribution** (balance stations upstream/downstream)
   - **Standard work** & operator training to reduce variability

10) **Monitoring** — After interventions, **re-sample** cycle times and rerun ANOVA to confirm improvement
    and sustained homogeneity of variances.
    """)

    st.subheader("Exports")
    st.download_button("Download Full Tukey HSD results (CSV)",
                       data=tuk_df_full_analysis.to_csv(index=False),
                       file_name="tukey_hsd_full_results.csv",
                       mime="text/csv")


# -------------------------
# MODIFIED PAGE: 2^(k-p) Fractional Factorial
# -------------------------
def fractional_factorial():
    st.title("Introduction to $2^{k-p}$ Fractional Factorial Designs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("""
A **full factorial** design ($2^k$) requires testing all possible combinations of $k$ factors at 2 levels.
This can quickly become too expensive.
- $2^3 = 8$ runs
- $2^4 = 16$ runs
- $2^5 = 32$ runs

A **fractional factorial** design ($2^{k-p}$) runs only a *fraction* (e.g., $1/2, 1/4$) of these experiments, sacrificing the ability to estimate certain interactions to save resources.
    """)

    st.sidebar.header("Design Controls ($2^{k-1}$)")
    design_choice = st.sidebar.selectbox(
        "Choose Design",
        ("$2^{3-1}$ (3 factors, 4 runs)", 
         "$2^{4-1}$ (4 factors, 8 runs)", 
         "$2^{5-1}$ (5 factors, 16 runs)"),
        index=0
    )

    design_df = pd.DataFrame()
    resolution_str = ""
    aliases_md = ""
    plot_title = ""
    fig = go.Figure()

    if design_choice.startswith("$2^{3-1}$"):
        k, p = 3, 1
        base_k = k - p
        base_factors = ['A', 'B']
        gen_factor = 'C'
        all_factors = ['A', 'B', 'C']
        
        generator = st.sidebar.radio("Generator", ("I = +ABC", "I = -ABC"), key="gen_3_1")
        sign = 1 if generator == "I = +ABC" else -1
        
        # 1. Build the fractional design
        design_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=base_k)), columns=base_factors)
        design_df[gen_factor] = sign * design_df['A'] * design_df['B']
        design_df = design_df[all_factors] # Reorder columns
        
        # 2. Build the full factorial for plotting
        full_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=k)), columns=all_factors)
        full_df['prod'] = full_df['A'] * full_df['B'] * full_df['C']
        
        selected_runs = full_df[full_df['prod'] == sign]
        alternate_runs = full_df[full_df['prod'] == -sign]

        # 3. Create the 3D plot
        fig.add_trace(go.Scatter3d(
            x=selected_runs['A'], y=selected_runs['B'], z=selected_runs['C'],
            mode='markers', marker=dict(size=10, color='blue', opacity=1.0),
            name=f"Principal Fraction ({generator})"
        ))
        fig.add_trace(go.Scatter3d(
            x=alternate_runs['A'], y=alternate_runs['B'], z=alternate_runs['C'],
            mode='markers', marker=dict(size=8, color='gray', opacity=0.2),
            name="Alternate Fraction (not run)"
        ))
        fig.update_layout(
            scene=dict(xaxis_title='Factor A', yaxis_title='Factor B', zaxis_title='Factor C',
                       xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
            height=550, legend_title="Fraction"
        )
        
        # 4. Analysis
        plot_title = f'Runs for $2^{3-1}$ Design (Selected: {generator})'
        resolution_str = "**Resolution III** (shortest word `ABC` has length 3). Main effects are aliased with two-way interactions."
        aliases_md = f"""
- **Defining Relation:** `{generator}`
- **Aliasing Structure:**
    - $I = {generator[4:]}$
    - $A = {generator[3]}BC$
    - $B = {generator[3]}AC$
    - $C = {generator[3]}AB$
"""

    elif design_choice.startswith("$2^{4-1}$"):
        k, p = 4, 1
        base_k = k - p
        base_factors = ['A', 'B', 'C']
        gen_factor = 'D'
        all_factors = ['A', 'B', 'C', 'D']
        
        generator = st.sidebar.radio("Generator", ("I = +ABCD", "I = -ABCD"), key="gen_4_1")
        sign = 1 if generator == "I = +ABCD" else -1

        # 1. Build the fractional design
        design_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=base_k)), columns=base_factors)
        design_df[gen_factor] = sign * design_df['A'] * design_df['B'] * design_df['C']
        design_df = design_df[all_factors]
        
        # 2. Build full factorial for plotting
        full_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=k)), columns=all_factors)
        full_df['prod'] = full_df['A'] * full_df['B'] * full_df['C'] * full_df['D']
        
        selected_runs = full_df[full_df['prod'] == sign]
        alternate_runs = full_df[full_df['prod'] == -sign]
        
        # 3. Create the 4D plot (as two 3D cubes)
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Cube: D = -1', 'Cube: D = +1')
        )
        
        # Plot for D = -1
        sel_d_neg = selected_runs[selected_runs['D'] == -1]
        alt_d_neg = alternate_runs[alternate_runs['D'] == -1]
        fig.add_trace(go.Scatter3d(
            x=sel_d_neg['A'], y=sel_d_neg['B'], z=sel_d_neg['C'],
            mode='markers', marker=dict(size=8, color='blue'),
            name="Selected Runs"
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=alt_d_neg['A'], y=alt_d_neg['B'], z=alt_d_neg['C'],
            mode='markers', marker=dict(size=6, color='gray', opacity=0.2),
            name="Alternate Runs"
        ), row=1, col=1)
        
        # Plot for D = +1
        sel_d_pos = selected_runs[selected_runs['D'] == 1]
        alt_d_pos = alternate_runs[alternate_runs['D'] == 1]
        fig.add_trace(go.Scatter3d(
            x=sel_d_pos['A'], y=sel_d_pos['B'], z=sel_d_pos['C'],
            mode='markers', marker=dict(size=8, color='blue'),
            showlegend=False # Hide redundant legend
        ), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=alt_d_pos['A'], y=alt_d_pos['B'], z=alt_d_pos['C'],
            mode='markers', marker=dict(size=6, color='gray', opacity=0.2),
            showlegend=False
        ), row=1, col=2)
        
        fig.update_layout(height=550, legend_title="Fraction")
        fig.update_scenes(
            xaxis_title='A', yaxis_title='B', zaxis_title='C',
            xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])
        )

        # 4. Analysis
        plot_title = f'Runs for $2^{4-1}$ Design (Selected: {generator})'
        resolution_str = "**Resolution IV** (shortest word `ABCD` has length 4). Main effects are clean, but two-way interactions are aliased with each other."
        aliases_md = f"""
- **Defining Relation:** `{generator}`
- **Aliasing Structure (Key Pairs):**
    - $I = {generator[4:]}$
    - $A = {generator[3]}BCD$
    - $B = {generator[3]}ACD$
    - $C = {generator[3]}ABD$
    - $D = {generator[3]}ABC$
    - $AB = {generator[3]}CD$
    - $AC = {generator[3]}BD$
    - $AD = {generator[3]}BC$
"""

    elif design_choice.startswith("$2^{5-1}$"):
        k, p = 5, 1
        base_k = k - p
        base_factors = ['A', 'B', 'C', 'D']
        gen_factor = 'E'
        all_factors = ['A', 'B', 'C', 'D', 'E']
        
        generator = st.sidebar.radio("Generator", ("I = +ABCDE", "I = -ABCDE"), key="gen_5_1")
        sign = 1 if generator == "I = +ABCDE" else -1

        # 1. Build the fractional design
        design_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=base_k)), columns=base_factors)
        design_df[gen_factor] = sign * design_df['A'] * design_df['B'] * design_df['C'] * design_df['D']
        design_df = design_df[all_factors]
        
        # 2. Build full factorial for plotting
        full_df = pd.DataFrame(list(itertools.product([-1, 1], repeat=k)), columns=all_factors)
        full_df['prod'] = full_df['A'] * full_df['B'] * full_df['C'] * full_df['D'] * full_df['E']
        
        selected_runs = full_df[full_df['prod'] == sign]
        alternate_runs = full_df[full_df['prod'] == -sign]
        
        # 3. Create the 5D plot (as two 3D cubes, with E as color/symbol)
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Cube: D = -1', 'Cube: D = +1')
        )
        
        # Plot for D = -1
        sel_d_neg = selected_runs[selected_runs['D'] == -1]
        alt_d_neg = alternate_runs[alternate_runs['D'] == -1]
        
        # D=-1, E=-1 (Selected)
        df_plot = sel_d_neg[sel_d_neg['E'] == -1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=8, color='blue', symbol='circle'), name="Selected, E = -1"
        ), row=1, col=1)
        # D=-1, E=+1 (Selected)
        df_plot = sel_d_neg[sel_d_neg['E'] == 1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=8, color='green', symbol='square'), name="Selected, E = +1"
        ), row=1, col=1)
        # D=-1, E=-1 (Alternate)
        df_plot = alt_d_neg[alt_d_neg['E'] == -1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=6, color='gray', opacity=0.2, symbol='circle'), name="Alternate, E = -1"
        ), row=1, col=1)
        # D=-1, E=+1 (Alternate)
        df_plot = alt_d_neg[alt_d_neg['E'] == 1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=6, color='gray', opacity=0.2, symbol='square'), name="Alternate, E = +1"
        ), row=1, col=1)

        # Plot for D = +1
        sel_d_pos = selected_runs[selected_runs['D'] == 1]
        alt_d_pos = alternate_runs[alternate_runs['D'] == 1]
        
        # D=+1, E=-1 (Selected)
        df_plot = sel_d_pos[sel_d_pos['E'] == -1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=8, color='blue', symbol='circle'), showlegend=False
        ), row=1, col=2)
        # D=+1, E=+1 (Selected)
        df_plot = sel_d_pos[sel_d_pos['E'] == 1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=8, color='green', symbol='square'), showlegend=False
        ), row=1, col=2)
        # D=+1, E=-1 (Alternate)
        df_plot = alt_d_pos[alt_d_pos['E'] == -1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=6, color='gray', opacity=0.2, symbol='circle'), showlegend=False
        ), row=1, col=2)
        # D=+1, E=+1 (Alternate)
        df_plot = alt_d_pos[alt_d_pos['E'] == 1]
        fig.add_trace(go.Scatter3d(
            x=df_plot['A'], y=df_plot['B'], z=df_plot['C'], mode='markers',
            marker=dict(size=6, color='gray', opacity=0.2, symbol='square'), showlegend=False
        ), row=1, col=2)

        fig.update_layout(height=600, legend_title="Fraction & Factor E")
        fig.update_scenes(
            xaxis_title='A', yaxis_title='B', zaxis_title='C',
            xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])
        )

        # 4. Analysis
        plot_title = f'Runs for $2^{5-1}$ Design (Selected: {generator})'
        resolution_str = "**Resolution V** (shortest word `ABCDE` has length 5). Main effects and two-way interactions are both clean (not aliased with each other)."
        aliases_md = f"""
- **Defining Relation:** `{generator}`
- **Aliasing Structure (Key Pairs):**
    - $I = {generator[4:]}$
    - $A = {generator[3]}BCDE$
    - $B = {generator[3]}ACDE$
    ... (Main effects are aliased with 4-way interactions)
    - $AB = {generator[3]}CDE$
    - $AC = {generator[3]}BDE$
    ... (Two-way interactions are aliased with 3-way interactions)
"""
    
    # --- Display Common Elements ---
    
    st.subheader("Factor Space Visualization")
    st.markdown(plot_title)
    _safe_plot(fig)
    st.markdown(f"The design matrix below shows the **{len(design_df)} selected runs**.")

    st.subheader("Design Matrix")
    st.dataframe(design_df.reset_index(drop=True))

    st.subheader("Design Analysis: Resolution & Aliasing")
    st.markdown("""
    **What is Resolution?**

    The **Resolution** of a design is a number (e.g., III, IV, V) that describes the degree of aliasing (confounding). It's a critical measure of the design's quality.

    - **Calculation:** The Resolution is the length of the *shortest "word"* in the design's **Defining Relation**.
    - A "word" is the product of letters (e.g., `ABC`, `ABCD`).
    - The "length" is the number of letters in the word (e.g., `ABC` has length 3).

    **Common Resolutions:**
    - **Resolution III:** The shortest word has length 3 (e.g., `I = ABC`). This is the lowest practical resolution.
        - **Consequence:** Main effects (like `A`) are aliased with two-way interactions (like `BC`).
        - **Use:** Good for *screening* many factors when you can assume interactions are negligible.
    - **Resolution IV:** The shortest word has length 4 (e.g., `I = ABCD`).
        - **Consequence:** Main effects are *not* aliased with two-way interactions (they are aliased with 3-way interactions, which are often ignored). Two-way interactions *are* aliased with other two-way interactions (e.g., `AB = CD`).
        - **Use:** A very popular and efficient design.
    - **Resolution V:** The shortest word has length 5 (e.g., `I = ABCDE`).
        - **Consequence:** Main effects are "clean" (aliased with 4-way interactions). Two-way interactions are also "clean" (aliased with 3-way interactions).
        - **Use:** A very high-quality design, great for estimating main effects and two-way interactions.
    """)
    
    st.info(f"**This design is {resolution_str}**")
    st.markdown(aliases_md)


# -------------------------
# NEW PAGE: Choice-Based Conjoint Analysis
# -------------------------
def conjoint_analysis():
    st.title("Conjoint Analysis (Choice-Based) — Shoe Components")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("""
In this exercise, students evaluate **pairwise alternatives** for a shoe made up of two attributes:

- **Upper Material** *(3 levels)* - **Sole Type** *(2 levels)*

Choose the preferred option in each task. After all choices are recorded, the app fits a **logistic regression (Logit)** to
estimate how each attribute level influences choice probability (part-worth utilities).
    """)

    # ---------- Sidebar controls ----------
    st.sidebar.header("Conjoint Settings")
    attr1_name = st.sidebar.text_input("Attribute 1 name (3 levels)", "Upper Material")
    attr1_levels = st.sidebar.text_input("Attribute 1 levels (comma)", "Leather,Synthetic,Mesh")
    attr1_levels = [s.strip() for s in attr1_levels.split(",") if s.strip()]
    if len(attr1_levels) != 3:
        st.warning("Attribute 1 must have exactly **3** levels. Using defaults: Leather, Synthetic, Mesh.")
        attr1_levels = ["Leather", "Synthetic", "Mesh"]

    attr2_name = st.sidebar.text_input("Attribute 2 name (2 levels)", "Sole Type")
    attr2_levels = st.sidebar.text_input("Attribute 2 levels (comma)", "Cushioned,Minimal")
    attr2_levels = [s.strip() for s in attr2_levels.split(",") if s.strip()]
    if len(attr2_levels) != 2:
        st.warning("Attribute 2 must have exactly **2** levels. Using defaults: Cushioned, Minimal.")
        attr2_levels = ["Cushioned", "Minimal"]

    n_tasks = st.sidebar.slider("Number of pairwise choice tasks", 3, 20, 8, 1)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=1, step=1)

    # ---------- Build full factorial profiles ----------
    profiles = pd.DataFrame(list(itertools.product(attr1_levels, attr2_levels)),
                            columns=[attr1_name, attr2_name])
    profiles["ProfileID"] = np.arange(1, len(profiles)+1)

    st.subheader("All Possible Alternatives (Profiles)")
    st.dataframe(profiles)

    # ---------- Create choice sets (2 profiles per task) ----------
    rng = np.random.default_rng(seed or None)
    # Build fixed pairs for reproducibility based on seed
    pairs = []
    used = set()
    for t in range(n_tasks):
        a, b = rng.choice(profiles["ProfileID"], size=2, replace=False)
        pairs.append((int(a), int(b)))
    choice_sets = pd.DataFrame(pairs, columns=["A_ProfileID", "B_ProfileID"])
    choice_sets.index.name = "Task"
    choice_sets.reset_index(inplace=True)
    choice_sets["Task"] += 1

    # Store in session to persist student picks
    if "conjoint_pairs" not in st.session_state or st.session_state.get("conjoint_seed") != seed or st.session_state.get("conjoint_tasks") != n_tasks:
        st.session_state["conjoint_pairs"] = choice_sets
        st.session_state["conjoint_choices"] = {}  # task -> "A" or "B"
        st.session_state["conjoint_seed"] = seed
        st.session_state["conjoint_tasks"] = n_tasks

    st.subheader("Choice Tasks — **Select the Best Option**")
    st.caption("For each task, compare alternatives A and B and choose the preferred one.")

    # Render each task
    for _, row in st.session_state["conjoint_pairs"].iterrows():
        task = int(row["Task"])
        a_id = int(row["A_ProfileID"])
        b_id = int(row["B_ProfileID"])
        a = profiles.loc[profiles.ProfileID == a_id, [attr1_name, attr2_name]].iloc[0]
        b = profiles.loc[profiles.ProfileID == b_id, [attr1_name, attr2_name]].iloc[0]

        c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
        with c1:
            st.markdown(f"**Task {task} — Option A**")
            st.table(pd.DataFrame(a).T.rename(index={0: f"A (ID {a_id})"}))
        with c2:
            st.markdown(f"**Task {task} — Option B**")
            st.table(pd.DataFrame(b).T.rename(index={0: f"B (ID {b_id})"}))
        with c3:
            st.radio("Your choice", options=["A", "B"], key=f"choice_task_{task}",
                     index=0 if st.session_state["conjoint_choices"].get(task) == "A" else
                     (1 if st.session_state["conjoint_choices"].get(task) == "B" else 0))
            # Persist
            st.session_state["conjoint_choices"][task] = st.session_state[f"choice_task_{task}"]

        st.markdown("---")

    # ---------- Build estimation dataset once student has chosen ----------
    choices = st.session_state["conjoint_choices"]
    all_answered = (len(choices) == n_tasks) and all(c in ("A", "B") for c in choices.values())

    st.subheader("Estimation Dataset (long format)")
    if not all_answered:
        st.info("Please complete all choices above to enable estimation.")
        return

    # Construct long dataset: each task produces two rows
    rows = []
    for _, row in st.session_state["conjoint_pairs"].iterrows():
        task = int(row["Task"])
        a_id = int(row["A_ProfileID"])
        b_id = int(row["B_ProfileID"])
        choice = choices[task]
        for alt, pid in zip(["A", "B"], [a_id, b_id]):
            prof = profiles.loc[profiles.ProfileID == pid, [attr1_name, attr2_name]].iloc[0]
            chosen = 1 if alt == choice else 0
            rows.append({
                "Task": task,
                "Alt": alt,
                "ProfileID": pid,
                attr1_name: prof[attr1_name],
                attr2_name: prof[attr2_name],
                "Chosen": chosen
            })
    long_df = pd.DataFrame(rows).sort_values(["Task", "Alt"]).reset_index(drop=True)
    st.dataframe(long_df)

    st.download_button("Download choice data (CSV)",
                       data=long_df.to_csv(index=False),
                       file_name="conjoint_choice_data.csv",
                       mime="text/csv")

    # ---------- Fit Logit ----------
    st.subheader("Logistic Regression (Choice ~ Attributes)")
    # Dummy-code attributes (reference: first level of each)
    X = pd.get_dummies(long_df[[attr1_name, attr2_name]], drop_first=True)
    X = sm.add_constant(X, has_constant="add")

    # Explicitly add the intercept column if it was dropped (e.g. if X was empty after drop_first=True)
    if 'const' not in X.columns:
        X.insert(0, 'const', 1.0)
        
    y = long_df["Chosen"].astype(int)

    # Explicitly cast X to float to avoid "dtype=object" error in statsmodels
    X = X.astype(float)

    # Guard against degenerate data (e.g., same choice always producing complete separation)
    if y.sum() == 0 or y.sum() == len(y):
        st.error("All choices are identical (all 0 or all 1). At least one task must have the other alternative chosen.")
        return
    if X.shape[1] == 1 and 'const' in X.columns: # Only intercept remains
        st.error("No attribute variation after encoding. Adjust attributes or tasks.")
        return

    try:
        logit_model = sm.Logit(y, X).fit(disp=False, maxiter=100) # Increased maxiter for robustness
        render_model_summary(logit_model, "Logit Regression Results")
    except Exception as e:
        st.error(f"Logit failed to converge: {e}")
        return

    # ---------- Odds ratios & CIs ----------
    st.subheader("Odds Ratios (exp(coef)) with 95% CI")
    params = logit_model.params
    conf = logit_model.conf_int()
    or_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
        "p_value": logit_model.pvalues.values
    })
    # Hide constant in the OR table for clarity
    or_df = or_df[or_df["term"] != "const"].reset_index(drop=True)
    st.dataframe(or_df)

    st.download_button("Download Odds Ratios (CSV)",
                       data=or_df.to_csv(index=False),
                       file_name="conjoint_odds_ratios.csv",
                       mime="text/csv")

    # ---------- Quick interpretation ----------
    st.subheader("Quick Interpretation")
    st.markdown(f"""
- Reference levels (baseline utilities): **{attr1_name} = {attr1_levels[0]}**, **{attr2_name} = {attr2_levels[0]}**.  
- A positive coefficient / odds ratio **> 1** means that level **increases** the probability of being chosen versus its reference.  
- A negative coefficient / odds ratio **< 1** means that level **decreases** the probability of being chosen versus its reference.  
- Use **p-values** (or confidence intervals not crossing 1) to judge statistical significance.
    """)



def missing_data_case_studies():
    st.title("Practice Cases: Missing-Data DOE Scenarios")
    st.markdown("""
Use these downloadable hypothetical datasets to practice real-world troubleshooting when runs are missing.
Each concept includes 3 cases and an analysis guidance.
""")

    def _download_case(df_case, label, key):
        st.download_button(
            label=f"Download {label} (CSV)",
            data=df_case.to_csv(index=False),
            file_name=f"{label.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=key
        )

    def _context_box(title, bullets):
        st.markdown(f"**Context — {title}**")
        st.markdown("\n".join([f"- {b}" for b in bullets]))

    # -------------------------
    # Concept 1: One-way ANOVA
    # -------------------------
    base_anova = pd.DataFrame({
        "Replication": np.tile(np.arange(1, 7), 3),
        "FactorLevel": np.repeat(["Low", "Medium", "High"], 6),
        "Y": [10.2, 9.8, 10.5, 10.1, 9.9, 10.3,
              11.4, 11.7, 11.5, 11.3, 11.8, 11.6,
              13.0, 12.8, 13.4, 13.1, 12.9, 13.3]
    })

    with st.expander("Concept 1 — One-Way ANOVA (3 missing-data cases)", expanded=False):
        _context_box(
            "Industrial Process Yield (One-way ANOVA)",
            [
                "**Experimental units:** independent production batches",
                "**Aim:** test whether mean yield differs across temperature levels",
                "**Factor (IV):** Temperature level (Low / Medium / High)",
                "**Replications:** 6 per level (nominal)",
                "**Dependent variable (Y):** Yield (%)",
                "**Model:** fixed-effects one-way ANOVA"
            ]
        )

        case1 = base_anova.copy()
        case1.loc[(case1["FactorLevel"] == "Medium") & (case1["Replication"] == 4), "Y"] = np.nan
        st.markdown("**Case 1: One isolated missing observation (Medium, Rep 4).**")
        st.caption("Missingness pattern: one batch result not recorded (single NA).")
        st.dataframe(case1.head(12))
        _download_case(case1, "anova_case_1_single_missing", "anova_case_1")
        st.caption("Guideline: fit ANOVA with listwise deletion and compare with simple imputation sensitivity (mean/median).")

        case2 = base_anova[base_anova["Replication"] != 6].copy()
        st.markdown("**Case 2: Entire replication 6 disappeared for all levels.**")
        st.caption("Missingness pattern: one full production day lost for all levels (complete replication removed).")
        st.dataframe(case2.head(12))
        _download_case(case2, "anova_case_2_replication6_missing", "anova_case_2")
        st.caption("Guideline: SST must be recomputed with new N. Design remains balanced but with fewer observations.")

        case3 = base_anova.copy()
        case3.loc[(case3["FactorLevel"] == "High") & (case3["Replication"].isin([2, 5])), "Y"] = np.nan
        st.markdown("**Case 3: Two missing values concentrated in High level.**")
        st.caption("Missingness pattern: clustered NAs in one factor level → unbalanced cells.")
        st.dataframe(case3.head(12))
        _download_case(case3, "anova_case_3_clustered_missing", "anova_case_3")
        st.caption("Guideline: unbalanced cells change Type I/II/III SS behavior; report which SS type you use.")

    # -------------------------
    # Concept 2: Randomized Blocks
    # -------------------------
    block_rows = []
    rng = np.random.default_rng(123)
    for b in range(1, 7):
        for t, mu in zip(["A", "B", "C"], [50, 53, 56]):
            block_rows.append({"Block": b, "Treatment": t, "Y": round(float(rng.normal(mu + 0.4*b, 0.7)), 2)})
    base_blocks = pd.DataFrame(block_rows)

    with st.expander("Concept 2 — Randomized Complete Block Design (3 missing-data cases)", expanded=False):
        _context_box(
            "Agricultural Field Trial (RCBD)",
            [
                "**Experimental units:** field plots",
                "**Aim:** estimate fertilizer effects while controlling soil heterogeneity",
                "**Blocking factor:** Block (1–6; soil gradient/field sections)",
                "**Treatment factor (IV):** Fertilizer (A / B / C)",
                "**Dependent variable (Y):** Yield (kg/plot)",
                "**Model:** RCBD ANOVA / GLM with Block + Treatment"
            ]
        )

        b1 = base_blocks.copy()
        b1.loc[(b1["Block"] == 4) & (b1["Treatment"] == "B"), "Y"] = np.nan
        st.markdown("**Case 1: One missing treatment within one block (B at Block 4).**")
        st.caption("Missingness pattern: one plot failure inside one block → incomplete block.")
        st.dataframe(b1.head(12))
        _download_case(b1, "blocks_case_1_single_cell_missing", "blocks_case_1")
        st.caption("Guideline: RCBD is no longer complete; consider GLM (block+treatment) and appropriate missing handling.")

        b2 = base_blocks[base_blocks["Block"] != 6].copy()
        st.markdown("**Case 2: Entire Block 6 disappeared.**")
        st.caption("Missingness pattern: entire block lost (e.g., flooding) → fewer blocks.")
        st.dataframe(b2.head(12))
        _download_case(b2, "blocks_case_2_block6_missing", "blocks_case_2")
        st.caption("Guideline: adjust total df and recompute SS totals using remaining blocks only.")

        b3 = base_blocks.copy()
        b3.loc[(b3["Block"].isin([2, 5])) & (b3["Treatment"] == "C"), "Y"] = np.nan
        st.markdown("**Case 3: Patterned missingness for one treatment across two blocks.**")
        st.caption("Missingness pattern: treatment-specific missingness across blocks → potential bias; check mechanism.")
        st.dataframe(b3.head(12))
        _download_case(b3, "blocks_case_3_pattern_missing", "blocks_case_3")
        st.caption("Guideline: check robustness under mixed-model or multiple-imputation sensitivity; report assumptions (MCAR/MAR/MNAR).")

    # -------------------------
    # Concept 3: 2×2 Factorial
    # -------------------------
    fac_rows = []
    rng2 = np.random.default_rng(777)
    for r in range(1, 7):
        for a in ["Low", "High"]:
            for b in ["Low", "High"]:
                mu = 20 + (2 if a == "High" else 0) + (1.5 if b == "High" else 0) + (1.2 if (a == "High" and b == "High") else 0)
                fac_rows.append({"Replication": r, "FactorA": a, "FactorB": b, "Y": round(float(rng2.normal(mu, 0.5)), 2)})
    base_fac = pd.DataFrame(fac_rows)

    with st.expander("Concept 3 — Two-Factor Factorial 2×2 (3 missing-data cases)", expanded=False):
        _context_box(
            "Process Optimization Study (2×2 Factorial)",
            [
                "**Experimental units:** independent experimental runs",
                "**Aim:** estimate main effects and interaction (A×B) under missing runs",
                "**Factor A (IV):** Pressure (Low / High)",
                "**Factor B (IV):** Catalyst type (Low / High)",
                "**Replications:** 6 per (A,B) cell (nominal)",
                "**Dependent variable (Y):** Output performance index",
                "**Model:** two-way fixed-effects factorial ANOVA (A + B + A×B)"
            ]
        )

        f1 = base_fac.copy()
        f1.loc[(f1["Replication"] == 3) & (f1["FactorA"] == "High") & (f1["FactorB"] == "High"), "Y"] = np.nan
        st.markdown("**Case 1: One corner cell missing in one replication (A=High, B=High, Rep 3).**")
        st.caption("Missingness pattern: one missing run in a single treatment combination (corner).")
        st.dataframe(f1.head(12))
        _download_case(f1, "factorial_case_1_corner_missing", "fac_case_1")
        st.caption("Guideline: interaction estimate remains possible, but precision drops; check residual diagnostics carefully.")

        f2 = base_fac[base_fac["Replication"] != 6].copy()
        st.markdown("**Case 2: Entire replication 6 missing for all 4 treatment combinations.**")
        st.caption("Missingness pattern: one full replication lost (all combinations).")
        st.dataframe(f2.head(12))
        _download_case(f2, "factorial_case_2_replication6_missing", "fac_case_2")
        st.caption("Guideline: all SS components must use updated N and updated grand mean from remaining runs.")

        f3 = base_fac.copy()
        f3.loc[(f3["FactorA"] == "Low") & (f3["FactorB"] == "High") & (f3["Replication"].isin([2, 5])), "Y"] = np.nan
        st.markdown("**Case 3: Two missing runs from one treatment combination (A=Low, B=High).**")
        st.caption("Missingness pattern: two missing runs in one cell → unbalanced factorial cell counts.")
        st.dataframe(f3.head(12))
        _download_case(f3, "factorial_case_3_combination_missing", "fac_case_3")
        st.caption("Guideline: use Type II/III SS and explicitly state missing-data mechanism assumptions (MCAR/MAR/MNAR).")

    st.info("Tip for students: Start with visual missingness checks, then compare complete-case ANOVA/OLS against a sensitivity method before final conclusions.")


# -------------------------
# Navigation
# -------------------------
PAGES = {
    "Anova One-way - Introduction": anova_oneway,
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial Designs with Three Factors and Three Levels": three_factorial,
    "Fractional Factorial Designs (2^k-p)": fractional_factorial,
    "Conjoint Analysis (Choice-Based)": conjoint_analysis,
    "Tips to Analyze the Statistical Outputs": Analysis,
    "Practice Cases: Missing Data": missing_data_case_studies,
}

st.title('Navigation')
choice = st.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
