# =========================
# Factorial/ANOVA Streamlit App (clean & robust) — Py3.9 safe
# =========================
import sys
import itertools
from itertools import combinations

sys.dont_write_bytecode = True

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

def _to_bool(value):
    """Robust boolean conversion for tables where booleans may come as strings."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(value)

def _format_generic_grouping_summary(data_df, group_col, comparisons_df, response_col='Y', reject_col='reject'):
    """Build grouping letters from a generic pairwise comparison table."""
    group_stats = data_df.groupby(group_col)[response_col].agg(['mean', 'count']).reset_index()
    group_stats.rename(columns={'mean': 'Mean', 'count': 'N', group_col: 'Group'}, inplace=True)
    group_stats['Mean'] = group_stats['Mean'].round(2)

    comp_df = comparisons_df.copy()
    comp_df[reject_col] = comp_df[reject_col].map(_to_bool)
    sig_df = comp_df[comp_df[reject_col]].copy()
    sorted_groups = group_stats.sort_values('Mean', ascending=False)['Group'].tolist()
    group_letters = {group: [] for group in sorted_groups}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_groups = []
    current_letter_index = 0

    for i, group1 in enumerate(sorted_groups):
        is_grouped = any(group1 in letter_group for letter_group in letter_groups)
        if is_grouped:
            continue
        new_letter_group = {group1}
        for group2 in sorted_groups[i + 1:]:
            is_non_significant_with_all = True
            for member in new_letter_group:
                is_diff = not sig_df[
                    ((sig_df['group1'] == group2) & (sig_df['group2'] == member)) |
                    ((sig_df['group1'] == member) & (sig_df['group2'] == group2))
                ].empty
                if is_diff:
                    is_non_significant_with_all = False
                    break
            if is_non_significant_with_all:
                new_letter_group.add(group2)
        letter_groups.append(new_letter_group)

    letter_groups.sort(key=lambda g: min(sorted_groups.index(m) for m in g))
    for letter_group in letter_groups:
        letter = letters[current_letter_index]
        for group in letter_group:
            group_letters[group].append(letter)
        current_letter_index += 1

    group_stats['Grouping'] = group_stats['Group'].map(lambda g: "".join(group_letters[g]))
    return group_stats.sort_values('Mean', ascending=False).reset_index(drop=True)

def _pairwise_matrix_from_results(comparisons_df, ordered_groups):
    """Return a symmetric Sig/NS matrix from pairwise results."""
    matrix = pd.DataFrame("", index=ordered_groups, columns=ordered_groups)
    for g in ordered_groups:
        matrix.loc[g, g] = "-"
    for _, row in comparisons_df.iterrows():
        g1 = row['group1']
        g2 = row['group2']
        label = "Sig" if _to_bool(row['reject']) else "NS"
        matrix.loc[g1, g2] = label
        matrix.loc[g2, g1] = label
    return matrix

def _oneway_anova_with_error_terms(df, group_col, response_col='Y'):
    """Fit one-way ANOVA model and return model, ANOVA table, MSE, and residual df."""
    model = smf.ols(f"{response_col} ~ C(Q('{group_col}'))", data=df).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    resid_idx = anova_tbl.index.str.contains("Residual", case=False, regex=False)
    if not resid_idx.any():
        raise ValueError("Residual row not found in ANOVA table.")
    sse = float(anova_tbl.loc[resid_idx, 'sum_sq'].iloc[0])
    df_error = float(anova_tbl.loc[resid_idx, 'df'].iloc[0])
    mse = sse / df_error if df_error > 0 else np.nan
    return model, anova_tbl, mse, df_error

def _compute_lsd_pairwise(df, group_col, response_col='Y', alpha=0.05, mse=None, df_error=None):
    """Fisher's LSD pairwise tests using pooled ANOVA MSE."""
    if mse is None or df_error is None:
        _, _, mse, df_error = _oneway_anova_with_error_terms(df, group_col, response_col)

    gstats = df.groupby(group_col)[response_col].agg(['mean', 'count']).reset_index()
    rows = []
    for i in range(len(gstats)):
        for j in range(i + 1, len(gstats)):
            gi = gstats.iloc[i]
            gj = gstats.iloc[j]
            diff = float(gi['mean'] - gj['mean'])
            se = np.sqrt(mse * (1.0 / float(gi['count']) + 1.0 / float(gj['count'])))
            t_stat = diff / se if se > 0 else np.nan
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_error)) if np.isfinite(t_stat) else np.nan
            t_crit = stats.t.ppf(1 - alpha / 2, df=df_error) if df_error > 0 else np.nan
            lsd_crit = t_crit * se if np.isfinite(t_crit) and np.isfinite(se) else np.nan
            reject = bool(abs(diff) > lsd_crit) if np.isfinite(lsd_crit) else False
            rows.append({
                'group1': gi[group_col],
                'group2': gj[group_col],
                'mean_diff': round(diff, 4),
                'SE': round(float(se), 4) if np.isfinite(se) else np.nan,
                't_stat': round(float(t_stat), 4) if np.isfinite(t_stat) else np.nan,
                'p_value': round(float(p_val), 6) if np.isfinite(p_val) else np.nan,
                'LSD_critical_diff': round(float(lsd_crit), 4) if np.isfinite(lsd_crit) else np.nan,
                'reject': reject
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[['group1', 'group2', 'mean_diff', 'SE', 't_stat', 'p_value', 'LSD_critical_diff', 'reject']]
    return out, mse, df_error

def _compute_duncan_pairwise(df, group_col, response_col='Y', alpha=0.05, mse=None, df_error=None):
    """
    Duncan's Multiple Range Test using rank distance r and studentized range.
    """
    if not hasattr(stats, 'studentized_range'):
        raise RuntimeError("scipy.stats.studentized_range is unavailable in this environment.")

    if mse is None or df_error is None:
        _, _, mse, df_error = _oneway_anova_with_error_terms(df, group_col, response_col)

    gstats = df.groupby(group_col)[response_col].agg(['mean', 'count']).reset_index()
    gstats = gstats.sort_values('mean', ascending=False).reset_index(drop=True)
    rows = []
    for i in range(len(gstats)):
        for j in range(i + 1, len(gstats)):
            gi = gstats.iloc[i]
            gj = gstats.iloc[j]
            r = j - i + 1
            diff = float(gi['mean'] - gj['mean'])
            se_q = np.sqrt((mse / 2.0) * (1.0 / float(gi['count']) + 1.0 / float(gj['count'])))
            q_stat = abs(diff) / se_q if se_q > 0 else np.nan
            alpha_r = 1 - (1 - alpha) ** (r - 1)
            q_crit = stats.studentized_range.ppf(1 - alpha_r, r, df_error) if df_error > 0 else np.nan
            p_val = 1 - stats.studentized_range.cdf(q_stat, r, df_error) if np.isfinite(q_stat) else np.nan
            reject = bool(q_stat > q_crit) if np.isfinite(q_stat) and np.isfinite(q_crit) else False
            rows.append({
                'group1': gi[group_col],
                'group2': gj[group_col],
                'range_r': int(r),
                'mean_diff': round(diff, 4),
                'q_stat': round(float(q_stat), 4) if np.isfinite(q_stat) else np.nan,
                'q_critical': round(float(q_crit), 4) if np.isfinite(q_crit) else np.nan,
                'p_value': round(float(p_val), 6) if np.isfinite(p_val) else np.nan,
                'reject': reject
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[['group1', 'group2', 'range_r', 'mean_diff', 'q_stat', 'q_critical', 'p_value', 'reject']]
    return out, mse, df_error


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
        anova_table.index = anova_table.index.str.replace(r"C\(Q\('", "", regex=True).str.replace(r"'\)\)", "", regex=True)
        
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
        anova_table.index = anova_table.index.str.replace(r"C\(Q\('", "", regex=True).str.replace(r"'\)\)", "", regex=True)

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
    anova_table.index = anova_table.index.str.replace(r"C\(Q\('", "", regex=True).str.replace(r"'\)\)", "", regex=True)

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


def introduction_profile():
    st.title("Introduction")
    st.header("Welcome to the DOE Interactive App")
    st.markdown(
        """
        This interactive application is designed to provide a hands-on understanding of
        Design of Experiments (DOE), ANOVA, contrasts, factorial structures, and post-hoc analysis.

        Use the navigation panel to explore each module and practice with live simulations
        and downloadable datasets.
        """
    )

    st.subheader("About the Author")
    col1, col2 = st.columns([1, 3])

    with col1:
        try:
            st.image("image_9095e1.jpeg")
        except FileNotFoundError:
            st.error("Profile image not found. Make sure 'image_9095e1.jpeg' is in your repo.")

    with col2:
        st.markdown(
            """
            **Leonardo H. Talero-Sarmiento** is a Ph.D. in Engineering from the
            Universidad Autónoma de Bucaramanga, with expertise in mathematical modeling,
            data analytics, operations research, manufacturing systems, process
            improvement, and technology adoption.

            His research addresses decision-making and production-planning challenges in
            agricultural and industrial contexts, applying operations research,
            bibliometric analysis, systematic reviews, and machine-learning methods
            to strengthen value-chain resilience, optimize healthcare delivery,
            and drive digital transformation.
            """
        )


def contrast_analysis():
    st.title("Contrast Analysis and Relationship with Factors")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("""
This page computes a planned contrast from one-way ANOVA treatment means and shows how each factor level
contributes to that contrast.
    """)

    st.sidebar.header("Simulation controls")
    n_treatments = st.sidebar.slider("Number of treatment levels", 3, 6, 4, 1)
    replications = st.sidebar.slider("Replications per treatment", 3, 100, 20, 1)
    noise_sd = st.sidebar.slider("Within-treatment sigma", 0.0, 10.0, 1.0, 0.1)
    alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="contrast_alpha")
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=7, step=1, key="contrast_seed")

    st.sidebar.header("Factor and treatment labels")
    factor_name = st.sidebar.text_input("Factor name", "Factor", key="contrast_factor_name")
    raw_levels = []
    for i in range(n_treatments):
        raw_levels.append(st.sidebar.text_input(f"Treatment {i+1} label", f"Level {i+1}", key=f"contrast_level_{i}"))
    level_names = [lv.strip() if lv and lv.strip() else f"Level {i+1}" for i, lv in enumerate(raw_levels)]
    if len(set(level_names)) != len(level_names):
        st.error("Treatment labels must be unique.")
        return

    st.sidebar.header("Treatment means")
    means_map = {}
    for lv in level_names:
        means_map[lv] = st.sidebar.slider(
            f"Mean for {lv}", -100.0, 100.0, 10.0 + 5.0 * len(means_map), 0.1, key=f"contrast_mean_{lv}"
        )

    rng = np.random.default_rng(seed=seed or None)
    df = pd.DataFrame({factor_name: np.repeat(level_names, replications)})
    df["Y"] = df[factor_name].map(means_map).astype(float) + rng.normal(0, noise_sd, len(df))

    st.subheader("Generated Data")
    st.dataframe(df)
    st.download_button(
        "Download generated contrast data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="contrast_analysis_data.csv",
        mime="text/csv",
        key="download_contrast_data"
    )

    _, anova_tbl, mse, df_error = _oneway_anova_with_error_terms(df, group_col=factor_name, response_col="Y")
    anova_view = anova_tbl.copy().round(4)
    anova_view.index = anova_view.index.str.replace(r"C\(Q\('", "", regex=True).str.replace(r"'\)\)", "", regex=True)
    gstats = (
        df.groupby(factor_name)["Y"]
        .agg(N="size", Mean="mean", SD="std")
        .assign(SE=lambda d: d["SD"] / np.sqrt(d["N"]))
        .reset_index()
    )

    st.subheader("Core ANOVA Results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Treatment summary**")
        st.dataframe(gstats.round(4))
    with c2:
        st.markdown("**ANOVA table**")
        st.dataframe(anova_view)
        st.caption(f"Pooled error terms: MSE = {mse:.4f}, df_error = {df_error:.0f}")

    st.subheader("Contrast Definition")
    st.markdown("A valid contrast in one-way ANOVA satisfies:")
    st.latex(r"\Gamma=\sum_{i=1}^{a} c_i\mu_i,\quad \sum_{i=1}^{a} c_i=0")
    st.latex(r"C=\sum_{i=1}^{a} c_i\bar{y}_i,\quad SE(C)=\sqrt{MSE\sum_{i=1}^{a}\frac{c_i^2}{n_i}}")
    st.latex(r"t_0=\frac{C}{SE(C)},\quad CI_{1-\alpha}: C\pm t_{\alpha/2,\nu}SE(C)")

    st.markdown("**Enter contrast coefficients (`c_i`) by treatment level**")
    default_coefs = [1.0, 1.0, -1.0, -1.0] if n_treatments == 4 else [1.0, -1.0] + [0.0] * (n_treatments - 2)
    coef_cols = st.columns(min(4, n_treatments))
    coef_map = {}
    for i, lv in enumerate(level_names):
        col = coef_cols[i % len(coef_cols)]
        with col:
            coef_map[lv] = st.number_input(
                f"c({lv})", value=float(default_coefs[i]), step=0.5, key=f"contrast_coef_{i}_{lv}"
            )

    coef_series = pd.Series(coef_map, dtype=float)
    mean_series = gstats.set_index(factor_name)["Mean"].reindex(level_names).astype(float)
    n_series = gstats.set_index(factor_name)["N"].reindex(level_names).astype(float)

    sum_c = float(coef_series.sum())
    sum_nc = float((coef_series * n_series).sum())
    if abs(sum_c) > 1e-10:
        st.warning(f"Not a strict contrast: sum(c_i) = {sum_c:.6f}. Set coefficients so they sum to zero.")
    else:
        st.success("Valid contrast: sum(c_i) = 0.")
    st.caption(f"For unequal n_i designs, also check sum(n_i*c_i) = {sum_nc:.6f}.")

    c_value = float((coef_series * mean_series).sum())
    denom = float(((coef_series ** 2) / n_series).sum())
    se_c = np.sqrt(mse * denom) if np.isfinite(mse) and np.isfinite(denom) and denom > 0 else np.nan
    t_stat = c_value / se_c if np.isfinite(se_c) and se_c > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_error)) if np.isfinite(t_stat) and df_error > 0 else np.nan
    t_crit = stats.t.ppf(1 - alpha / 2, df_error) if df_error > 0 else np.nan
    ci_low = c_value - t_crit * se_c if np.isfinite(t_crit) and np.isfinite(se_c) else np.nan
    ci_high = c_value + t_crit * se_c if np.isfinite(t_crit) and np.isfinite(se_c) else np.nan

    st.subheader("Contrast Test and Confidence Interval")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated contrast (C)", f"{c_value:.4f}")
    m2.metric("SE(C)", f"{se_c:.4f}" if np.isfinite(se_c) else "NA")
    m3.metric("t-statistic", f"{t_stat:.4f}" if np.isfinite(t_stat) else "NA")
    m4.metric("p-value", f"{p_value:.6f}" if np.isfinite(p_value) else "NA")
    st.write(f"{(1 - alpha) * 100:.0f}% CI for contrast: [{ci_low:.4f}, {ci_high:.4f}]")
    st.caption("If CI includes 0, do not reject H0: contrast equals zero.")

    scheffe_f = stats.f.ppf(1 - alpha, n_treatments - 1, df_error) if df_error > 0 else np.nan
    scheffe_mult = np.sqrt((n_treatments - 1) * scheffe_f) if np.isfinite(scheffe_f) else np.nan
    scheffe_margin = scheffe_mult * se_c if np.isfinite(scheffe_mult) and np.isfinite(se_c) else np.nan
    s_low = c_value - scheffe_margin if np.isfinite(scheffe_margin) else np.nan
    s_high = c_value + scheffe_margin if np.isfinite(scheffe_margin) else np.nan
    st.write(f"Scheffe simultaneous CI: [{s_low:.4f}, {s_high:.4f}]")

    contrib_df = pd.DataFrame({
        factor_name: level_names,
        "n_i": n_series.values,
        "mean_i": mean_series.values,
        "c_i": coef_series.values,
        "c_i*mean_i": (coef_series * mean_series).values
    })

    if np.isfinite(denom) and denom > 0 and np.isfinite(mse) and mse > 0:
        c_star = coef_series / np.sqrt(denom)
        c_std = float((c_star * mean_series).sum())
        t_std = c_std / np.sqrt(mse)
        st.subheader("Standardized Contrast")
        st.write(f"Standardized contrast C*: {c_std:.4f}")
        st.write(f"t(C*) = C*/sqrt(MSE): {t_std:.4f}")
        contrib_df["c_i_star"] = c_star.values

    st.subheader("Relationship Between Factor Levels and Contrast")
    st.dataframe(contrib_df.round(4))

    corr = np.nan
    if np.std(coef_series.values) > 0 and np.std(mean_series.values) > 0:
        corr = float(np.corrcoef(coef_series.values, mean_series.values)[0, 1])
    st.caption(
        f"Correlation between contrast coefficients and level means: {corr:.4f}"
        if np.isfinite(corr) else
        "Correlation unavailable (insufficient variation)."
    )

    fig_rel = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rel.add_trace(
        go.Bar(x=level_names, y=mean_series.values, name="Treatment mean", marker_color="#1f77b4"),
        secondary_y=False
    )
    fig_rel.add_trace(
        go.Scatter(x=level_names, y=coef_series.values, mode="lines+markers", name="Contrast coefficient", line=dict(color="#d62728")),
        secondary_y=True
    )
    fig_rel.update_layout(
        title="Treatment Means vs Contrast Coefficients",
        xaxis_title=factor_name,
        height=430
    )
    fig_rel.update_yaxes(title_text="Mean of Y", secondary_y=False)
    fig_rel.update_yaxes(title_text="Coefficient c_i", secondary_y=True)
    _safe_plot(fig_rel)

    fig_contrib = go.Figure()
    fig_contrib.add_trace(go.Bar(
        x=level_names,
        y=(coef_series * mean_series).values,
        marker_color="#2ca02c",
        name="Contribution c_i*mean_i"
    ))
    fig_contrib.update_layout(
        title="Level Contributions to the Contrast Estimate",
        xaxis_title=factor_name,
        yaxis_title="c_i * mean_i",
        height=400
    )
    _safe_plot(fig_contrib)


def posthoc_live_three_tests():
    st.title("Post-hoc Live Analyzer: LSD, Tukey, and Duncan")
    st.markdown("By **Leonardo H. Talero-Sarmiento** "
                "[View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown(
        "Move the three treatment sliders to recompute one-way ANOVA and all post-hoc tests in real time."
    )
    with st.expander("Formulas for the Three Post-hoc Tests", expanded=True):
        st.markdown("**LSD (Fisher)**")
        st.latex(r"t_{ij}=\frac{|\bar{Y}_i-\bar{Y}_j|}{\sqrt{MSE\left(\frac{1}{n_i}+\frac{1}{n_j}\right)}}")
        st.latex(r"LSD_{ij}=t_{1-\alpha/2,\;df_e}\sqrt{MSE\left(\frac{1}{n_i}+\frac{1}{n_j}\right)}")
        st.latex(r"\text{Reject }H_0\text{ if }|\bar{Y}_i-\bar{Y}_j|>LSD_{ij}")
        st.markdown("**Tukey HSD (balanced n)**")
        st.latex(r"q_{ij}=\frac{|\bar{Y}_i-\bar{Y}_j|}{\sqrt{MSE/n}}")
        st.latex(r"\text{Reject }H_0\text{ if }q_{ij}>q_{1-\alpha;\,k,df_e}")
        st.markdown("**Duncan Multiple Range**")
        st.latex(r"q_{ij}^{(r)}=\frac{|\bar{Y}_i-\bar{Y}_j|}{\sqrt{\frac{MSE}{2}\left(\frac{1}{n_i}+\frac{1}{n_j}\right)}}")
        st.latex(r"\alpha_r=1-(1-\alpha)^{r-1},\quad \text{Reject if }q_{ij}^{(r)}>q_{1-\alpha_r;\,r,df_e}")

    st.sidebar.header("Simulation controls")
    replications = st.sidebar.slider("Replications per treatment", 5, 200, 30, 1)
    noise_sd = st.sidebar.slider("Within-treatment sigma", 0.0, 10.0, 1.0, 0.1)
    alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    seed = st.sidebar.number_input("Random seed (optional)", min_value=0, value=0, step=1)

    st.sidebar.header("Treatment labels")
    raw_names = [
        st.sidebar.text_input("Treatment 1 label", "Treatment A"),
        st.sidebar.text_input("Treatment 2 label", "Treatment B"),
        st.sidebar.text_input("Treatment 3 label", "Treatment C"),
    ]
    names = [n.strip() if n and n.strip() else f"Treatment {i + 1}" for i, n in enumerate(raw_names)]
    if len(set(names)) != 3:
        st.error("Treatment labels must be unique to build pairwise comparisons.")
        return

    st.sidebar.header("Treatment mean sliders")
    m1 = st.sidebar.slider(f"Mean for {names[0]}", -100.0, 100.0, 20.0, 0.1)
    m2 = st.sidebar.slider(f"Mean for {names[1]}", -100.0, 100.0, 23.0, 0.1)
    m3 = st.sidebar.slider(f"Mean for {names[2]}", -100.0, 100.0, 27.0, 0.1)

    rng = np.random.default_rng(seed=seed or None)
    means_map = {names[0]: m1, names[1]: m2, names[2]: m3}
    df = pd.DataFrame({"Treatment": np.repeat(names, replications)})
    df["Y"] = df["Treatment"].map(means_map).astype(float) + rng.normal(0, noise_sd, len(df))

    _, anova_tbl, mse, df_error = _oneway_anova_with_error_terms(df, group_col="Treatment", response_col="Y")

    group_summary = (
        df.groupby("Treatment")["Y"]
        .agg(N="size", Mean="mean", SD="std")
        .assign(SE=lambda d: d["SD"] / np.sqrt(d["N"]))
        .sort_values("Mean", ascending=False)
        .round(4)
    )
    ordered_groups = group_summary.index.tolist()

    lsd_df, _, _ = _compute_lsd_pairwise(
        df, group_col="Treatment", response_col="Y", alpha=alpha, mse=mse, df_error=df_error
    )
    lsd_df["reject"] = lsd_df["reject"].map(_to_bool)
    lsd_sig = lsd_df[lsd_df["reject"]].copy()

    tukey = pairwise_tukeyhsd(endog=df["Y"], groups=df["Treatment"], alpha=alpha)
    tuk_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tuk_df["reject"] = tuk_df["reject"].map(_to_bool)
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in tuk_df.columns:
            tuk_df[c] = pd.to_numeric(tuk_df[c], errors="coerce").round(4)
    tuk_sig = tuk_df[tuk_df["reject"]].copy()
    tuk_pair_df = tuk_df[["group1", "group2", "reject"]].copy()

    duncan_df = pd.DataFrame()
    duncan_sig = pd.DataFrame()
    duncan_error = None
    try:
        duncan_df, _, _ = _compute_duncan_pairwise(
            df, group_col="Treatment", response_col="Y", alpha=alpha, mse=mse, df_error=df_error
        )
        duncan_df["reject"] = duncan_df["reject"].map(_to_bool)
        duncan_sig = duncan_df[duncan_df["reject"]].copy()
    except Exception as e:
        duncan_error = str(e)

    mean_lookup = group_summary["Mean"].to_dict()
    n_lookup = group_summary["N"].to_dict()
    lsd_t_crit = stats.t.ppf(1 - alpha / 2, df_error) if df_error > 0 else np.nan
    tukey_q_crit = np.nan
    if hasattr(stats, 'studentized_range') and df_error > 0:
        try:
            tukey_q_crit = stats.studentized_range.ppf(1 - alpha, len(names), df_error)
        except Exception:
            tukey_q_crit = np.nan

    def _fmt_num(v, digits=3):
        return f"{v:.{digits}f}" if np.isfinite(v) else "NA"

    summary_cols = st.columns(3)
    for idx, g in enumerate(names):
        summary_cols[idx].metric(f"Sample mean: {g}", f"{group_summary.loc[g, 'Mean']:.3f}")

    st.subheader("Core Tables")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Treatment Summary (N, Mean, SD, SE)**")
        st.dataframe(group_summary.reset_index())
    with c2:
        st.markdown("**ANOVA Table (One-way)**")
        st.dataframe(anova_tbl.round(4))
        st.caption(f"Pooled error terms for pairwise tests: MSE = {mse:.4f}, df_error = {df_error:.0f}")

    st.subheader("Distribution by Treatment")
    fig_box = go.Figure()
    for g in names:
        vals = list(np.asarray(df.loc[df["Treatment"] == g, "Y"]).ravel())
        if not _is_finite_array(vals):
            st.error("Non-finite values in treatment distribution.")
            return
        fig_box.add_trace(go.Box(y=vals, name=g, boxmean=True, showlegend=False))
    fig_box.update_layout(
        xaxis_title="Treatment",
        yaxis_title="Y",
        title="Box Plot by Treatment",
        height=430
    )
    _safe_plot(fig_box)

    st.subheader("Comparison Agreement Across Tests")
    pair_rows = []
    for g1, g2 in combinations(ordered_groups, 2):
        pair_rows.append({"group1": g1, "group2": g2, "Comparison": f"{g1} vs {g2}"})
    agreement_df = pd.DataFrame(pair_rows)

    def _lookup_decision(comp_df, g1, g2):
        needed_cols = {"group1", "group2", "reject"}
        if comp_df.empty or not needed_cols.issubset(comp_df.columns):
            return "NA"
        mask = (
            ((comp_df["group1"] == g1) & (comp_df["group2"] == g2)) |
            ((comp_df["group1"] == g2) & (comp_df["group2"] == g1))
        )
        if not mask.any():
            return "NA"
        return "Sig" if _to_bool(comp_df.loc[mask, "reject"].iloc[0]) else "NS"

    agreement_df["LSD"] = [
        _lookup_decision(lsd_df, r["group1"], r["group2"]) for _, r in agreement_df.iterrows()
    ]
    agreement_df["Tukey"] = [
        _lookup_decision(tuk_pair_df, r["group1"], r["group2"]) for _, r in agreement_df.iterrows()
    ]
    agreement_df["Duncan"] = (
        [_lookup_decision(duncan_df, r["group1"], r["group2"]) for _, r in agreement_df.iterrows()]
        if duncan_error is None else "Unavailable"
    )
    st.dataframe(agreement_df[["Comparison", "LSD", "Tukey", "Duncan"]])

    tab_lsd, tab_tukey, tab_duncan, tab_data = st.tabs(
        ["LSD", "Tukey HSD", "Duncan", "Generated Data"]
    )

    with tab_lsd:
        st.markdown("**Live Substituted Equations (updates with sliders)**")
        lsd_eq_rows = []
        for _, row in lsd_df.iterrows():
            g1 = row["group1"]
            g2 = row["group2"]
            m_i = float(mean_lookup[g1])
            m_j = float(mean_lookup[g2])
            n_i = float(n_lookup[g1])
            n_j = float(n_lookup[g2])
            se_ij = np.sqrt(mse * (1.0 / n_i + 1.0 / n_j))
            t_ij = abs(m_i - m_j) / se_ij if se_ij > 0 else np.nan
            lsd_ij = lsd_t_crit * se_ij if np.isfinite(lsd_t_crit) else np.nan
            is_sig = np.isfinite(lsd_ij) and abs(m_i - m_j) > lsd_ij
            lsd_eq_rows.append({
                "Comparison": f"{g1} vs {g2}",
                "t_ij": (
                    f"|{_fmt_num(m_i)}-{_fmt_num(m_j)}| / "
                    f"sqrt({_fmt_num(mse)}*(1/{int(n_i)}+1/{int(n_j)})) = {_fmt_num(t_ij)}"
                ),
                "LSD_ij": (
                    f"{_fmt_num(lsd_t_crit)} * sqrt({_fmt_num(mse)}*(1/{int(n_i)}+1/{int(n_j)})) = "
                    f"{_fmt_num(lsd_ij)}"
                ),
                "Decision": (
                    f"|Delta|={_fmt_num(abs(m_i-m_j))} {'>' if is_sig else '<='} "
                    f"{_fmt_num(lsd_ij)} -> {'Sig' if is_sig else 'NS'}"
                )
            })
        st.dataframe(pd.DataFrame(lsd_eq_rows))

        st.markdown("**Significant Pairwise Differences**")
        if lsd_sig.empty:
            st.info(f"No significant pairwise differences for LSD at alpha = {alpha:.2f}.")
        else:
            st.dataframe(lsd_sig)
        st.markdown("**Full LSD Pairwise Table**")
        st.dataframe(lsd_df)
        st.markdown("**LSD Comparison Matrix (Sig/NS)**")
        st.dataframe(_pairwise_matrix_from_results(lsd_df[["group1", "group2", "reject"]], ordered_groups))
        st.markdown("**LSD Grouping Summary**")
        st.dataframe(_format_generic_grouping_summary(df, "Treatment", lsd_df, response_col="Y", reject_col="reject"))
        st.download_button(
            "Download LSD full report (CSV)",
            data=lsd_df.to_csv(index=False).encode("utf-8"),
            file_name="posthoc_lsd_full_report.csv",
            mime="text/csv",
            key="download_lsd_full_posthoc"
        )

    with tab_tukey:
        st.markdown("**Live Substituted Equations (updates with sliders)**")
        tukey_eq_rows = []
        for _, row in tuk_pair_df.iterrows():
            g1 = row["group1"]
            g2 = row["group2"]
            m_i = float(mean_lookup[g1])
            m_j = float(mean_lookup[g2])
            se_t = np.sqrt(mse / replications) if replications > 0 else np.nan
            q_ij = abs(m_i - m_j) / se_t if np.isfinite(se_t) and se_t > 0 else np.nan
            sig_from_rule = np.isfinite(tukey_q_crit) and np.isfinite(q_ij) and q_ij > tukey_q_crit
            sig_final = sig_from_rule if np.isfinite(tukey_q_crit) else _to_bool(row["reject"])
            tukey_eq_rows.append({
                "Comparison": f"{g1} vs {g2}",
                "q_ij": f"|{_fmt_num(m_i)}-{_fmt_num(m_j)}| / sqrt({_fmt_num(mse)}/{replications}) = {_fmt_num(q_ij)}",
                "q_critical": f"q_(1-alpha;k={len(names)},df={int(df_error)}) = {_fmt_num(tukey_q_crit)}",
                "Decision": (
                    f"{_fmt_num(q_ij)} {'>' if sig_final else '<='} {_fmt_num(tukey_q_crit)} -> "
                    f"{'Sig' if sig_final else 'NS'}"
                ) if np.isfinite(tukey_q_crit) else f"Using statsmodels reject -> {'Sig' if sig_final else 'NS'}"
            })
        st.dataframe(pd.DataFrame(tukey_eq_rows))

        st.markdown("**Significant Pairwise Differences**")
        if tuk_sig.empty:
            st.info(f"No significant pairwise differences for Tukey at alpha = {alpha:.2f}.")
        else:
            st.dataframe(tuk_sig)
        st.markdown("**Full Tukey Pairwise Table**")
        st.dataframe(tuk_df)
        st.markdown("**Tukey Comparison Matrix (Sig/NS)**")
        st.dataframe(_pairwise_matrix_from_results(tuk_pair_df, ordered_groups))
        st.markdown("**Tukey Grouping Summary**")
        st.dataframe(_format_tukey_summary_for_display(tukey, df, "Treatment", response_col="Y"))
        st.download_button(
            "Download Tukey full report (CSV)",
            data=tuk_df.to_csv(index=False).encode("utf-8"),
            file_name="posthoc_tukey_full_report.csv",
            mime="text/csv",
            key="download_tukey_full_posthoc"
        )

    with tab_duncan:
        if duncan_error is not None:
            st.error(f"Could not compute Duncan test: {duncan_error}")
        else:
            st.markdown("**Live Substituted Equations (updates with sliders)**")
            duncan_eq_rows = []
            for _, row in duncan_df.iterrows():
                g1 = row["group1"]
                g2 = row["group2"]
                r = int(row["range_r"])
                m_i = float(mean_lookup[g1])
                m_j = float(mean_lookup[g2])
                n_i = float(n_lookup[g1])
                n_j = float(n_lookup[g2])
                alpha_r = 1 - (1 - alpha) ** (r - 1)
                se_q = np.sqrt((mse / 2.0) * (1.0 / n_i + 1.0 / n_j))
                q_ij = abs(m_i - m_j) / se_q if se_q > 0 else np.nan
                q_crit_r = (
                    stats.studentized_range.ppf(1 - alpha_r, r, df_error)
                    if hasattr(stats, 'studentized_range') and df_error > 0 else np.nan
                )
                is_sig = np.isfinite(q_ij) and np.isfinite(q_crit_r) and q_ij > q_crit_r
                duncan_eq_rows.append({
                    "Comparison": f"{g1} vs {g2}",
                    "q_ij^(r)": (
                        f"|{_fmt_num(m_i)}-{_fmt_num(m_j)}| / "
                        f"sqrt(({_fmt_num(mse)}/2)*(1/{int(n_i)}+1/{int(n_j)})) = {_fmt_num(q_ij)}"
                    ),
                    "q_critical^(r)": (
                        f"alpha_r=1-(1-{alpha:.2f})^({r}-1)={_fmt_num(alpha_r,4)}; "
                        f"q_(1-alpha_r;r={r},df={int(df_error)})={_fmt_num(q_crit_r)}"
                    ),
                    "Decision": (
                        f"{_fmt_num(q_ij)} {'>' if is_sig else '<='} {_fmt_num(q_crit_r)} -> "
                        f"{'Sig' if is_sig else 'NS'}"
                    )
                })
            st.dataframe(pd.DataFrame(duncan_eq_rows))

            st.markdown("**Significant Pairwise Differences**")
            if duncan_sig.empty:
                st.info(f"No significant pairwise differences for Duncan at alpha = {alpha:.2f}.")
            else:
                st.dataframe(duncan_sig)
            st.markdown("**Full Duncan Pairwise Table**")
            st.dataframe(duncan_df)
            st.markdown("**Duncan Comparison Matrix (Sig/NS)**")
            st.dataframe(_pairwise_matrix_from_results(duncan_df[["group1", "group2", "reject"]], ordered_groups))
            st.markdown("**Duncan Grouping Summary**")
            st.dataframe(_format_generic_grouping_summary(df, "Treatment", duncan_df, response_col="Y", reject_col="reject"))
            st.download_button(
                "Download Duncan full report (CSV)",
                data=duncan_df.to_csv(index=False).encode("utf-8"),
                file_name="posthoc_duncan_full_report.csv",
                mime="text/csv",
                key="download_duncan_full_posthoc"
            )

    with tab_data:
        st.dataframe(df.head(30))
        st.caption("Showing first 30 rows for readability.")
        st.download_button(
            "Download generated dataset (CSV)",
            data=df.to_csv(index=False),
            file_name="posthoc_live_data.csv",
            mime="text/csv",
            key="download_posthoc_data"
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
    st.title("Practice Cases: Missing-Data DOE Scenarios and Contrasts")
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

    # -------------------------
    # Concept 4: Planned Contrasts
    # -------------------------
    con_rows = []
    rng3 = np.random.default_rng(2026)
    treatment_levels = ["A", "B", "C", "D"]
    treatment_means = {"A": 40.0, "B": 42.0, "C": 49.0, "D": 51.0}
    for r in range(1, 9):
        for t in treatment_levels:
            con_rows.append({
                "Replication": r,
                "Treatment": t,
                "Y": round(float(rng3.normal(treatment_means[t], 1.3)), 2)
            })
    base_con = pd.DataFrame(con_rows)

    with st.expander("Concept 4 - Planned Contrasts and Factor Relationships (3 cases)", expanded=False):
        _context_box(
            "Coating Process Optimization (One-way, 4 levels)",
            [
                "**Experimental units:** coated panels",
                "**Aim:** compare low settings (A,B) against high settings (C,D) and explore trend effects",
                "**Factor (IV):** Process setting level (A / B / C / D)",
                "**Dependent variable (Y):** Adhesion score",
                "**Model:** one-way ANOVA with planned contrasts"
            ]
        )

        c_case1 = base_con.copy()
        st.markdown("**Case 1: Balanced design for planned contrast (A+B) vs (C+D).**")
        st.caption("Suggested coefficients: c = [1, 1, -1, -1].")
        st.dataframe(c_case1.head(12))
        _download_case(c_case1, "contrast_case_1_balanced", "contrast_case_1")
        st.caption("Guideline: test C = sum(c_i * ybar_i) and report t-test plus CI for the contrast.")

        c_case2 = base_con[~((base_con["Treatment"] == "D") & (base_con["Replication"].isin([2, 4, 6])))].copy()
        st.markdown("**Case 2: Unequal sample sizes due to missing runs in treatment D.**")
        st.caption("Suggested coefficients: c = [1, 1, -1, -1], but use unequal-n variance formula.")
        st.dataframe(c_case2.head(12))
        _download_case(c_case2, "contrast_case_2_unbalanced", "contrast_case_2")
        st.caption("Guideline: compute SE(C) with sum(c_i^2 / n_i), not with a common n.")

        c_case3 = base_con.copy()
        dose_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        c_case3["DoseIndex"] = c_case3["Treatment"].map(dose_map)
        st.markdown("**Case 3: Ordered factor levels to evaluate a linear trend contrast.**")
        st.caption("Suggested linear trend coefficients: c = [-3, -1, 1, 3].")
        st.dataframe(c_case3.head(12))
        _download_case(c_case3, "contrast_case_3_linear_trend", "contrast_case_3")
        st.caption("Guideline: relate sign and magnitude of c_i with group means to interpret trend direction.")

    st.info("Tip for students: Start with visual missingness checks, then compare complete-case ANOVA/OLS against a sensitivity method before final conclusions.")


# -------------------------
# NEW PAGE: Diseño robusto de Taguchi — caso café monodosis
# -------------------------
def _taguchi_coffee_data(tds_target=1.25):
    """Datos sintéticos del caso docente de café monodosis."""
    df = pd.DataFrame({
        "Corrida": np.arange(1, 9),
        "A": [1, 1, 1, 1, 2, 2, 2, 2],
        "B": [1, 1, 2, 2, 1, 1, 2, 2],
        "C": [1, 2, 1, 2, 1, 2, 1, 2],
        "D": [1, 2, 2, 1, 2, 1, 1, 2],
        "TDS_87": [1.170, 1.165, 1.200, 1.175, 1.200, 1.175, 1.210, 1.205],
        "TDS_93": [1.370, 1.365, 1.330, 1.365, 1.320, 1.355, 1.320, 1.315],
    })
    df["y_87"] = (df["TDS_87"] - tds_target).abs()
    df["y_93"] = (df["TDS_93"] - tds_target).abs()
    df["Q"] = (df["y_87"] ** 2 + df["y_93"] ** 2) / 2.0
    df["eta"] = -10.0 * np.log10(df["Q"])
    df["desviacion_media"] = (df["y_87"] + df["y_93"]) / 2.0
    return df


def _taguchi_factor_info():
    return {
        "A": {"nombre": "Tostión", "n1": "Medio", "n2": "Medio-alto"},
        "B": {"nombre": "Molienda", "n1": "Medio", "n2": "Medio-fino"},
        "C": {"nombre": "Masa de café", "n1": "11 g", "n2": "13 g"},
        "D": {"nombre": "Resistencia del filtro", "n1": "Baja", "n2": "Alta"},
    }


def _taguchi_main_effect_table(df, response_col, maximize=True):
    info = _taguchi_factor_info()
    rows = []
    for factor, meta in info.items():
        m1 = float(df.loc[df[factor] == 1, response_col].mean())
        m2 = float(df.loc[df[factor] == 2, response_col].mean())
        preferred = 1 if (m1 >= m2 if maximize else m1 <= m2) else 2
        rows.append({
            "Factor": factor,
            "Nombre": meta["nombre"],
            "Nivel 1": m1,
            "Nivel 2": m2,
            "Efecto (N2-N1)": m2 - m1,
            "Nivel preferido": f"{factor}{preferred}",
        })
    return pd.DataFrame(rows)


def _taguchi_anova_eta(df):
    """ANOVA pedagógica sobre eta usando solo efectos principales del L8."""
    info = _taguchi_factor_info()
    y = df["eta"].astype(float)
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    rows = []
    ss_factors = 0.0
    for factor, meta in info.items():
        grouped = df.groupby(factor)["eta"].agg(["mean", "count"])
        ss = float(sum(grouped.loc[level, "count"] * (grouped.loc[level, "mean"] - grand) ** 2
                       for level in grouped.index))
        ss_factors += ss
        rows.append({
            "Fuente": f"{factor} — {meta['nombre']}",
            "SC": ss,
            "gl": 1,
        })
    ss_error = max(ss_total - ss_factors, 0.0)
    df_error = 3  # 7 gl totales - 4 efectos principales
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    for row in rows:
        row["CM"] = row["SC"] / row["gl"]
        row["F"] = row["CM"] / ms_error if ms_error > 0 else np.nan
        row["p"] = 1 - stats.f.cdf(row["F"], row["gl"], df_error) if np.isfinite(row["F"]) else np.nan
        row["Contribución (%)"] = 100.0 * row["SC"] / ss_total if ss_total > 0 else np.nan
    rows.append({
        "Fuente": "Residuo / términos no modelados",
        "SC": ss_error,
        "gl": df_error,
        "CM": ms_error,
        "F": np.nan,
        "p": np.nan,
        "Contribución (%)": 100.0 * ss_error / ss_total if ss_total > 0 else np.nan,
    })
    rows.append({
        "Fuente": "Total",
        "SC": ss_total,
        "gl": 7,
        "CM": np.nan,
        "F": np.nan,
        "p": np.nan,
        "Contribución (%)": 100.0,
    })
    return pd.DataFrame(rows)


def _taguchi_process_figure():
    """Diagrama ilustrativo del proceso y de dónde se fijan los factores A-D."""
    stages = [
        ("Tostión", "A: nivel de tostión"),
        ("Molienda", "B: tamaño de molienda"),
        ("Dosificación", "C: masa de café"),
        ("Filtro y sellado", "D: resistencia del filtro"),
        ("Producto", "Monodosis"),
        ("Uso", "87–93 °C"),
    ]
    fig = go.Figure()
    xs = np.arange(len(stages))
    for i, (title, subtitle) in enumerate(stages):
        fill = "#e8f3f5" if i < 4 else ("#f4ead5" if i == 4 else "#f8dddd")
        border = "#0b7285" if i < 4 else ("#a06b19" if i == 4 else "#a63d40")
        fig.add_shape(
            type="rect", x0=i - 0.40, x1=i + 0.40, y0=0.34, y1=0.82,
            fillcolor=fill, line=dict(color=border, width=2)
        )
        fig.add_annotation(x=i, y=0.66, text=f"<b>{title}</b>", showarrow=False, font=dict(size=14))
        fig.add_annotation(x=i, y=0.48, text=subtitle, showarrow=False, font=dict(size=11))
        if i < len(stages) - 1:
            fig.add_annotation(x=i + 0.52, y=0.58, ax=i + 0.34, ay=0.58,
                               xref="x", yref="y", axref="x", ayref="y",
                               text="", showarrow=True, arrowhead=3, arrowsize=1.3,
                               arrowwidth=2, arrowcolor="#566573")
    fig.add_annotation(
        x=4.5, y=0.12,
        text="La empresa fija A–D; la temperatura de preparación aparece después, durante el uso.",
        showarrow=False, font=dict(size=12, color="#444")
    )
    fig.update_xaxes(visible=False, range=[-0.6, 5.6])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=300, margin=dict(l=15, r=15, t=20, b=20), plot_bgcolor="white")
    return fig


def _taguchi_main_effect_plot(effect_df, y_title, title, prefer_high=True):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        f"A — Tostión", "B — Molienda", "C — Masa de café", "D — Filtro"
    ])
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (_, row), (r, c) in zip(effect_df.iterrows(), positions):
        vals = [float(row["Nivel 1"]), float(row["Nivel 2"])]
        fig.add_trace(
            go.Scatter(
                x=["Nivel 1", "Nivel 2"], y=vals, mode="lines+markers+text",
                text=[f"{vals[0]:.3f}", f"{vals[1]:.3f}"], textposition="top center",
                line=dict(width=3), marker=dict(size=9), showlegend=False
            ), row=r, col=c
        )
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=2, col=1)
    fig.update_layout(title=title, height=560, margin=dict(t=70, l=40, r=20, b=35))
    return fig


def taguchi_robust_design():
    st.title("Diseño robusto de Taguchi — Caso café monodosis")
    st.markdown(
        "Esta sección sigue un único caso docente: diseñar una monodosis de café que mantenga "
        "su concentración próxima al objetivo aun cuando el consumidor prepare la bebida con "
        "agua a temperaturas diferentes. Los datos son **sintéticos** y se usan con fines didácticos."
    )

    # Navegación interna con botones, toda en español.
    if "taguchi_view" not in st.session_state:
        st.session_state["taguchi_view"] = "Caso y proceso"

    labels = [
        "Caso y proceso",
        "Arreglo L8",
        "Cálculo S/N",
        "Efectos principales",
        "ANOVA",
        "Confirmación",
    ]
    cols = st.columns(6)
    for col, label in zip(cols, labels):
        with col:
            if st.button(label, key=f"taguchi_nav_{label}"):
                st.session_state["taguchi_view"] = label

    view = st.session_state["taguchi_view"]
    st.caption(f"Vista actual: **{view}**")

    tds_target = 1.25
    df = _taguchi_coffee_data(tds_target=tds_target)
    info = _taguchi_factor_info()

    # =====================================================
    # 1. CASO Y PROCESO
    # =====================================================
    if view == "Caso y proceso":
        st.subheader("1. Del proceso productivo al problema de robustez")
        st.plotly_chart(_taguchi_process_figure(), use_container_width=True)

        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("### Factores de control")
            factor_rows = []
            for f, meta in info.items():
                factor_rows.append({
                    "Factor": f,
                    "Decisión de la empresa": meta["nombre"],
                    "Nivel 1": meta["n1"],
                    "Nivel 2": meta["n2"],
                })
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True)

        with c2:
            st.markdown("### Condición de ruido del ejercicio")
            st.info(
                "**Temperatura del agua usada por el consumidor:** 87 °C y 93 °C. "
                "La empresa puede recomendar una temperatura, pero no garantizarla durante el uso."
            )
            st.markdown("### Respuesta de calidad")
            st.latex(r"TDS^*=1.25\%")
            st.latex(r"y=|TDS-TDS^*|")
            st.write(
                "No buscamos minimizar el TDS. Buscamos minimizar la **distancia al objetivo**: "
                "por debajo, la bebida tiende a ser más débil; por encima, más intensa."
            )

        with st.expander("Tipos de variación en el diseño robusto", expanded=True):
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.markdown("**Ruido externo**")
                st.write("Temperatura y humedad del entorno, almacenamiento o planta.")
            with v2:
                st.markdown("**Ruido interno**")
                st.write("Variación de materia prima y tolerancias de componentes.")
            with v3:
                st.markdown("**Deterioro**")
                st.write("Desgaste, deriva o pérdida de calibración de equipos.")
            with v4:
                st.markdown("**Factor de señal**")
                st.write("Parámetro que el usuario modifica intencionalmente para pedir otra respuesta.")
            st.caption(
                "En este caso la temperatura se modela como **ruido de uso**, porque el objetivo de TDS permanece fijo. "
                "En un diseño dinámico, una variable manipulada para solicitar deliberadamente distintos niveles de respuesta "
                "podría modelarse como factor de señal."
            )

        st.markdown("### ¿Por qué usar Taguchi aquí?")
        st.success(
            "El problema ya no es identificar una causa, sino elegir una configuración A–D "
            "que sea poco sensible a una condición de uso que seguirá variando."
        )

    # =====================================================
    # 2. ARREGLO L8
    # =====================================================
    elif view == "Arreglo L8":
        st.subheader("2. Arreglo ortogonal L8: ocho configuraciones, dos condiciones de ruido")
        st.markdown(
            "Con cuatro factores a dos niveles, un factorial completo requeriría 16 combinaciones. "
            "El arreglo L8 utiliza ocho corridas para estudiar de forma eficiente los efectos principales."
        )

        design_view = df[["Corrida", "A", "B", "C", "D", "TDS_87", "TDS_93"]].copy()
        design_view.columns = ["Corrida", "A", "B", "C", "D", "TDS a 87 °C (%)", "TDS a 93 °C (%)"]
        st.dataframe(design_view.round(3), use_container_width=True)

        st.markdown("### Visualización de la ortogonalidad")
        coded = df[["A", "B", "C", "D"]].replace({1: -1, 2: 1})
        corr = coded.corr()
        fig_orth = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title="Correlación")
        ))
        fig_orth.update_layout(
            title="Factores codificados (-1,+1): correlaciones entre columnas del L8",
            height=410, margin=dict(t=65, l=40, r=30, b=40)
        )
        _safe_plot(fig_orth)
        st.caption(
            "Los valores fuera de la diagonal son 0: las columnas A–D son ortogonales entre sí. "
            "Cada efecto principal puede estimarse sin correlación lineal con los demás efectos principales incluidos."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Factorial completo", "16 corridas")
        c2.metric("Arreglo L8", "8 corridas")
        c3.metric("Reducción", "50 %")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Corrida"], y=df["TDS_87"], mode="lines+markers", name="87 °C"
        ))
        fig.add_trace(go.Scatter(
            x=df["Corrida"], y=df["TDS_93"], mode="lines+markers", name="93 °C"
        ))
        fig.add_hline(y=tds_target, line_dash="dash", annotation_text="Objetivo TDS = 1.25 %")
        fig.update_layout(
            title="Respuesta observada en las dos condiciones de ruido",
            xaxis_title="Corrida",
            yaxis_title="TDS (%)",
            height=430,
        )
        _safe_plot(fig)

        st.download_button(
            "Descargar datos del caso (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="taguchi_cafe_monodosis_L8.csv",
            mime="text/csv",
            key="taguchi_download_l8"
        )

    # =====================================================
    # 3. CÁLCULO S/N
    # =====================================================
    elif view == "Cálculo S/N":
        st.subheader("3. Del TDS a la razón señal/ruido")
        run = st.selectbox("Seleccione una corrida para desarrollar el cálculo", df["Corrida"].tolist(), index=0)
        row = df.loc[df["Corrida"] == run].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("TDS a 87 °C", f"{row['TDS_87']:.3f} %")
        c2.metric("TDS a 93 °C", f"{row['TDS_93']:.3f} %")
        c3.metric("Objetivo", f"{tds_target:.2f} %")

        st.markdown("#### Paso 1 — desviación respecto al objetivo")
        st.latex(r"y_{87}=|TDS_{87}-TDS^*|")
        st.latex(r"y_{93}=|TDS_{93}-TDS^*|")
        st.write(f"Para la corrida {run}:  y87 = **{row['y_87']:.3f}**,  y93 = **{row['y_93']:.3f}**.")

        st.markdown("#### Paso 2 — pérdida cuadrática media")
        st.latex(r"Q_i=\frac{y_{87}^2+y_{93}^2}{2}")
        st.code(
            f"Q{run} = ({row['y_87']:.3f}² + {row['y_93']:.3f}²) / 2 = {row['Q']:.5f}",
            language="text"
        )

        st.markdown("#### Paso 3 — razón señal/ruido para menor-es-mejor")
        st.latex(r"\eta_i=-10\log_{10}(Q_i)")
        st.code(
            f"η{run} = -10 log10({row['Q']:.5f}) = {row['eta']:.2f} dB",
            language="text"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Desviación media", f"{row['desviacion_media']:.3f}")
        m2.metric("Q", f"{row['Q']:.5f}")
        m3.metric("η", f"{row['eta']:.2f} dB")
        st.success("Regla de lectura: menor desviación y menor Q son mejores; en la escala S/N, **mayor η es mejor**.")

        calc_table = df[["Corrida", "y_87", "y_93", "Q", "eta"]].copy()
        calc_table.columns = ["Corrida", "y(87 °C)", "y(93 °C)", "Q", "η (dB)"]
        st.dataframe(calc_table.round({"y(87 °C)": 3, "y(93 °C)": 3, "Q": 5, "η (dB)": 2}),
                     use_container_width=True, hide_index=True)

    # =====================================================
    # 4. EFECTOS PRINCIPALES
    # =====================================================
    elif view == "Efectos principales":
        st.subheader("4. ¿Qué niveles mejoran la robustez y el centrado?")
        eta_eff = _taguchi_main_effect_table(df, "eta", maximize=True)
        dev_eff = _taguchi_main_effect_table(df, "desviacion_media", maximize=False)

        tab_sn, tab_media = st.tabs(["Efectos sobre S/N", "Efectos sobre la desviación media"])

        with tab_sn:
            st.markdown(
                "La razón S/N resume el comportamiento bajo las dos condiciones de temperatura. "
                "Para este criterio, el nivel preferido es el que tenga **mayor η**."
            )
            _safe_plot(_taguchi_main_effect_plot(
                eta_eff, "η (dB)", "Efectos principales sobre la razón señal/ruido"
            ))
            show = eta_eff[["Factor", "Nombre", "Nivel 1", "Nivel 2", "Efecto (N2-N1)", "Nivel preferido"]].copy()
            st.dataframe(show.round(3), use_container_width=True)

        with tab_media:
            st.markdown(
                "Esta gráfica responde una pregunta complementaria: ¿qué niveles mantienen, en promedio, "
                "la bebida más cerca del TDS objetivo? Aquí **menor desviación media es mejor**."
            )
            _safe_plot(_taguchi_main_effect_plot(
                dev_eff, "Desviación media", "Efectos principales sobre la media de la desviación"
            ))
            show = dev_eff[["Factor", "Nombre", "Nivel 1", "Nivel 2", "Efecto (N2-N1)", "Nivel preferido"]].copy()
            st.dataframe(show.round(4), use_container_width=True)

        best = "–".join(eta_eff["Nivel preferido"].tolist())
        st.success(f"Combinación sugerida por los efectos principales de S/N: **{best}**.")
        st.caption(
            "En el caso sintético, S/N y desviación media conducen a la misma dirección de niveles: "
            "A2, B2, C1 y D2."
        )

    # =====================================================
    # 5. ANOVA
    # =====================================================
    elif view == "ANOVA":
        st.subheader("5. ANOVA de la razón S/N: ¿qué factores explican más variación?")
        anova = _taguchi_anova_eta(df)
        st.dataframe(anova.round({"SC": 4, "CM": 4, "F": 2, "p": 6, "Contribución (%)": 2}),
                     use_container_width=True, hide_index=True)

        plot_df = anova[anova["Fuente"].str.contains("—", regex=False)].copy()
        fig = go.Figure(go.Bar(
            x=plot_df["Contribución (%)"],
            y=plot_df["Fuente"],
            orientation="h",
            text=plot_df["Contribución (%)"].map(lambda v: f"{v:.1f}%"),
            textposition="outside"
        ))
        fig.update_layout(
            title="Contribución de los factores a la variación de η",
            xaxis_title="Contribución (%)",
            yaxis_title="Factor",
            height=430,
            yaxis=dict(autorange="reversed")
        )
        _safe_plot(fig)

        top2 = plot_df.sort_values("Contribución (%)", ascending=False).head(2)["Fuente"].tolist()
        st.info(f"En estos datos sintéticos, las mayores contribuciones corresponden a **{top2[0]}** y **{top2[1]}**.")
        st.warning(
            "**Lectura metodológica:** el L8 no tiene réplica de error puro. El residuo mostrado corresponde a "
            "variación no explicada por los cuatro efectos principales (incluye posibles interacciones y error). "
            "Por eso, los porcentajes de contribución son más útiles pedagógicamente que interpretar los p-valores "
            "como evidencia confirmatoria independiente."
        )

    # =====================================================
    # 6. CONFIRMACIÓN
    # =====================================================
    elif view == "Confirmación":
        st.subheader("6. La combinación recomendada todavía debe confirmarse")
        eta_eff = _taguchi_main_effect_table(df, "eta", maximize=True)
        best = "–".join(eta_eff["Nivel preferido"].tolist())
        st.markdown(f"La combinación sugerida es **{best}**.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Corrida de confirmación sintética")
            t87 = st.number_input("TDS de confirmación a 87 °C (%)", value=1.23, step=0.005, format="%.3f")
            t93 = st.number_input("TDS de confirmación a 93 °C (%)", value=1.28, step=0.005, format="%.3f")
        y87 = abs(float(t87) - tds_target)
        y93 = abs(float(t93) - tds_target)
        q = (y87 ** 2 + y93 ** 2) / 2.0
        eta_conf = -10 * np.log10(q) if q > 0 else np.inf

        with c2:
            st.markdown("#### Resultado")
            st.metric("y(87 °C)", f"{y87:.3f}")
            st.metric("y(93 °C)", f"{y93:.3f}")
            st.metric("η de confirmación", "∞" if not np.isfinite(eta_conf) else f"{eta_conf:.2f} dB")

        best_observed = float(df["eta"].max())
        st.latex(r"Q_{conf}=\frac{y_{87}^2+y_{93}^2}{2}")
        st.write(f"Q de confirmación = **{q:.5f}**")
        if np.isfinite(eta_conf) and eta_conf > best_observed:
            st.success(
                f"η_confirmación = **{eta_conf:.2f} dB** supera la mejor corrida observada "
                f"(**{best_observed:.2f} dB**). La confirmación es coherente con una mejora."
            )
        elif np.isfinite(eta_conf):
            st.warning(
                f"η_confirmación = **{eta_conf:.2f} dB** no supera la mejor corrida observada "
                f"(**{best_observed:.2f} dB**). Conviene revisar la recomendación o repetir la confirmación."
            )
        else:
            st.success("La confirmación coincide exactamente con el objetivo en ambas condiciones; Q = 0.")

        st.markdown("### Lectura final")
        st.write(
            "La mejor corrida observada no es necesariamente la combinación recomendada. "
            "Taguchi utiliza los efectos de los niveles para proponer una configuración y después exige "
            "comprobarla experimentalmente."
        )
        st.caption("Todos los datos de esta sección son sintéticos y sirven únicamente para la sesión docente.")



# -------------------------
# NEW PAGE: Tipos de respuesta Taguchi y razón señal/ruido
# -------------------------
def _taguchi_sn_value(values, case):
    """Calcula la razón señal/ruido de Taguchi para cuatro casos estándar."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return np.nan

    if case == "Menor es mejor":
        return -10.0 * np.log10(np.mean(y ** 2))

    if case == "Mayor es mejor":
        if np.any(y == 0):
            return np.nan
        return -10.0 * np.log10(np.mean(1.0 / (y ** 2)))

    if len(y) < 2:
        return np.nan

    mean_y = float(np.mean(y))
    sd_y = float(np.std(y, ddof=1))

    if case == "Nominal — varianza dependiente de la media":
        if mean_y <= 0 or sd_y <= 0:
            return np.nan
        return 20.0 * np.log10(mean_y / sd_y)

    if case == "Nominal — varianza independiente de la media":
        if sd_y <= 0:
            return np.inf
        return -20.0 * np.log10(sd_y)

    return np.nan


def _taguchi_case_defaults(case):
    """Datos ilustrativos para tres configuraciones y cuatro réplicas."""
    if case == "Menor es mejor":
        return pd.DataFrame({
            "Réplica": [1, 2, 3, 4],
            "Configuración A": [5.2, 4.9, 5.5, 5.1],
            "Configuración B": [3.2, 3.5, 3.1, 3.4],
            "Configuración C": [4.0, 2.0, 5.5, 3.0],
        })
    if case == "Mayor es mejor":
        return pd.DataFrame({
            "Réplica": [1, 2, 3, 4],
            "Configuración A": [82, 85, 80, 84],
            "Configuración B": [92, 95, 93, 94],
            "Configuración C": [98, 70, 100, 76],
        })
    if case == "Nominal — varianza dependiente de la media":
        return pd.DataFrame({
            "Réplica": [1, 2, 3, 4],
            "Configuración A": [48.0, 52.0, 49.0, 51.0],
            "Configuración B": [58.0, 62.0, 59.0, 61.0],
            "Configuración C": [49.8, 50.2, 50.0, 50.1],
        })
    return pd.DataFrame({
        "Réplica": [1, 2, 3, 4],
        "Configuración A": [48.0, 52.0, 49.0, 51.0],
        "Configuración B": [49.7, 50.4, 50.1, 49.8],
        "Configuración C": [54.8, 55.2, 55.1, 54.9],
    })


def _taguchi_case_metadata(case):
    meta = {
        "Menor es mejor": {
            "pregunta": "¿Cómo hacer que una respuesta no negativa sea lo más pequeña y estable posible?",
            "ejemplos": "Defectos, rugosidad, tiempo de espera, emisiones, desviación respecto a un objetivo.",
            "formula": r"\eta=-10\log_{10}\left(\frac{1}{n}\sum_{i=1}^{n}y_i^2\right)",
            "regla": "Mayor η es mejor; valores pequeños y poco variables de y producen una razón S/N mayor.",
            "restriccion": "Adecuado cuando el ideal físico es cero o cuando se transforma la respuesta a una pérdida/desviación no negativa.",
            "objetivo_default": 0.0,
        },
        "Mayor es mejor": {
            "pregunta": "¿Cómo hacer que una respuesta positiva sea lo más grande y estable posible?",
            "ejemplos": "Resistencia, rendimiento, vida útil, eficiencia, capacidad de carga.",
            "formula": r"\eta=-10\log_{10}\left(\frac{1}{n}\sum_{i=1}^{n}\frac{1}{y_i^2}\right)",
            "regla": "Mayor η es mejor; penaliza especialmente respuestas pequeñas.",
            "restriccion": "Requiere respuestas distintas de cero y, en la práctica, normalmente positivas.",
            "objetivo_default": None,
        },
        "Nominal — varianza dependiente de la media": {
            "pregunta": "¿Cómo reducir la variación cuando la dispersión tiende a crecer con la media?",
            "ejemplos": "Dimensiones, voltaje, fuerza o caudal cuando la desviación estándar escala con el nivel medio.",
            "formula": r"\eta=20\log_{10}\left(\frac{\bar y}{s}\right)",
            "regla": "Mayor η es mejor: busca una relación media/desviación estándar grande.",
            "restriccion": "La S/N mide robustez relativa; el ajuste de la media al objetivo nominal debe verificarse por separado.",
            "objetivo_default": 50.0,
        },
        "Nominal — varianza independiente de la media": {
            "pregunta": "¿Cómo minimizar la variabilidad cuando la dispersión no depende materialmente de la media?",
            "ejemplos": "Características nominales en las que se puede ajustar el centrado independientemente de la dispersión.",
            "formula": r"\eta=-20\log_{10}(s)",
            "regla": "Mayor η es mejor porque corresponde a menor desviación estándar.",
            "restriccion": "La S/N se concentra en la dispersión; la cercanía de la media al objetivo nominal se analiza aparte.",
            "objetivo_default": 50.0,
        },
    }
    return meta[case]


def taguchi_response_types():
    st.title("Taguchi — Tipos de respuesta y razón señal/ruido")
    st.markdown(
        "Esta página compara los casos clásicos de razón señal/ruido de Taguchi. "
        "En todos ellos, la regla final es la misma: **se selecciona el nivel o configuración con mayor S/N**, "
        "pero la función utilizada depende de lo que signifique calidad para la respuesta."
    )

    if "taguchi_sn_case" not in st.session_state:
        st.session_state["taguchi_sn_case"] = "Menor es mejor"

    cases = [
        "Menor es mejor",
        "Mayor es mejor",
        "Nominal — varianza dependiente de la media",
        "Nominal — varianza independiente de la media",
    ]

    st.markdown("### Seleccione el tipo de característica de calidad")
    bcols = st.columns(4)
    for col, case in zip(bcols, cases):
        with col:
            if st.button(case, key=f"btn_sn_{case}", use_container_width=True):
                st.session_state["taguchi_sn_case"] = case

    case = st.session_state["taguchi_sn_case"]
    meta = _taguchi_case_metadata(case)

    st.info(f"**Caso activo:** {case}")
    st.markdown(f"**Pregunta de ingeniería:** {meta['pregunta']}")
    st.caption(f"Ejemplos: {meta['ejemplos']}")

    # Resumen comparativo
    with st.expander("Ver mapa completo de los cuatro casos", expanded=False):
        summary = pd.DataFrame([
            {
                "Caso": "Menor es mejor",
                "Qué se desea": "Acercar y a 0",
                "S/N": "−10 log10(promedio(y²))",
                "Lectura": "Penaliza magnitud y variación",
            },
            {
                "Caso": "Mayor es mejor",
                "Qué se desea": "Hacer y grande",
                "S/N": "−10 log10(promedio(1/y²))",
                "Lectura": "Penaliza respuestas pequeñas",
            },
            {
                "Caso": "Nominal; varianza depende de media",
                "Qué se desea": "Reducir variación relativa",
                "S/N": "20 log10(media/s)",
                "Lectura": "Robustez relativa a la escala",
            },
            {
                "Caso": "Nominal; varianza independiente",
                "Qué se desea": "Reducir dispersión",
                "S/N": "−20 log10(s)",
                "Lectura": "Separa dispersión del centrado",
            },
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "En los casos nominales, una S/N alta no demuestra por sí sola que la media esté en el objetivo. "
            "El centrado debe revisarse además de la robustez."
        )

    # Fórmula y explicación
    f1, f2 = st.columns([1.05, 1])
    with f1:
        st.markdown("### Fórmula")
        st.latex(meta["formula"])
        st.success(meta["regla"])
        st.warning(meta["restriccion"])

    with f2:
        st.markdown("### ¿Qué ocurre si cambia la media o la variabilidad?")
        conceptual = go.Figure()

        if case == "Menor es mejor":
            x = np.array([1, 2, 3, 4, 5], dtype=float)
            eta = [-10*np.log10(np.mean(np.repeat(v, 4)**2)) for v in x]
            conceptual.add_trace(go.Scatter(x=x, y=eta, mode="lines+markers"))
            conceptual.update_xaxes(title="Nivel medio de la respuesta (menor es preferible)")
        elif case == "Mayor es mejor":
            x = np.array([20, 40, 60, 80, 100], dtype=float)
            eta = [-10*np.log10(np.mean(1/(np.repeat(v, 4)**2))) for v in x]
            conceptual.add_trace(go.Scatter(x=x, y=eta, mode="lines+markers"))
            conceptual.update_xaxes(title="Nivel medio de la respuesta (mayor es preferible)")
        elif case == "Nominal — varianza dependiente de la media":
            s_vals = np.array([8, 6, 4, 2, 1], dtype=float)
            eta = 20*np.log10(50/s_vals)
            conceptual.add_trace(go.Scatter(x=s_vals, y=eta, mode="lines+markers"))
            conceptual.update_xaxes(title="Desviación estándar s, con media fija")
        else:
            s_vals = np.array([8, 6, 4, 2, 1], dtype=float)
            eta = -20*np.log10(s_vals)
            conceptual.add_trace(go.Scatter(x=s_vals, y=eta, mode="lines+markers"))
            conceptual.update_xaxes(title="Desviación estándar s")

        conceptual.update_yaxes(title="Razón S/N, η (dB)")
        conceptual.update_layout(height=340, margin=dict(t=20, l=30, r=20, b=40))
        _safe_plot(conceptual)

    # Dataset editable
    st.markdown("### Experimento ilustrativo")
    st.write(
        "Edite las réplicas si desea experimentar. Cada columna representa una configuración candidata "
        "sometida a varias réplicas o condiciones de ruido."
    )
    defaults = _taguchi_case_defaults(case)
    edited = st.data_editor(
        defaults,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=f"editor_{case}",
    )

    config_cols = [c for c in edited.columns if c != "Réplica"]
    rows = []
    for c in config_cols:
        vals = pd.to_numeric(edited[c], errors="coerce").dropna().to_numpy(dtype=float)
        eta_val = _taguchi_sn_value(vals, case)
        rows.append({
            "Configuración": c,
            "n": len(vals),
            "Media": float(np.mean(vals)) if len(vals) else np.nan,
            "Desv. estándar": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "S/N η (dB)": eta_val,
        })
    result_df = pd.DataFrame(rows)

    # Métricas y objetivo nominal
    target = meta["objetivo_default"]
    if target is not None:
        target = st.number_input(
            "Objetivo nominal (para evaluar centrado)",
            value=float(target),
            step=0.5,
            key=f"target_{case}"
        )
        result_df["|Media - objetivo|"] = (result_df["Media"] - float(target)).abs()

    st.markdown("### Resultados por configuración")
    st.dataframe(result_df.round(4), use_container_width=True, hide_index=True)

    valid_eta = result_df[np.isfinite(result_df["S/N η (dB)"])].copy()
    if not valid_eta.empty:
        best_idx = valid_eta["S/N η (dB)"].idxmax()
        best_name = valid_eta.loc[best_idx, "Configuración"]
        best_eta = valid_eta.loc[best_idx, "S/N η (dB)"]
        st.success(f"Según la razón S/N, la mejor alternativa es **{best_name}** con η = **{best_eta:.2f} dB**.")

    # Gráficas
    g1, g2 = st.columns(2)

    with g1:
        fig_raw = go.Figure()
        for c in config_cols:
            fig_raw.add_trace(go.Scatter(
                x=edited["Réplica"],
                y=pd.to_numeric(edited[c], errors="coerce"),
                mode="lines+markers",
                name=c,
            ))
        if target is not None:
            fig_raw.add_hline(y=float(target), line_dash="dash", annotation_text="Objetivo")
        fig_raw.update_layout(
            title="Respuestas observadas",
            xaxis_title="Réplica / condición de ruido",
            yaxis_title="Respuesta y",
            height=430,
        )
        _safe_plot(fig_raw)

    with g2:
        fig_sn = go.Figure(go.Bar(
            x=result_df["Configuración"],
            y=result_df["S/N η (dB)"],
            text=result_df["S/N η (dB)"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "NA"),
            textposition="outside",
        ))
        fig_sn.update_layout(
            title="Comparación por razón señal/ruido",
            xaxis_title="Configuración",
            yaxis_title="η (dB) — mayor es mejor",
            height=430,
        )
        _safe_plot(fig_sn)

    # Cálculo detallado
    st.markdown("### Cálculo paso a paso")
    chosen = st.selectbox(
        "Seleccione una configuración",
        config_cols,
        key=f"detail_{case}"
    )
    vals = pd.to_numeric(edited[chosen], errors="coerce").dropna().to_numpy(dtype=float)

    if len(vals):
        st.write("Observaciones:", ", ".join(f"{v:.4g}" for v in vals))
        mean_y = float(np.mean(vals))
        sd_y = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        eta_y = _taguchi_sn_value(vals, case)

        if case == "Menor es mejor":
            q = float(np.mean(vals**2))
            st.latex(r"Q=\frac{1}{n}\sum y_i^2")
            st.code(f"Q = ({' + '.join([f'{v:.4g}²' for v in vals])}) / {len(vals)} = {q:.6f}", language="text")
            st.latex(r"\eta=-10\log_{10}(Q)")
            st.code(f"η = -10 log10({q:.6f}) = {eta_y:.3f} dB", language="text")

        elif case == "Mayor es mejor":
            if np.any(vals == 0):
                st.error("La fórmula mayor-es-mejor no está definida si alguna respuesta es 0.")
            else:
                q = float(np.mean(1.0/(vals**2)))
                st.latex(r"Q=\frac{1}{n}\sum \frac{1}{y_i^2}")
                st.code(
                    f"Q = ({' + '.join([f'1/{v:.4g}²' for v in vals])}) / {len(vals)} = {q:.8f}",
                    language="text"
                )
                st.latex(r"\eta=-10\log_{10}(Q)")
                st.code(f"η = -10 log10({q:.8f}) = {eta_y:.3f} dB", language="text")

        elif case == "Nominal — varianza dependiente de la media":
            st.latex(r"\eta=20\log_{10}\left(\frac{\bar y}{s}\right)")
            st.code(
                f"ȳ = {mean_y:.4f}; s = {sd_y:.4f}; η = 20 log10({mean_y:.4f}/{sd_y:.4f}) = {eta_y:.3f} dB",
                language="text"
            )

        else:
            st.latex(r"\eta=-20\log_{10}(s)")
            st.code(
                f"s = {sd_y:.4f}; η = -20 log10({sd_y:.4f}) = {eta_y:.3f} dB",
                language="text"
            )

        if target is not None:
            delta = abs(mean_y - float(target))
            st.metric("Desviación de la media respecto al objetivo", f"{delta:.4f}")
            if delta > 0:
                st.caption(
                    "Una configuración puede tener una S/N alta y, sin embargo, estar descentrada. "
                    "En nominal-es-mejor se debe estudiar **robustez + centrado**."
                )

    # Vinculación con el caso del café
    st.markdown("### Conexión con el caso del café monodosis")
    if case == "Menor es mejor":
        st.info(
            "En el caso del café no se minimiza directamente el TDS. Primero se transforma la respuesta: "
            "**y = |TDS − 1.25|**. Como esa desviación idealmente vale 0, se analiza correctamente con "
            "la razón S/N de **menor es mejor**."
        )
    elif "Nominal" in case:
        st.info(
            "También sería posible formular un estudio nominal directamente sobre TDS, pero entonces deben "
            "separarse explícitamente dos preguntas: **¿qué tan variable es el TDS?** y "
            "**¿qué tan cerca está su media de 1.25 %?**. La formulación actual con la desviación hace esa "
            "lógica más transparente para una sesión de 30 minutos."
        )
    else:
        st.info(
            "Mayor-es-mejor sería pertinente si la característica de calidad fuera, por ejemplo, resistencia, "
            "rendimiento o vida útil y no existiera un objetivo nominal intermedio."
        )

    st.caption(
        "Referencia conceptual: las fórmulas corresponden a las variantes Taguchi SN−, SN+, SN0 y SN00. "
        "En todos los casos se busca maximizar η; lo que cambia es cómo se define la pérdida frente al ruido."
    )


# -------------------------
# Navigation
# -------------------------
PAGES = {
    "Introduction": introduction_profile,
    "Diseño robusto de Taguchi — Café monodosis": taguchi_robust_design,
    "Taguchi — Tipos de respuesta y razón S/N": taguchi_response_types,
    "Anova One-way - Introduction": anova_oneway,
    "Contrast Analysis and Factor Relationships": contrast_analysis,
    "Post-hoc Live Analyzer (LSD, Tukey, Duncan)": posthoc_live_three_tests,
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial Designs with Three Factors and Three Levels": three_factorial,
    "Fractional Factorial Designs (2^k-p)": fractional_factorial,
    "Conjoint Analysis (Choice-Based)": conjoint_analysis,
    "Tips to Analyze the Statistical Outputs": Analysis,
    "Practice Cases: Missing Data": missing_data_case_studies,
}

st.title('Navegación')
choice = st.radio("Ir a", list(PAGES.keys()))
PAGES[choice]()
