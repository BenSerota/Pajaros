"""
Central statistics module for the bird mortality analysis dashboard.

Contains all statistical tests used across dashboard pages. Every analytical
page imports from this module. Tests cover categorical comparisons, trend
analysis, spatial clustering, circular statistics, and regression modeling.

Imports SIG_LEVELS from config for Spanish significance labels in stat_badge.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import (
    binomtest,
    chi2_contingency,
    chisquare,
    fisher_exact,
    kruskal,
    norm,
    poisson,
    rankdata,
)
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def months_to_radians(months: pd.Series) -> np.ndarray:
    """Convert month numbers (1-12) to radians on the unit circle.

    January maps to 0 radians, proceeding counter-clockwise so that
    December maps to just under 2*pi.

    Parameters
    ----------
    months : pd.Series
        Series of integer month numbers (1-12).

    Returns
    -------
    np.ndarray
        Angles in radians.
    """
    m = np.asarray(months, dtype=float)
    return (m - 1) * 2 * np.pi / 12


def cramers_v(chi2: float, n: int, min_dim: int) -> float:
    """Compute Cramer's V effect size from a chi-squared statistic.

    Parameters
    ----------
    chi2 : float
        Chi-squared test statistic.
    n : int
        Total number of observations.
    min_dim : int
        min(rows, cols) - 1 of the contingency table.

    Returns
    -------
    float
        Cramer's V, clamped to [0, 1].
    """
    if n == 0 or min_dim == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min_dim)))


def format_p_value(p: float) -> str:
    """Format a p-value for display.

    Parameters
    ----------
    p : float
        Raw p-value.

    Returns
    -------
    str
        Human-readable p-value string.
    """
    if p is None or np.isnan(p):
        return "p = NaN"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# 1. Chi-squared test of independence
# ---------------------------------------------------------------------------


def chi_squared_test(contingency_table: pd.DataFrame) -> dict:
    """Perform a chi-squared test of independence on a contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Rows and columns represent categorical groups; cell values are counts.

    Returns
    -------
    dict
        Keys: statistic, p_value, dof, effect_size (Cramer's V), expected.
    """
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "dof": 0,
            "effect_size": np.nan,
            "expected": np.array([]),
        }

    table = contingency_table.values
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum()
    min_dim = min(table.shape[0], table.shape[1]) - 1
    v = cramers_v(chi2, n, min_dim)

    return {
        "statistic": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "effect_size": float(v),
        "expected": expected,
    }


# ---------------------------------------------------------------------------
# 2. Pairwise Fisher exact tests
# ---------------------------------------------------------------------------


def fisher_pairwise(
    contingency_table: pd.DataFrame,
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """Pairwise Fisher exact tests on rows of a contingency table.

    For tables larger than 2x2, every unique pair of rows is extracted into
    a 2xC sub-table and tested with Fisher's exact test (or chi-squared
    when the sub-table is larger than 2x2, since scipy's fisher_exact only
    handles 2x2).

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Count data with groups as rows and categories as columns.
    correction : str, optional
        Multiple-testing correction method. Only ``"bonferroni"`` is
        currently implemented (default).

    Returns
    -------
    pd.DataFrame
        Columns: group1, group2, odds_ratio, p_value, p_adjusted.
    """
    rows = contingency_table.index.tolist()
    if len(rows) < 2:
        return pd.DataFrame(columns=["group1", "group2", "odds_ratio", "p_value", "p_adjusted"])

    results = []
    pairs = list(combinations(rows, 2))

    for r1, r2 in pairs:
        sub = contingency_table.loc[[r1, r2]]
        vals = sub.values

        if vals.shape[1] == 2:
            # True 2x2 -- use Fisher exact
            odds, p = fisher_exact(vals)
        else:
            # Collapse into 2x2 is not straightforward; use chi-squared
            # approximation for the pair (still a valid pairwise comparison).
            try:
                _, p, _, _ = chi2_contingency(vals)
                odds = np.nan
            except ValueError:
                p = np.nan
                odds = np.nan

        results.append({"group1": r1, "group2": r2, "odds_ratio": odds, "p_value": p})

    df_out = pd.DataFrame(results)

    # Apply correction
    n_tests = len(df_out)
    if correction == "bonferroni" and n_tests > 0:
        df_out["p_adjusted"] = np.minimum(df_out["p_value"] * n_tests, 1.0)
    else:
        df_out["p_adjusted"] = df_out["p_value"]

    return df_out


# ---------------------------------------------------------------------------
# 3. Kruskal-Wallis test
# ---------------------------------------------------------------------------


def kruskal_wallis_test(groups: list) -> dict:
    """Non-parametric Kruskal-Wallis H test across multiple groups.

    Parameters
    ----------
    groups : list[np.ndarray]
        Each element is a 1-D array of observations for one group.

    Returns
    -------
    dict
        Keys: statistic (H), p_value, effect_size (epsilon-squared).
        Epsilon-squared = (H - k + 1) / (n - k).
    """
    # Filter out empty groups
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]

    k = len(groups)
    if k < 2:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}

    n = sum(len(g) for g in groups)
    if n <= k:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}

    try:
        h_stat, p = kruskal(*groups)
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}

    epsilon_sq = (h_stat - k + 1) / (n - k) if (n - k) > 0 else np.nan

    return {
        "statistic": float(h_stat),
        "p_value": float(p),
        "effect_size": float(epsilon_sq),
    }


# ---------------------------------------------------------------------------
# 4. Dunn's test (post-hoc pairwise after Kruskal-Wallis)
# ---------------------------------------------------------------------------


def dunns_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Post-hoc Dunn's test with Bonferroni correction.

    Uses rank-based z-statistics to compare all pairs of groups after a
    significant Kruskal-Wallis test.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    value_col : str
        Column containing numeric values.
    group_col : str
        Column containing group labels.

    Returns
    -------
    pd.DataFrame
        Columns: group1, group2, z_statistic, p_value, p_adjusted.
    """
    working = df[[group_col, value_col]].dropna()
    if working.empty:
        return pd.DataFrame(
            columns=["group1", "group2", "z_statistic", "p_value", "p_adjusted"]
        )

    groups = working[group_col].unique()
    if len(groups) < 2:
        return pd.DataFrame(
            columns=["group1", "group2", "z_statistic", "p_value", "p_adjusted"]
        )

    # Global ranks
    n = len(working)
    working = working.copy()
    working["_rank"] = rankdata(working[value_col].values)

    # Mean rank and size per group
    group_stats = working.groupby(group_col)["_rank"].agg(["mean", "count"])

    # Tied-rank correction factor
    ranks = working["_rank"].values
    unique_ranks, counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(counts ** 3 - counts)
    tie_correction = 1 - tie_sum / (n ** 3 - n) if (n ** 3 - n) > 0 else 1.0

    sigma_base = (n * (n + 1) / 12.0) * tie_correction

    results = []
    pairs = list(combinations(groups, 2))

    for g1, g2 in pairs:
        n1 = group_stats.loc[g1, "count"]
        n2 = group_stats.loc[g2, "count"]
        mean1 = group_stats.loc[g1, "mean"]
        mean2 = group_stats.loc[g2, "mean"]

        se = np.sqrt(sigma_base * (1.0 / n1 + 1.0 / n2))
        if se == 0:
            z = np.nan
            p = np.nan
        else:
            z = (mean1 - mean2) / se
            p = 2.0 * norm.sf(np.abs(z))

        results.append({
            "group1": g1,
            "group2": g2,
            "z_statistic": float(z) if not np.isnan(z) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
        })

    df_out = pd.DataFrame(results)
    n_tests = len(df_out)
    if n_tests > 0:
        df_out["p_adjusted"] = np.minimum(df_out["p_value"] * n_tests, 1.0)
    else:
        df_out["p_adjusted"] = np.nan

    return df_out


# ---------------------------------------------------------------------------
# 5. Mann-Kendall trend test
# ---------------------------------------------------------------------------


def mann_kendall_test(series: np.ndarray) -> dict:
    """Non-parametric Mann-Kendall trend test with Sen's slope estimator.

    Parameters
    ----------
    series : np.ndarray
        1-D array of observations ordered in time.

    Returns
    -------
    dict
        Keys: tau, p_value, slope (Sen's slope), intercept.
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 3:
        return {"tau": np.nan, "p_value": np.nan, "slope": np.nan, "intercept": np.nan}

    # S statistic
    s = 0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            s += np.sign(diff)
            if (j - i) != 0:
                slopes.append(diff / (j - i))

    # Variance of S (with tie correction)
    unique_vals, counts = np.unique(x, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0

    # Tau
    denominator = n * (n - 1) / 2.0
    tau = s / denominator if denominator > 0 else np.nan

    # Z-statistic and p-value
    if var_s <= 0:
        p = 1.0
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0
        p = 2.0 * norm.sf(np.abs(z))

    # Sen's slope
    if len(slopes) > 0:
        slope = float(np.median(slopes))
    else:
        slope = np.nan

    # Intercept: median of (y_i - slope * i)
    if not np.isnan(slope):
        intercepts = x - slope * np.arange(n)
        intercept = float(np.median(intercepts))
    else:
        intercept = np.nan

    return {
        "tau": float(tau),
        "p_value": float(p),
        "slope": slope,
        "intercept": intercept,
    }


# ---------------------------------------------------------------------------
# 6. Poisson test per span
# ---------------------------------------------------------------------------


def poisson_test_per_span(
    df: pd.DataFrame,
    span_col: str = "span_id",
    count_col: str = None,
) -> pd.DataFrame:
    """Test whether each span has significantly more events than expected
    under a uniform Poisson model.

    If *count_col* is ``None``, the number of rows per span is used as the
    observed count.

    Parameters
    ----------
    df : pd.DataFrame
        Data with at least a *span_col* column.
    span_col : str
        Column identifying spans (e.g., bridge spans).
    count_col : str, optional
        Column with pre-aggregated event counts.  When ``None``, the function
        counts rows per span.

    Returns
    -------
    pd.DataFrame
        Columns: span_id, observed, expected, p_value, significant.
    """
    if df.empty or span_col not in df.columns:
        return pd.DataFrame(
            columns=["span_id", "observed", "expected", "p_value", "significant"]
        )

    if count_col is not None and count_col in df.columns:
        counts = df.groupby(span_col)[count_col].sum()
    else:
        counts = df.groupby(span_col).size()

    total_events = counts.sum()
    n_spans = len(counts)

    if n_spans == 0 or total_events == 0:
        return pd.DataFrame(
            columns=["span_id", "observed", "expected", "p_value", "significant"]
        )

    expected_rate = total_events / n_spans

    records = []
    for span, obs in counts.items():
        # One-tailed test: P(X >= obs) under Poisson(expected_rate)
        p = float(poisson.sf(obs - 1, expected_rate))
        records.append({
            "span_id": span,
            "observed": int(obs),
            "expected": round(expected_rate, 2),
            "p_value": p,
            "significant": p < 0.05,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Negative binomial regression
# ---------------------------------------------------------------------------


def negative_binomial_regression(df: pd.DataFrame, formula: str) -> dict:
    """Fit a negative binomial regression model.

    Uses ``statsmodels`` GLM with NegativeBinomial family.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the columns referenced in *formula*.
    formula : str
        A patsy/statsmodels formula string, e.g.
        ``"count ~ C(season) + wind_speed"``.

    Returns
    -------
    dict
        Keys: summary (model summary text), params (dict per covariate with
        coef, se, pvalue, irr), aic, bic, alpha (dispersion).
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return {
            "summary": "statsmodels not installed",
            "params": {},
            "aic": np.nan,
            "bic": np.nan,
            "alpha": np.nan,
        }

    if df.empty:
        return {
            "summary": "Empty dataframe",
            "params": {},
            "aic": np.nan,
            "bic": np.nan,
            "alpha": np.nan,
        }

    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial(),
        ).fit()
    except Exception as exc:
        return {
            "summary": f"Model failed: {exc}",
            "params": {},
            "aic": np.nan,
            "bic": np.nan,
            "alpha": np.nan,
        }

    params_dict = {}
    for name in model.params.index:
        coef = model.params[name]
        se = model.bse[name]
        pval = model.pvalues[name]
        irr = np.exp(coef)
        params_dict[name] = {
            "coef": float(coef),
            "se": float(se),
            "pvalue": float(pval),
            "irr": float(irr),
        }

    alpha = getattr(model.family, "alpha", np.nan)

    return {
        "summary": str(model.summary()),
        "params": params_dict,
        "aic": float(model.aic),
        "bic": float(model.bic_llf) if hasattr(model, "bic_llf") else float(model.bic),
        "alpha": float(alpha) if alpha is not None else np.nan,
    }


# ---------------------------------------------------------------------------
# 8. Rayleigh test for circular uniformity
# ---------------------------------------------------------------------------


def rayleigh_test(angles_rad: np.ndarray) -> dict:
    """Rayleigh test for non-uniformity of circular data.

    Parameters
    ----------
    angles_rad : np.ndarray
        Angles in radians.

    Returns
    -------
    dict
        Keys: Z (Rayleigh statistic), p_value, mean_direction (radians),
        mean_resultant_length (R-bar).
    """
    a = np.asarray(angles_rad, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)

    if n == 0:
        return {
            "Z": np.nan,
            "p_value": np.nan,
            "mean_direction": np.nan,
            "mean_resultant_length": np.nan,
        }

    # Mean resultant vector
    c_bar = np.mean(np.cos(a))
    s_bar = np.mean(np.sin(a))
    r_bar = np.sqrt(c_bar ** 2 + s_bar ** 2)
    mean_dir = np.arctan2(s_bar, c_bar) % (2 * np.pi)

    # Rayleigh statistic
    z = n * r_bar ** 2

    # Approximation for p-value (valid for large n)
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    p = max(0.0, min(1.0, p))

    return {
        "Z": float(z),
        "p_value": float(p),
        "mean_direction": float(mean_dir),
        "mean_resultant_length": float(r_bar),
    }


# ---------------------------------------------------------------------------
# 9. Watson's U-squared two-sample circular test
# ---------------------------------------------------------------------------


def watson_u2_test(angles1: np.ndarray, angles2: np.ndarray) -> dict:
    """Watson's U-squared two-sample test for circular data.

    Tests whether two samples of angles come from the same circular
    distribution.

    Parameters
    ----------
    angles1 : np.ndarray
        First sample of angles in radians.
    angles2 : np.ndarray
        Second sample of angles in radians.

    Returns
    -------
    dict
        Keys: U2 (test statistic), p_value (approximate from critical values).
    """
    a1 = np.asarray(angles1, dtype=float)
    a2 = np.asarray(angles2, dtype=float)
    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    n1 = len(a1)
    n2 = len(a2)
    n = n1 + n2

    if n1 < 2 or n2 < 2:
        return {"U2": np.nan, "p_value": np.nan}

    # Normalize angles to [0, 2*pi)
    a1 = a1 % (2 * np.pi)
    a2 = a2 % (2 * np.pi)

    # Combine and sort
    combined = np.concatenate([a1, a2])
    labels = np.concatenate([np.ones(n1), np.zeros(n2)])
    order = np.argsort(combined)
    sorted_labels = labels[order]

    # Compute cumulative differences
    # d_k = (cumulative proportion sample 1) - (cumulative proportion sample 2)
    cum1 = np.cumsum(sorted_labels) / n1
    cum2 = np.cumsum(1 - sorted_labels) / n2
    d = cum1 - cum2

    d_bar = np.mean(d)

    u2 = (n1 * n2 / (n ** 2)) * (np.sum((d - d_bar) ** 2))

    # Approximate p-value from critical value table
    # Common critical values for Watson's U^2:
    # 0.187 (alpha=0.10), 0.152 (alpha=0.15), 0.187 (alpha=0.10),
    # 0.268 (alpha=0.025), 0.385 (alpha=0.005)
    if u2 >= 0.385:
        p = 0.001
    elif u2 >= 0.268:
        p = 0.01
    elif u2 >= 0.187:
        p = 0.05
    elif u2 >= 0.152:
        p = 0.10
    else:
        p = 0.20

    return {
        "U2": float(u2),
        "p_value": float(p),
    }


# ---------------------------------------------------------------------------
# 10. Binomial test
# ---------------------------------------------------------------------------


def binomial_test(
    observed: int,
    total: int,
    expected_proportion: float,
) -> dict:
    """Test whether an observed proportion differs from an expected proportion.

    Parameters
    ----------
    observed : int
        Number of "successes" (e.g., bird species count).
    total : int
        Total number of trials / observations.
    expected_proportion : float
        The null-hypothesis proportion, in [0, 1].

    Returns
    -------
    dict
        Keys: observed, total, expected, p_value.
    """
    if total <= 0 or not (0 <= expected_proportion <= 1):
        return {
            "observed": observed,
            "total": total,
            "expected": expected_proportion,
            "p_value": np.nan,
        }

    result = binomtest(observed, total, expected_proportion)

    return {
        "observed": int(observed),
        "total": int(total),
        "expected": float(expected_proportion),
        "p_value": float(result.pvalue),
    }


# ---------------------------------------------------------------------------
# 11. DBSCAN spatial clustering
# ---------------------------------------------------------------------------


def dbscan_clusters(
    coords: np.ndarray,
    eps: float = 500,
    min_samples: int = 3,
) -> np.ndarray:
    """Spatial clustering using DBSCAN.

    Parameters
    ----------
    coords : np.ndarray
        UTM coordinates, shape (n, 2).
    eps : float
        Maximum distance (in coordinate units, e.g. meters for UTM) between
        two samples to be considered neighbours (default 500).
    min_samples : int
        Minimum number of samples in a neighbourhood to form a core point
        (default 3).

    Returns
    -------
    np.ndarray
        Cluster labels; -1 indicates noise.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != 2:
        return np.array([], dtype=int)

    # Drop rows with NaN
    valid_mask = ~np.isnan(c).any(axis=1)
    valid_coords = c[valid_mask]

    if len(valid_coords) < min_samples:
        labels = np.full(c.shape[0], -1, dtype=int)
        return labels

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_coords)

    # Map labels back to full array
    labels = np.full(c.shape[0], -1, dtype=int)
    labels[valid_mask] = db.labels_

    return labels


# ---------------------------------------------------------------------------
# 12. Statistical badge for Streamlit display
# ---------------------------------------------------------------------------


def stat_badge(
    test_name: str,
    statistic: float,
    p_value: float,
    effect_size: float = None,
    n: int = None,
) -> str:
    """Return styled HTML for a statistical significance badge.

    Significance levels (Spanish, from config.SIG_LEVELS):
      - p < 0.001: green,  "***", "Altamente significativo"
      - p < 0.01:  yellow, "**",  "Muy significativo"
      - p < 0.05:  orange, "*",   "Significativo"
      - p >= 0.05: gray,   "ns",  "No significativo"

    Parameters
    ----------
    test_name : str
        Name of the test (e.g., "Chi-cuadrado").
    statistic : float
        Test statistic value.
    p_value : float
        P-value from the test.
    effect_size : float, optional
        Effect size metric to display.
    n : int, optional
        Sample size to display.

    Returns
    -------
    str
        HTML string suitable for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    # Import significance level labels from config; fall back to hardcoded
    # Spanish labels if config is unavailable.
    try:
        from config import SIG_LEVELS
        _sig_map = {row[0]: (row[1], row[2]) for row in SIG_LEVELS}
    except Exception:
        _sig_map = {
            0.001: ("***", "Altamente significativo"),
            0.01:  ("**",  "Muy significativo"),
            0.05:  ("*",   "Significativo"),
            1.0:   ("ns",  "No significativo"),
        }

    if p_value is None or np.isnan(p_value):
        color = "#9e9e9e"
        bg = "#f5f5f5"
        stars = "?"
        label = "No concluyente"
    elif p_value < 0.001:
        color = "#2e7d32"
        bg = "#e8f5e9"
        stars = _sig_map.get(0.001, ("***", "Altamente significativo"))[0]
        label = _sig_map.get(0.001, ("***", "Altamente significativo"))[1]
    elif p_value < 0.01:
        color = "#f9a825"
        bg = "#fffde7"
        stars = _sig_map.get(0.01, ("**", "Muy significativo"))[0]
        label = _sig_map.get(0.01, ("**", "Muy significativo"))[1]
    elif p_value < 0.05:
        color = "#ef6c00"
        bg = "#fff3e0"
        stars = _sig_map.get(0.05, ("*", "Significativo"))[0]
        label = _sig_map.get(0.05, ("*", "Significativo"))[1]
    else:
        color = "#757575"
        bg = "#f5f5f5"
        stars = _sig_map.get(1.0, ("ns", "No significativo"))[0]
        label = _sig_map.get(1.0, ("ns", "No significativo"))[1]

    p_str = format_p_value(p_value) if p_value is not None and not np.isnan(p_value) else "p = NaN"

    stat_str = f"{statistic:.2f}" if statistic is not None and not np.isnan(statistic) else "N/D"

    parts = [
        f'<span style="'
        f"display:inline-block; padding:6px 14px; border-radius:20px; "
        f"background:{bg}; color:{color}; font-weight:600; "
        f'font-size:0.9em; border:1px solid {color}40;">'
        f"{test_name} {stars} &nbsp; "
        f"estad. = {stat_str}, {p_str}",
    ]

    if effect_size is not None and not np.isnan(effect_size):
        parts.append(f", efecto = {effect_size:.3f}")
    if n is not None:
        parts.append(f", n = {n:,}")

    parts.append(f" &mdash; <em>{label}</em></span>")

    return "".join(parts)


# ---------------------------------------------------------------------------
# 13. Chi-squared goodness-of-fit test
# ---------------------------------------------------------------------------


def chi_squared_gof(observed: np.ndarray, expected: np.ndarray = None) -> dict:
    """Chi-squared goodness-of-fit test.

    Tests whether observed counts differ from expected counts (uniform by
    default).

    Parameters
    ----------
    observed : np.ndarray
        Observed counts per category.
    expected : np.ndarray, optional
        Expected counts. If ``None``, assumes uniform distribution.

    Returns
    -------
    dict
        Keys: statistic, p_value, dof.
    """
    obs = np.asarray(observed, dtype=float)
    obs = obs[~np.isnan(obs)]

    if len(obs) < 2 or obs.sum() == 0:
        return {"statistic": np.nan, "p_value": np.nan, "dof": 0}

    if expected is not None:
        exp = np.asarray(expected, dtype=float)
    else:
        exp = None

    try:
        stat, p = chisquare(obs, f_exp=exp)
    except Exception:
        return {"statistic": np.nan, "p_value": np.nan, "dof": len(obs) - 1}

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "dof": int(len(obs) - 1),
    }


# ---------------------------------------------------------------------------
# 14. Poisson rate comparison (E-test / conditional exact test)
# ---------------------------------------------------------------------------


def poisson_rate_test(
    count1: int,
    exposure1: float,
    count2: int,
    exposure2: float,
) -> dict:
    """Compare two Poisson rates using an exact conditional test.

    Tests H0: rate1 == rate2, where rate = count / exposure.
    Uses a binomial exact test on count1 given total = count1 + count2,
    with expected proportion = exposure1 / (exposure1 + exposure2).

    Parameters
    ----------
    count1 : int
        Observed event count in group 1 (e.g., before period).
    exposure1 : float
        Exposure (e.g., months of observation) in group 1.
    count2 : int
        Observed event count in group 2 (e.g., after period).
    exposure2 : float
        Exposure in group 2.

    Returns
    -------
    dict
        Keys: rate1, rate2, rate_ratio, p_value, count1, count2,
        exposure1, exposure2.
    """
    total = count1 + count2
    if total == 0 or exposure1 <= 0 or exposure2 <= 0:
        return {
            "rate1": np.nan, "rate2": np.nan,
            "rate_ratio": np.nan, "p_value": np.nan,
            "count1": count1, "count2": count2,
            "exposure1": exposure1, "exposure2": exposure2,
        }

    rate1 = count1 / exposure1
    rate2 = count2 / exposure2
    rate_ratio = rate1 / rate2 if rate2 > 0 else np.inf

    # Conditional exact test: under H0 equal rates,
    # P(X1 = count1 | X1 + X2 = total) ~ Binomial(total, p)
    # where p = exposure1 / (exposure1 + exposure2)
    p_null = exposure1 / (exposure1 + exposure2)
    result = binomtest(count1, total, p_null, alternative="two-sided")

    return {
        "rate1": float(rate1),
        "rate2": float(rate2),
        "rate_ratio": float(rate_ratio),
        "p_value": float(result.pvalue),
        "count1": int(count1),
        "count2": int(count2),
        "exposure1": float(exposure1),
        "exposure2": float(exposure2),
    }


# ---------------------------------------------------------------------------
# 15. Fisher pairwise with Bonferroni (from 2x2 tables)
# ---------------------------------------------------------------------------


def fisher_pairwise_from_counts(
    group_labels: list,
    group_counts: list,
    total_counts: list = None,
) -> pd.DataFrame:
    """Pairwise Fisher exact tests between groups using count data.

    Each group has a count of events and a total (or uses counts directly
    for a chi-squared comparison).  Useful for comparing signal types.

    Parameters
    ----------
    group_labels : list[str]
        Names of the groups.
    group_counts : list[int]
        Event counts per group.
    total_counts : list[int], optional
        Total exposure per group.  If ``None``, uses counts as the basis
        for chi-squared pairwise tests (2x1 tables expanded to 2x2 with
        complement).

    Returns
    -------
    pd.DataFrame
        Columns: group1, group2, odds_ratio, p_value, p_adjusted.
    """
    if len(group_labels) < 2:
        return pd.DataFrame(columns=["group1", "group2", "odds_ratio", "p_value", "p_adjusted"])

    results = []
    pairs = list(combinations(range(len(group_labels)), 2))

    for i, j in pairs:
        if total_counts is not None:
            # Build 2x2: [[event_i, non_event_i], [event_j, non_event_j]]
            table = np.array([
                [group_counts[i], total_counts[i] - group_counts[i]],
                [group_counts[j], total_counts[j] - group_counts[j]],
            ])
            try:
                odds, p = fisher_exact(table)
            except Exception:
                odds, p = np.nan, np.nan
        else:
            # Simple 2x2 from counts
            table = np.array([
                [group_counts[i], sum(group_counts) - group_counts[i]],
                [group_counts[j], sum(group_counts) - group_counts[j]],
            ])
            try:
                odds, p = fisher_exact(table)
            except Exception:
                odds, p = np.nan, np.nan

        results.append({
            "group1": group_labels[i],
            "group2": group_labels[j],
            "odds_ratio": odds,
            "p_value": p,
        })

    df_out = pd.DataFrame(results)
    n_tests = len(df_out)
    if n_tests > 0:
        df_out["p_adjusted"] = np.minimum(df_out["p_value"] * n_tests, 1.0)
    else:
        df_out["p_adjusted"] = np.nan

    return df_out
