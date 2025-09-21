import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

def calculate_psi(expected, actual, buckets=10):
    """Calculate PSI (Population Stability Index) for one numeric feature"""
    quantiles = np.linspace(0, 1, buckets + 1)
    bins = np.quantile(expected, quantiles)
    bins[0] -= 1e-8
    bins[-1] += 1e-8

    expected_perc = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=bins)[0] / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-8, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-8, actual_perc)

    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    return np.sum(psi_values)

def chi2_drift(expected, actual):
    """Chi-square test for categorical drift"""
    freq_expected = expected.value_counts()
    freq_actual = actual.value_counts()
    all_categories = set(freq_expected.index).union(set(freq_actual.index))
    table = []
    for cat in all_categories:
        table.append([freq_expected.get(cat,0), freq_actual.get(cat,0)])
    chi2_stat, p, _, _ = chi2_contingency(table)
    return chi2_stat, p

train_df = pd.read_csv("cleaned.csv")
test_df = pd.read_csv("processed_test.csv")

cols_to_drop = ["Date", "Weekly_Sales"]
train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

common_cols = [col for col in train_df.columns if col in test_df.columns]
train_df = train_df[common_cols]
test_df = test_df[common_cols]

print(f"{'Feature':<15} {'PSI/Chi2':<10} {'KS/NA':<8} {'p-value':<10} {'Drift?':<6}")
print("-"*60)

drift_count = 0
for col in common_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        psi = calculate_psi(train_df[col].values, test_df[col].values)
        ks_stat, ks_p = ks_2samp(train_df[col], test_df[col])
        drift = "Yes" if psi >= 0.25 or ks_p < 0.05 else "No"
        if drift == "Yes":
            drift_count += 1
        print(f"{col:<15} {psi:<10.2f} {ks_stat:<8.2f} {ks_p:<10.3f} {drift:<6}")
    else:
        chi2_stat, p_val = chi2_drift(train_df[col], test_df[col])
        drift = "Yes" if p_val < 0.05 else "No"
        if drift == "Yes":
            drift_count += 1
        print(f"{col:<15} {chi2_stat:<10.2f} {'NA':<8} {p_val:<10.3f} {drift:<6}")

print("-"*60)
print(f"Summary: {drift_count}/{len(common_cols)} features show significant drift.")
