#!/usr/bin/env python3
"""
Plot Context Invariance Results with TOST Statistical Tests

Generates plots and statistical analysis for the context invariance experiment.
Uses TOST (Two One-Sided Tests) to demonstrate that predicted lengths are
statistically equivalent across different context sizes.

Usage:
    python scripts/plot_context_invariance.py \
        --input_path results/context_invariance/run_001/results.jsonl \
        --output_dir results/context_invariance/run_001/
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from itertools import combinations


# Plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})


def load_results(input_path: Path) -> pd.DataFrame:
    """Load JSONL results into a DataFrame."""
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Extract predictions for each context size
    records = []
    for item in data:
        doc_id = item['doc_id']
        prompt_len = item.get('prompt_len', 0)
        predictions = item.get('predictions', {})
        
        row = {'doc_id': doc_id, 'prompt_len': prompt_len}
        for ctx, pred_data in predictions.items():
            if 'error' not in pred_data and pred_data.get('predicted_new_tokens') is not None:
                row[f"ctx_{ctx}"] = pred_data['predicted_new_tokens']
            else:
                row[f"ctx_{ctx}"] = np.nan
        records.append(row)
    
    return pd.DataFrame(records)


def tost_paired(sample1: np.ndarray, sample2: np.ndarray, epsilon: float) -> dict:
    """
    Two One-Sided Tests (TOST) for equivalence testing.
    
    Tests whether |mean(sample1) - mean(sample2)| < epsilon
    
    H0: |μ1 - μ2| >= ε (not equivalent)
    H1: |μ1 - μ2| < ε (equivalent)
    
    Args:
        sample1: First sample (paired with sample2)
        sample2: Second sample
        epsilon: Equivalence bound (in same units as samples)
    
    Returns:
        Dict with p_lower, p_upper, p_tost, equivalent, mean_diff, ci
    """
    # Compute paired differences
    diff = sample1 - sample2
    n = len(diff)
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(n)
    
    # 95% CI for the difference
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # TOST: Two one-sided tests
    # Test 1: H0: μ_diff <= -ε vs H1: μ_diff > -ε
    t_lower = (mean_diff - (-epsilon)) / se_diff
    p_lower = 1 - stats.t.cdf(t_lower, df=n-1)
    
    # Test 2: H0: μ_diff >= ε vs H1: μ_diff < ε
    t_upper = (mean_diff - epsilon) / se_diff
    p_upper = stats.t.cdf(t_upper, df=n-1)
    
    # TOST p-value is the maximum of the two one-sided p-values
    p_tost = max(p_lower, p_upper)
    
    return {
        'mean_diff': float(mean_diff),
        'std_diff': float(np.std(diff, ddof=1)),
        'ci': (float(ci_lower), float(ci_upper)),
        'p_lower': float(p_lower),
        'p_upper': float(p_upper),
        'p_tost': float(p_tost),
        'epsilon': float(epsilon),
        'equivalent': bool(p_tost < 0.05),
        'n': int(n)
    }


def compute_equivalence_bound(df: pd.DataFrame, ctx_cols: list, method: str = 'percentage') -> float:
    """
    Compute the equivalence bound epsilon.
    
    Args:
        df: DataFrame with prediction columns
        ctx_cols: List of context column names
        method: 'percentage' (5% of mean) or 'fixed' (10 tokens)
    
    Returns:
        Epsilon value
    """
    all_values = df[ctx_cols].values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    if method == 'percentage':
        return min(0.05 * np.mean(all_values), 10.0)  # 5% of mean
    else:
        return 10.0  # Fixed 10 tokens


def plot_distributions(df: pd.DataFrame, ctx_cols: list, output_dir: Path):
    """Plot KDE distributions of predicted lengths for each context size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = sns.color_palette("husl", len(ctx_cols))
    
    # Plot 1: KDE overlay
    ax1 = axes[0]
    for i, col in enumerate(ctx_cols):
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            ctx_size = col.replace('ctx_', '')
            sns.kdeplot(
                data=valid_data, 
                label=f"Context {ctx_size}", 
                color=colors[i], 
                fill=True, 
                alpha=0.2, 
                linewidth=2,
                ax=ax1,
                cut=0
            )
    
    ax1.set_title("Distribution of Predicted Lengths by Context Size")
    ax1.set_xlabel("Predicted New Tokens")
    ax1.set_ylabel("Density")
    ax1.legend(title="Context Size", loc='upper right')
    
    # Plot 2: Boxplot comparison
    ax2 = axes[1]
    plot_data = []
    for col in ctx_cols:
        valid_data = df[col].dropna()
        ctx_size = col.replace('ctx_', '')
        for val in valid_data:
            plot_data.append({'Context Size': ctx_size, 'Predicted Tokens': val})
    
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(
        data=plot_df, 
        x='Context Size', 
        y='Predicted Tokens',
        hue='Context Size',
        palette=colors,
        legend=False,
        ax=ax2
    )
    ax2.set_title("Predicted Length Distribution by Context Size")
    
    plt.tight_layout()
    
    # Save
    png_path = output_dir / "invariance_distributions.png"
    pdf_path = output_dir / "invariance_distributions.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plots: {png_path}")
    return png_path


def plot_tost_results(tost_results: list, epsilon: float, output_dir: Path):
    """Plot TOST equivalence test results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    pairs = [r['pair'] for r in tost_results]
    mean_diffs = [r['mean_diff'] for r in tost_results]
    cis = [r['ci'] for r in tost_results]
    equivalents = [r['equivalent'] for r in tost_results]
    
    # Plot 1: Mean differences with confidence intervals
    ax1 = axes[0]
    y_pos = np.arange(len(pairs))
    colors = ['green' if eq else 'red' for eq in equivalents]
    
    # Plot CIs
    for i, (ci, md) in enumerate(zip(cis, mean_diffs)):
        ax1.hlines(y=i, xmin=ci[0], xmax=ci[1], colors=colors[i], linewidth=2)
        ax1.scatter([md], [i], color=colors[i], s=80, zorder=5)
    
    # Plot equivalence bounds
    ax1.axvline(-epsilon, color='gray', linestyle='--', alpha=0.7, label=f'±ε = ±{epsilon:.1f}')
    ax1.axvline(epsilon, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pairs)
    ax1.set_xlabel("Mean Difference (tokens)")
    ax1.set_title("TOST Equivalence Test: Mean Differences with 95% CI")
    ax1.legend(loc='upper right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Equivalent (p < 0.05)'),
        Patch(facecolor='red', label='Not Equivalent')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: P-values
    ax2 = axes[1]
    p_values = [r['p_tost'] for r in tost_results]
    bar_colors = ['green' if p < 0.05 else 'red' for p in p_values]
    
    bars = ax2.barh(y_pos, p_values, color=bar_colors, alpha=0.7)
    ax2.axvline(0.05, color='black', linestyle='--', label='α = 0.05')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pairs)
    ax2.set_xlabel("TOST p-value")
    ax2.set_title("TOST p-values (smaller = more evidence for equivalence)")
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, max(0.1, max(p_values) * 1.1))
    
    # Add p-value labels
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        ax2.text(p + 0.005, i, f'{p:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    png_path = output_dir / "tost_equivalence.png"
    pdf_path = output_dir / "tost_equivalence.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved TOST results: {png_path}")
    return png_path


def plot_pairwise_scatter(df: pd.DataFrame, ctx_cols: list, output_dir: Path):
    """Plot pairwise scatter plots to show correlation between context sizes."""
    n_cols = len(ctx_cols)
    if n_cols < 2:
        return None
    
    # Select key comparisons
    pairs_to_plot = [
        (ctx_cols[0], ctx_cols[-1]),  # Smallest vs largest
        (ctx_cols[0], ctx_cols[1]),   # First two
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (col1, col2) in zip(axes, pairs_to_plot):
        valid_mask = df[col1].notna() & df[col2].notna()
        x = df.loc[valid_mask, col1]
        y = df.loc[valid_mask, col2]
        
        ax.scatter(x, y, alpha=0.5, s=30)
        
        # Add identity line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', alpha=0.7)
        
        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        ctx1 = col1.replace('ctx_', '')
        ctx2 = col2.replace('ctx_', '')
        ax.set_xlabel(f"Predicted Tokens (Context {ctx1})")
        ax.set_ylabel(f"Predicted Tokens (Context {ctx2})")
        ax.set_title(f"Context {ctx1} vs {ctx2}\nr = {corr:.4f}")
        ax.legend()
    
    plt.tight_layout()
    
    png_path = output_dir / "pairwise_scatter.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter plots: {png_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot Context Invariance Results with TOST Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to results.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Equivalence bound for TOST (default: 5%% of mean prediction)"
    )
    parser.add_argument(
        "--epsilon_method",
        type=str,
        choices=['percentage', 'fixed'],
        default='percentage',
        help="Method to compute epsilon if not specified"
    )
    parser.add_argument(
        "--exclude_contexts",
        type=str,
        nargs="+",
        default=[],
        help="Context sizes to exclude from analysis (e.g., --exclude_contexts 256)"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {input_path}")
    df = load_results(input_path)
    
    # Get context columns
    ctx_cols = sorted([c for c in df.columns if c.startswith('ctx_')], 
                      key=lambda x: int(x.replace('ctx_', '')))
    
    # Exclude specified contexts
    if args.exclude_contexts:
        exclude_set = set(args.exclude_contexts)
        ctx_cols = [c for c in ctx_cols if c.replace('ctx_', '') not in exclude_set]
        print(f"Excluding contexts: {args.exclude_contexts}")
    
    print(f"Found {len(df)} samples with context sizes: {[c.replace('ctx_', '') for c in ctx_cols]}")
    
    # Filter to samples with valid predictions for all contexts
    valid_mask = df[ctx_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    print(f"Samples with valid predictions for all contexts: {len(df_valid)}")
    
    if len(df_valid) < 10:
        print("ERROR: Not enough valid samples for statistical analysis!")
        return
    
    # Compute epsilon
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = compute_equivalence_bound(df_valid, ctx_cols, args.epsilon_method)
    print(f"Using equivalence bound ε = {epsilon:.2f} tokens")
    
    # ========== STATISTICS ==========
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    stats_summary = {}
    for col in ctx_cols:
        ctx = col.replace('ctx_', '')
        values = df_valid[col].values
        stats_summary[ctx] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        print(f"Context {ctx}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, "
              f"median={np.median(values):.2f}")
    
    # ========== TOST TESTS ==========
    print("\n" + "="*60)
    print("TOST EQUIVALENCE TESTS")
    print("="*60)
    print(f"Equivalence bound: ε = {epsilon:.2f} tokens")
    print("")
    
    tost_results = []
    for col1, col2 in combinations(ctx_cols, 2):
        ctx1 = col1.replace('ctx_', '')
        ctx2 = col2.replace('ctx_', '')
        
        result = tost_paired(
            df_valid[col1].values,
            df_valid[col2].values,
            epsilon
        )
        result['pair'] = f"{ctx1} vs {ctx2}"
        result['ctx1'] = ctx1
        result['ctx2'] = ctx2
        # Convert numpy types to native Python for JSON serialization
        result['equivalent'] = bool(result['equivalent'])
        tost_results.append(result)
        
        equiv_str = "✓ EQUIVALENT" if result['equivalent'] else "✗ NOT equivalent"
        print(f"{ctx1} vs {ctx2}: mean_diff={result['mean_diff']:+.2f}, "
              f"CI=[{result['ci'][0]:.2f}, {result['ci'][1]:.2f}], "
              f"p_TOST={result['p_tost']:.4f} {equiv_str}")
    
    # Count equivalence
    n_equiv = sum(1 for r in tost_results if r['equivalent'])
    n_total = len(tost_results)
    print(f"\nSummary: {n_equiv}/{n_total} pairs are statistically equivalent (p < 0.05)")
    
    # ========== ADDITIONAL TESTS ==========
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    
    # Per-sample standard deviation
    per_sample_std = df_valid[ctx_cols].std(axis=1)
    avg_std = per_sample_std.mean()
    print(f"Average within-sample std dev: {avg_std:.2f} tokens")
    
    # Intraclass Correlation Coefficient (ICC) - simplified
    all_values = df_valid[ctx_cols].values
    grand_mean = np.mean(all_values)
    between_var = np.var(np.mean(all_values, axis=1))
    within_var = np.mean(np.var(all_values, axis=1))
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
    print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")
    
    # ========== GENERATE PLOTS ==========
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_distributions(df_valid, ctx_cols, output_dir)
    plot_tost_results(tost_results, epsilon, output_dir)
    plot_pairwise_scatter(df_valid, ctx_cols, output_dir)
    
    # ========== SAVE STATISTICS ==========
    stats_output = {
        'epsilon': epsilon,
        'epsilon_method': args.epsilon_method if args.epsilon is None else 'user_specified',
        'n_samples': len(df_valid),
        'descriptive_stats': stats_summary,
        'tost_results': tost_results,
        'n_equivalent_pairs': n_equiv,
        'n_total_pairs': n_total,
        'avg_within_sample_std': float(avg_std),
        'icc': float(icc)
    }
    
    stats_path = output_dir / "statistical_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nSaved statistical analysis: {stats_path}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
