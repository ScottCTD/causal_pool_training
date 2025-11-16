#!/usr/bin/env python3
"""
Plot per-category per-question and per-option accuracy for all models.

This script loads all eval*.json files from results/1k_simple/ and creates
visualizations showing per-category metrics (per-question and per-option accuracy)
for each model.
"""

import argparse
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")
    print("The script will still generate summary tables.")


def extract_model_name(filepath: str) -> str:
    """Extract model name from eval file path."""
    filename = os.path.basename(filepath)
    # Remove 'eval_' prefix and '.json' suffix
    model_name = filename.replace('eval_', '').replace('.json', '')
    return model_name


def load_all_eval_results(dataset_dir: str) -> Dict[str, Dict]:
    """
    Load all eval*.json files from the dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory (e.g., 'datasets/1k_simple')
    
    Returns:
        Dictionary mapping model names to their evaluation results
    """
    eval_files = glob.glob(os.path.join(dataset_dir, 'eval*.json'))
    results = {}
    
    for filepath in eval_files:
        model_name = extract_model_name(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
            results[model_name] = data
    
    return results


def extract_category_metrics(results: Dict[str, Dict]) -> Tuple[Dict[str, List], Dict[str, List], List[str]]:
    """
    Extract per-category metrics from all model results.
    
    Args:
        results: Dictionary mapping model names to evaluation results
    
    Returns:
        Tuple of (per_question_metrics, per_option_metrics, categories)
        where each metrics dict maps category -> [values for each model]
    """
    # Collect all unique categories across all models
    all_categories = set()
    for model_data in results.values():
        per_question_type = model_data.get('metrics', {}).get('per_question_type', {})
        all_categories.update(per_question_type.keys())
    
    categories = sorted(all_categories)
    # Sort model names, but put "random" first as baseline
    all_model_names = sorted(results.keys())
    if "random" in all_model_names:
        model_names = ["random"] + [m for m in all_model_names if m != "random"]
    else:
        model_names = all_model_names
    
    per_question_metrics = {cat: [] for cat in categories}
    per_option_metrics = {cat: [] for cat in categories}
    
    for model_name in model_names:
        model_data = results[model_name]
        per_question_type = model_data.get('metrics', {}).get('per_question_type', {})
        
        for category in categories:
            category_data = per_question_type.get(category, {})
            per_question_metrics[category].append(
                category_data.get('per_question_accuracy', 0.0)
            )
            per_option_metrics[category].append(
                category_data.get('per_option_accuracy', 0.0)
            )
    
    return per_question_metrics, per_option_metrics, categories, model_names


def plot_category_metrics(
    per_question_metrics: Dict[str, List],
    per_option_metrics: Dict[str, List],
    categories: List[str],
    model_names: List[str],
    output_path: str = 'eval/category_metrics_plot.png'
):
    """
    Create plots for per-category metrics.
    
    Args:
        per_question_metrics: Dict mapping category -> list of per-question accuracies
        per_option_metrics: Dict mapping category -> list of per-option accuracies
        categories: List of category names
        model_names: List of model names (in same order as metric lists)
        output_path: Base path to save the plots (will generate two files)
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return
    
    # Generate two output paths
    base_path = Path(output_path)
    per_question_path = base_path.parent / f"{base_path.stem}_per_question{base_path.suffix}"
    per_option_path = base_path.parent / f"{base_path.stem}_per_option{base_path.suffix}"
    
    n_categories = len(categories)
    n_models = len(model_names)
    x = np.arange(len(categories))
    width = 0.8 / n_models
    
    # Plot per-question accuracy (separate figure)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        values = [per_question_metrics[cat][i] for cat in categories]
        bars = ax1.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        # Add accuracy labels on top of each bar
        for bar, val in zip(bars, values):
            if val > 0:
                # Position text on top of the bar
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f'{val:.3f}',
                    va='bottom',
                    ha='center',
                    fontsize=8
                )
            else:
                # For zero values, show label inside the bar
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    f'{val:.3f}',
                    va='bottom',
                    ha='center',
                    fontsize=8,
                    color='gray'
                )
    
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Per-Question Accuracy', fontsize=12)
    ax1.set_title('Per-Category Per-Question Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
    max_val = max(max(vals) for vals in per_question_metrics.values()) if per_question_metrics else 1.0
    ax1.set_ylim(0, max(1.0, max_val * 1.15))
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(str(per_question_path), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Plot saved to {per_question_path}")
    
    # Plot per-option accuracy (separate figure)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        values = [per_option_metrics[cat][i] for cat in categories]
        bars = ax2.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        # Add accuracy labels on top of each bar
        for bar, val in zip(bars, values):
            if val > 0:
                # Position text on top of the bar
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f'{val:.3f}',
                    va='bottom',
                    ha='center',
                    fontsize=8
                )
            else:
                # For zero values, show label inside the bar
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    f'{val:.3f}',
                    va='bottom',
                    ha='center',
                    fontsize=8,
                    color='gray'
                )
    
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Per-Option Accuracy', fontsize=12)
    ax2.set_title('Per-Category Per-Option Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
    max_val = max(max(vals) for vals in per_option_metrics.values()) if per_option_metrics else 1.0
    ax2.set_ylim(0, max(1.0, max_val * 1.15))
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(str(per_option_path), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Plot saved to {per_option_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot per-category per-question and per-option accuracy for all models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (results/1k_simple/)
  python plot_category_metrics.py

  # Specify dataset name (uses results/<dataset>/)
  python plot_category_metrics.py -d ds1

  # Specify custom results directory (overrides dataset)
  python plot_category_metrics.py -r results/ds1

  # Specify custom output path
  python plot_category_metrics.py -d ds1 -o plots/my_metrics.png

  # Skip plot generation (only print tables)
  python plot_category_metrics.py -np

  # Skip table printing (only generate plots)
  python plot_category_metrics.py -nt
        """
    )
    
    # Get the project root directory (assuming script is run from project root)
    project_root = Path(__file__).parent.parent
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='1k_simple',
        help='Dataset name (results/<dataset>/ will be used, default: 1k_simple)'
    )
    
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default=None,
        help='Full path to directory containing eval*.json files (overrides --dataset if specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for plots (default: <results_dir>/category_metrics_plot.png)'
    )
    
    parser.add_argument(
        '--no-plots', '-np',
        action='store_true',
        help='Skip plot generation (only print summary tables)'
    )
    
    parser.add_argument(
        '--no-tables', '-nt',
        action='store_true',
        help='Skip table printing (only generate plots)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine results directory: use --results-dir if specified, otherwise use --dataset
    project_root = Path(__file__).parent.parent
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = project_root / 'results' / args.dataset
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Loading evaluation results from {results_dir}...")
    results = load_all_eval_results(str(results_dir))
    
    if not results:
        raise ValueError(f"No eval*.json files found in {results_dir}")
    
    print(f"Found {len(results)} model evaluation files:")
    for model_name in sorted(results.keys()):
        print(f"  - {model_name}")
    
    print("\nExtracting category metrics...")
    per_question_metrics, per_option_metrics, categories, model_names = extract_category_metrics(results)
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / 'category_metrics_plot.png'
    
    # Generate plots if requested and matplotlib is available
    if not args.no_plots:
        if HAS_MATPLOTLIB:
            print("\nGenerating plots...")
            plot_category_metrics(
                per_question_metrics,
                per_option_metrics,
                categories,
                model_names,
                str(output_path)
            )
        else:
            print("\nSkipping plot generation (matplotlib not available)")
    else:
        print("\nSkipping plot generation (--no-plots flag set)")
    
    # Print summary tables if requested
    if not args.no_tables:
        col_width = max(25, max(len(m) for m in model_names) + 2)
        total_width = 30 + len(model_names) * col_width
        
        print("\n" + "="*total_width)
        print("Summary Table: Per-Category Per-Question Accuracy")
        print("="*total_width)
        print(f"{'Category':<30}", end="")
        for model_name in model_names:
            print(f"{model_name:<{col_width}}", end="")
        print()
        print("-"*total_width)
        for category in categories:
            print(f"{category:<30}", end="")
            for val in per_question_metrics[category]:
                print(f"{val:<{col_width}.4f}", end="")
            print()
        
        print("\n" + "="*total_width)
        print("Summary Table: Per-Category Per-Option Accuracy")
        print("="*total_width)
        print(f"{'Category':<30}", end="")
        for model_name in model_names:
            print(f"{model_name:<{col_width}}", end="")
        print()
        print("-"*total_width)
        for category in categories:
            print(f"{category:<30}", end="")
            for val in per_option_metrics[category]:
                print(f"{val:<{col_width}.4f}", end="")
            print()


if __name__ == '__main__':
    main()

