#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a markdown report based on doc1 template with:
1. Merged experiment results (grouping by configuration, only seed different)
2. Plots from plot_curves_modified.py
3. Environment info from information.yaml
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def read_experiments_data(csv_path):
    """Read experiments CSV and return DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read {len(df)} experiments from {csv_path}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()


def group_experiments(df):
    """
    Group experiments that only differ by seed.
    Returns a DataFrame with merged results.
    """
    if df.empty:
        return pd.DataFrame()

    print("Original data:")
    print(df[['exp_id', 'exp_name', 'seed', 'epochs', 'top-1_acc']])

    # Create a configuration key that excludes seed
    # We'll use all columns except seed and result columns
    config_columns = [col for col in df.columns
                      if col not in ['exp_id', 'seed', 'top-1_acc', 'duration']]

    print(f"Configuration columns: {config_columns}")

    # Create a string representation of configuration for grouping
    df['config_key'] = df[config_columns].astype(str).agg('|'.join, axis=1)

    result_rows = []

    for config_key in df['config_key'].unique():
        group_df = df[df['config_key'] == config_key]
        print(f"Processing group with {len(group_df)} experiments")
        print(f"Config key: {config_key}")

        if len(group_df) == 1:
            # Single experiment
            row = group_df.iloc[0].copy()
            row['epochs_formatted'] = f"{int(row['epochs'])}"
            row['top1_formatted'] = f"{row['top-1_acc']:.4f}"
            row['exp_count'] = 1
            row['exp_ids'] = [row['exp_id']]
            row['seeds'] = [row['seed']]
            result_rows.append(row)
        else:
            # Multiple experiments with same config, different seeds
            first_row = group_df.iloc[0].copy()

            # Calculate mean ± std for epochs and top-1 accuracy
            epochs_mean = group_df['epochs'].mean()
            epochs_std = group_df['epochs'].std()
            top1_mean = group_df['top-1_acc'].mean()
            top1_std = group_df['top-1_acc'].std()

            # Format the strings
            if epochs_std > 0:
                first_row['epochs_formatted'] = f"{epochs_mean:.1f}±{epochs_std:.1f}"
            else:
                first_row['epochs_formatted'] = f"{int(epochs_mean)}"

            if top1_std > 0:
                first_row['top1_formatted'] = f"{top1_mean:.4f}±{top1_std:.4f}"
            else:
                first_row['top1_formatted'] = f"{top1_mean:.4f}"

            first_row['exp_count'] = len(group_df)
            first_row['exp_ids'] = group_df['exp_id'].tolist()
            first_row['seeds'] = group_df['seed'].tolist()

            result_rows.append(first_row)

    result_df = pd.DataFrame(result_rows)
    print(f"Grouped into {len(result_df)} rows")
    return result_df


def generate_weights_link(exp_id, epochs, github_prefix="https://github.com/username/repo/blob/main/"):
    """Generate GitHub link for model weights."""
    # Use the actual epochs value (not the formatted string)
    if isinstance(epochs, str) and '±' in epochs:
        # Extract mean from formatted string like "155.0±0.0"
        epochs_value = int(float(epochs.split('±')[0]))
    else:
        try:
            epochs_value = int(float(epochs))
        except:
            epochs_value = int(epochs)

    weights_path = f"scripts/experiments/{exp_id}/model_final_epoch_{epochs_value}.pth"
    return f"[model]({github_prefix}{weights_path})"


def get_environment_info(exp_id, experiments_dir):
    """Read environment information from information.yaml."""
    info_file = experiments_dir / exp_id / "information.yaml"

    if not info_file.exists():
        print(f"Environment info file not found: {info_file}")
        return None

    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            info = yaml.safe_load(f)

        env_info = {
            "torch_version": info.get("torch_version", "N/A"),
            "torchvision_version": info.get("torchvision_version", "N/A"),
            "cuda_version": info.get("cuda_version", "N/A"),
            "driver_version": info.get("driver_version", "N/A"),
            "graphic_card": info.get("graphic_card", "N/A")
        }
        print(f"Environment info loaded for {exp_id}")
        return env_info
    except Exception as e:
        print(f"Error reading environment info from {info_file}: {e}")
        return None


def generate_markdown_report(grouped_df, env_info, figs_dir, github_prefix):
    """Generate the markdown report content."""
    md_content = []

    # Title
    md_content.append("# Weekly Result\n")

    # Result Card
    md_content.append("## Result Card\n")
    md_content.append("| Exp ID | Exp Name | Model | Resolution | Epoch | Optimizer | Top-1 | Weight |")
    md_content.append("|----------|-------|------------|-------|-----------|-------|--------|--------|")

    if grouped_df.empty:
        md_content.append("| No data available | - | - | - | - | - | - | - |")
    else:
        for _, row in grouped_df.iterrows():
            exp_id = row['exp_id']
            if row['exp_count'] > 1:
                # For merged experiments, show first exp_id with count
                exp_id_display = f"{exp_id} (+{row['exp_count'] - 1})"
            else:
                exp_id_display = exp_id

            exp_name = row['exp_name']
            epochs = row['epochs_formatted']
            top1 = row['top1_formatted']
            optimizer = row['optimizer']

            # Generate weights link using the first exp_id in the group
            first_exp_id = row['exp_ids'][0]
            weights_link = generate_weights_link(first_exp_id, row['epochs'], github_prefix)

            md_content.append(
                f"| {exp_id_display} | {exp_name} | - | - | {epochs} | {optimizer} | {top1} | {weights_link} |")

    md_content.append("\n")

    # Plots section
    md_content.append("## Plots\n")

    # Check if plot files exist
    acc_plot_path = figs_dir / "acc_comparison.png"
    loss_plot_path = figs_dir / "loss_comparison.png"

    md_content.append("### Accuracy")
    if acc_plot_path.exists():
        md_content.append(f"![Accuracy Comparison]({acc_plot_path.as_posix()})\n")
    else:
        md_content.append(f"*Accuracy plot not found at {acc_plot_path}*\n")

    md_content.append("### Loss")
    if loss_plot_path.exists():
        md_content.append(f"![Loss Comparison]({loss_plot_path.as_posix()})\n")
    else:
        md_content.append(f"*Loss plot not found at {loss_plot_path}*\n")

    # Reproduce section (from template)
    md_content.append("## Reproduce\n")
    md_content.append(" - One-key Script:\n")
    md_content.append(" ```bash")
    md_content.append("    reproduce.sh -m check                           # Environment Check")
    md_content.append("    reproduce.sh -m train -e exp251022_172342       # Train")
    md_content.append("    reproduce.sh -m predict -e exp251022_172342     # Predict")
    md_content.append("    reproduce.sh -m download -e exp251022_172342    # Download Model")
    md_content.append(" ```\n")

    # Environment section
    md_content.append("## Environment\n")
    if env_info:
        md_content.append("| Item | Version |")
        md_content.append("|------|---------|")
        md_content.append(f"| PyTorch | {env_info['torch_version']} |")
        md_content.append(f"| TorchVision | {env_info['torchvision_version']} |")
        md_content.append(f"| CUDA | {env_info['cuda_version']} |")
        md_content.append(f"| Driver | {env_info['driver_version']} |")
        md_content.append(f"| GPU | {env_info['graphic_card']} |")
    else:
        md_content.append("*Environment information not available*\n")

    return "\n".join(md_content)


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--exp_csv", type=Path, default=Path("scripts/experiments/experiments.csv"),
                        help="Path to experiments CSV file")
    parser.add_argument("--out_md", type=Path, default=Path("report.md"),
                        help="Output markdown file path")
    parser.add_argument("--figs_dir", type=Path, default=Path("figs"),
                        help="Directory containing generated plots")
    parser.add_argument("--experiments_dir", type=Path, default=Path("scripts/experiments"),
                        help="Directory containing experiment data")
    parser.add_argument("--github_prefix", type=str,
                        default="https://github.com/wyywnab/machine_learning_study/tree/main/week5/",
                        help="GitHub URL prefix for model weight links")
    parser.add_argument("--env_exp_id", type=str,
                        default="exp251024_120200",
                        help="Experiment ID to use for environment info (uses first experiment if not specified)")

    args = parser.parse_args()

    print(f"Input CSV: {args.exp_csv}")
    print(f"Output MD: {args.out_md}")
    print(f"Figures dir: {args.figs_dir}")
    print(f"Experiments dir: {args.experiments_dir}")

    # Read and process experiments data
    df = read_experiments_data(args.exp_csv)

    if df.empty:
        print("Error: No data found in experiments CSV file")
        return

    grouped_df = group_experiments(df)

    if grouped_df.empty:
        print("Error: Grouped DataFrame is empty")
        return

    # Get environment info
    if args.env_exp_id:
        env_exp_id = args.env_exp_id
    else:
        # Use the first experiment ID if not specified
        env_exp_id = grouped_df.iloc[0]['exp_id']

    print(f"Using environment info from: {env_exp_id}")
    env_info = get_environment_info(env_exp_id, args.experiments_dir)

    # Generate markdown content
    md_content = generate_markdown_report(
        grouped_df,
        env_info,
        args.figs_dir,
        args.github_prefix
    )

    # Write output file
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Report generated: {args.out_md}")
    print(f"Used environment info from: {env_exp_id}")


if __name__ == "__main__":
    main()