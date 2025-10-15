#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a one-page markdown report:
- Experiment summary table (selected columns)
- Environment information (torch/cuda/torchvision versions, GPU, driver)
- Attach links to val_acc.png and a chosen loss figure
"""
import argparse
import os
import yaml
import sys

import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plot_curves import plot_val_acc, plot_loss_for_exp

COLUMNS = ["exp_id", "exp_name", "seed", "epochs", "learning_rate", "optimizer",
           "lr_scheduler", "data_enhancement", "cbam_enabled", "top-1_acc", "duration"]


def get_environment_info(exp_id, experiments_dir):
    info_file = experiments_dir / exp_id / "information.yaml"

    if not info_file.exists():
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
        return env_info
    except Exception as e:
        print(f"Error reading environment info from {info_file}: {e}")
        return None


def ensure_figures_exist(curves_csv, loss_exp, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not curves_csv.exists():
        print(f"Warning: Curves CSV file not found at {curves_csv}")
        return False

    try:
        df = pd.read_csv(curves_csv)

        val_acc_path = fig_dir / "val_acc.png"
        if not val_acc_path.exists():
            print(f"Generating validation accuracy plot: {val_acc_path}")
            plot_val_acc(df, val_acc_path)

        loss_path = fig_dir / f"loss_{loss_exp}.png"
        if not loss_path.exists() and loss_exp in df['exp_name'].values:
            print(f"Generating loss plot for {loss_exp}: {loss_path}")
            plot_loss_for_exp(df, loss_exp, loss_path)
        elif loss_exp not in df['exp_name'].values:
            print(f"Warning: Experiment '{loss_exp}' not found in curves data")
            return False

        return True
    except Exception as e:
        print(f"Error generating figures: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_csv", type=Path, default=Path(os.path.join("scripts", "experiments", "experiments.csv")))
    ap.add_argument("--out_md", type=Path, default=Path("report.md"))
    ap.add_argument("--fig_dir", type=Path, default=Path("figs"))
    ap.add_argument("--loss_exp", type=str, default="baseline_enhancement")
    ap.add_argument("--experiments_dir", type=Path, default=Path(os.path.join("scripts", "experiments")))
    ap.add_argument("--curves_csv", type=Path, default=Path(os.path.join("scripts", "experiments", "all_curves.csv")))
    args = ap.parse_args()

    figures_exist = ensure_figures_exist(args.curves_csv, args.loss_exp, args.fig_dir)

    df = pd.read_csv(args.exp_csv)
    md = []
    md.append("# Weekly Result (One-Pager)\n")

    md.append("## Environment Information\n")

    loss_exp_info = None
    if args.loss_exp in df['exp_name'].values:
        exp_id = df[df['exp_name'] == args.loss_exp]['exp_id'].iloc[0]
        loss_exp_info = get_environment_info(exp_id, args.experiments_dir)

    if loss_exp_info:
        env_table = f"""
| Item | Version |
|------|---------|
| PyTorch | {loss_exp_info['torch_version']} |
| TorchVision | {loss_exp_info['torchvision_version']} |
| CUDA | {loss_exp_info['cuda_version']} |
| Driver | {loss_exp_info['driver_version']} |
| GPU | {loss_exp_info['graphic_card']} |\n
"""
        md.append(env_table)
    else:
        md.append("*Environment information not available for the specified experiment*\n")

    md.append("## Summary Table\n")
    sub = df[COLUMNS].copy()
    numeric_columns = ["seed", "epochs", "learning_rate", "top-1_acc", "duration"]
    for col in numeric_columns:
        if col in sub.columns:
            try:
                sub[col] = sub[col].astype(float).round(4)
            except:
                pass
    md.append(sub.to_markdown(index=False))

    md.append("\n\n## Curves\n")

    val_acc_path = args.fig_dir / "val_acc.png"
    if val_acc_path.exists():
        md.append(f"### Validation Accuracy\n")
        md.append(f"![*Figure: Validation accuracy across all experiments0*]({val_acc_path.as_posix()})\n")
    else:
        md.append(f"### Validation Accuracy\n")
        md.append("*Validation accuracy plot not available*\n\n")

    loss_path = args.fig_dir / f"loss_{args.loss_exp}.png"
    if loss_path.exists():
        md.append(f"### Loss Curves for {args.loss_exp}\n")
        md.append(f"![*Figure: Training and validation loss for experiment '{args.loss_exp}'*]({loss_path.as_posix()})")
    else:
        md.append(f"### Loss Curves for {args.loss_exp}\n")
        md.append(f"*Loss plot for experiment '{args.loss_exp}' not available*\n\n")

    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Report generated: {args.out_md}")

    if figures_exist:
        print("Figures included in the report:")
        if val_acc_path.exists():
            print(f"  - {val_acc_path}")
        if loss_path.exists():
            print(f"  - {loss_path}")
    else:
        print("Warning: Some figures could not be generated or found")


if __name__ == "__main__":
    main()