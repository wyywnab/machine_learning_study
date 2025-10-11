#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read curves_template.csv and generate two separate charts:
1) Validation accuracy vs. epoch (saved as val_acc.png)
2) Train & Val loss vs. epoch for a chosen experiment (saved as loss_{exp}.png)
Rules: matplotlib only, one chart per plot, no explicit colors or styles.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_val_acc(df, out_path):
    plt.figure()
    for exp, g in df.groupby("exp_name"):
        g = g.sort_values("epoch")
        if g["val_acc"].notna().any():
            plt.plot(g["epoch"], g["val_acc"], label=exp)
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def plot_loss_for_exp(df, exp_name, out_path):
    sub = df[df["exp_name"] == exp_name].sort_values("epoch")
    plt.figure()
    if sub["train_loss"].notna().any():
        plt.plot(sub["epoch"], sub["train_loss"], label="train_loss")
    if sub["val_loss"].notna().any():
        plt.plot(sub["epoch"], sub["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_csv", type=Path, default=Path("curves_template.csv"))
    ap.add_argument("--loss_exp", type=str, default="baseline")
    ap.add_argument("--out_dir", type=Path, default=Path("."))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.curves_csv)

    plot_val_acc(df, args.out_dir / "val_acc.png")
    plot_loss_for_exp(df, args.loss_exp, args.out_dir / f"loss_{args.loss_exp}.png")

if __name__ == "__main__":
    main()