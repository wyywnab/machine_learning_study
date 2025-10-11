#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a one-page markdown report:
- Experiment summary table (selected columns)
- Attach links to val_acc.png and a chosen loss figure
"""
import argparse
import pandas as pd
from pathlib import Path

COLUMNS = ["exp_name","seed","epochs","backbone","img_size","batch","optimizer","lr","lr_scheduler",
           "augment","gem","non_local","cbam","wrt","rank1","mAP","mINP","train_time_min","throughput_img_per_s"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_csv", type=Path, default=Path("experiments_template.csv"))
    ap.add_argument("--out_md", type=Path, default=Path("report.md"))
    ap.add_argument("--fig_dir", type=Path, default=Path("."))
    ap.add_argument("--loss_exp", type=str, default="baseline")
    args = ap.parse_args()

    df = pd.read_csv(args.exp_csv)
    md = []
    md.append("# Re-ID Weekly Result (One-Pager)\\n")
    md.append("## Summary Table\\n")
    sub = df[COLUMNS].copy()
    # round numbers if present
    for col in ["rank1","mAP","mINP","train_time_min","throughput_img_per_s","lr"]:
        if col in sub.columns:
            try:
                sub[col] = sub[col].astype(float).round(4)
            except:
                pass
    md.append(sub.to_markdown(index=False))
    md.append("\\n\\n## Curves\\n")
    md.append(f"![Val Acc](./{args.fig_dir}/val_acc.png)\\n\\n")
    md.append(f"![Loss {args.loss_exp}](./{args.fig_dir}/loss_{args.loss_exp}.png)\\n")
    args.out_md.write_text("\\n".join(md), encoding="utf-8")

if __name__ == "__main__":
    main()