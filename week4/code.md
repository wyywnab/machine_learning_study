# Create a lightweight, ready-to-run template pack for experiment logging & plotting
from pathlib import Path
from textwrap import dedent
import pandas as pd
import caas_jupyter_tools as cj

root = Path("/mnt/data/reid_exp_template")
root.mkdir(parents=True, exist_ok=True)

# 1) experiments_template.csv (summary across runs)
exp_csv = root / "experiments_template.csv"
df_exp = pd.DataFrame(
    [
        {
            "exp_name": "baseline",
            "seed": 42,
            "epochs": 120,
            "backbone": "ResNet50",
            "img_size": "256x128",
            "batch": 64,
            "optimizer": "SGD",
            "lr": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.0005,
            "augment": "basic",
            "gem": 0,
            "non_local": 0,
            "cbam": 0,
            "wrt": 0,
            "rank1": "",
            "mAP": "",
            "mINP": "",
            "train_time_min": "",
            "throughput_img_per_s": "",
            "commit_id": "",
            "weights_path": "",
            "notes": "",
        },
        {
            "exp_name": "baseline_gem",
            "seed": 42,
            "epochs": 120,
            "backbone": "ResNet50",
            "img_size": "256x128",
            "batch": 64,
            "optimizer": "SGD",
            "lr": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.0005,
            "augment": "basic",
            "gem": 1,
            "non_local": 0,
            "cbam": 0,
            "wrt": 0,
            "rank1": "",
            "mAP": "",
            "mINP": "",
            "train_time_min": "",
            "throughput_img_per_s": "",
            "commit_id": "",
            "weights_path": "",
            "notes": "",
        },
        {
            "exp_name": "baseline_gem_nonlocal",
            "seed": 42,
            "epochs": 120,
            "backbone": "ResNet50",
            "img_size": "256x128",
            "batch": 64,
            "optimizer": "SGD",
            "lr": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.0005,
            "augment": "basic",
            "gem": 1,
            "non_local": 1,
            "cbam": 0,
            "wrt": 0,
            "rank1": "",
            "mAP": "",
            "mINP": "",
            "train_time_min": "",
            "throughput_img_per_s": "",
            "commit_id": "",
            "weights_path": "",
            "notes": "",
        },
        {
            "exp_name": "baseline_gem_nonlocal_wrt",
            "seed": 42,
            "epochs": 120,
            "backbone": "ResNet50",
            "img_size": "256x128",
            "batch": 64,
            "optimizer": "SGD",
            "lr": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.0005,
            "augment": "basic",
            "gem": 1,
            "non_local": 1,
            "cbam": 0,
            "wrt": 1,
            "rank1": "",
            "mAP": "",
            "mINP": "",
            "train_time_min": "",
            "throughput_img_per_s": "",
            "commit_id": "",
            "weights_path": "",
            "notes": "",
        },
    ]
)
df_exp.to_csv(exp_csv, index=False)

# 2) curves_template.csv (per-epoch curves for plots)
curves_csv = root / "curves_template.csv"
df_curves = pd.DataFrame(
    [
        {"exp_name": "baseline", "epoch": 1, "train_loss": "", "val_loss": "", "val_acc": ""},
        {"exp_name": "baseline", "epoch": 2, "train_loss": "", "val_loss": "", "val_acc": ""},
        {"exp_name": "baseline_gem", "epoch": 1, "train_loss": "", "val_loss": "", "val_acc": ""},
        {"exp_name": "baseline_gem", "epoch": 2, "train_loss": "", "val_loss": "", "val_acc": ""},
    ]
)
df_curves.to_csv(curves_csv, index=False)

# 3) plot_curves.py (matplotlib only, one chart per figure, no styles/colors set)
plot_py = root / "plot_curves.py"
plot_code = dedent('''
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
''')
plot_py.write_text(plot_code, encoding="utf-8")

# 4) make_report.py: generate a one-pager markdown from experiments_template.csv + produced images
report_py = root / "make_report.py"
report_code = dedent('''
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
''')
report_py.write_text(report_code, encoding="utf-8")

# 5) README.md with quick usage
readme = root / "README.md"
readme.write_text(dedent('''
# Experiment Template (Re-ID / Classification friendly)

This pack gives you:
- `experiments_template.csv`: a summary table to log each run.
- `curves_template.csv`: per-epoch logs for plotting.
- `plot_curves.py`: generates two PNGs (`val_acc.png`, `loss_<exp>.png`).
- `make_report.py`: builds a one-page `report.md` from the CSV + images.

## Quickstart

1) Fill your results into `experiments_template.csv`.
2) Log per-epoch metrics into `curves_template.csv` (columns: exp_name, epoch, train_loss, val_loss, val_acc).
3) Make figures:
```bash
python plot_curves.py --curves_csv curves_template.csv --loss_exp baseline --out_dir figs



Running：

python make_report.py --exp_csv experiments_template.csv --out_md report.md --fig_dir figs --loss_exp baseline
# 1) 填写两份CSV（实验表+逐epoch曲线）
# 2) 生成曲线图（会得到 figs/val_acc.png 与 figs/loss_baseline.png）

python plot_curves.py --curves_csv curves_template.csv --loss_exp baseline --out_dir figs

# 3) 生成一页式结果（report.md）
python make_report.py --exp_csv experiments_template.csv --out_md report.md --fig_dir figs --loss_exp baseline
