#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Full imports =========
import pandas as pd
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt

# 你的工程里常见的依赖（保留以避免改动过大）
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Your local modules（按你的工程结构保留）
from single_lane_env import *
from utils import *
from Generator_model import *

# ========== IV/IEEE-friendly + 彩色单栏风格 ==========
# 单栏不超宽：3.25 in；紧凑高度：1.8 in；字体 8pt；细网格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,            # 基准 8pt
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============== Helpers ==============
def compute_weighted_mean_std(dist_dict: dict):
    """
    输入: dist_dict = {bin_center: probability}
    返回: (weighted_mean, weighted_std)
    """
    if not dist_dict:
        return float('nan'), float('nan')
    xs, ps = [], []
    for k, v in dist_dict.items():
        try:
            x = float(k); p = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(p):
            xs.append(x); ps.append(p)
    if not xs:
        return float('nan'), float('nan')
    xs = np.asarray(xs, dtype=float)
    ps = np.asarray(ps, dtype=float)
    s = ps.sum()
    if s <= 0:
        return float('nan'), float('nan')
    ps = ps / s
    mean = float(np.sum(xs * ps))
    var  = float(np.sum(ps * (xs - mean) ** 2))
    std  = math.sqrt(max(var, 0.0))
    return mean, std


def load_generated_from_csv(csv_path='generated_distribution.csv'):
    """
    读取生成分布 CSV: Bin Type, Bin, Probability
    返回: (gen_speed_dict, gen_distance_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}. Run the generator first.")
    gen_speed, gen_dist = {}, {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 3: 
                continue
            btype, b, p = row[0], row[1], row[2]
            try:
                b = float(b); p = float(p)
            except (TypeError, ValueError):
                continue
            t = (btype or "").strip().lower()
            if t.startswith('speed'):
                gen_speed[b] = gen_speed.get(b, 0.0) + p
            elif t.startswith('distance'):
                gen_dist[b]  = gen_dist.get(b, 0.0) + p
    return gen_speed, gen_dist


def load_gt_speed_distribution():
    """
    优先: load_gt_speed_hist_centered() -> (centers, probs)
    回退: load_distribution_from_csv()  -> (speed_dict, ...)
    """
    try:
        centers, probs = load_gt_speed_hist_centered()
        return {float(c): float(p) for c, p in zip(centers, probs)}
    except Exception:
        pass
    try:
        speed_distribution_gt, *_ = load_distribution_from_csv()
        return {float(k): float(v) for k, v in speed_distribution_gt.items()}
    except Exception as e:
        raise RuntimeError(
            "Failed to load GT speed distribution. "
            "Please provide load_gt_speed_hist_centered() or ensure "
            "load_distribution_from_csv() returns a speed dict."
        ) from e


def load_gt_distance_distribution():
    centers, probs = load_gt_distance_hist_centered()
    return {float(c): float(p) for c, p in zip(centers, probs)}


def save_summary_csv(rows, out_csv="mean_std_summary.csv"):
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Set", "Mean", "Std"])
        for r in rows:
            w.writerow(r)
    print(f"Saved summary: {out_csv}")

# ============== 彩色紧凑绘图（点+误差棒） ==============
def _annotate_values(ax, xs, means, stds):
    # 在误差棒上方标注 “mean±std”
    top = np.nanmax(np.array(means) + np.array(stds))
    rng = top - 0.0
    offset = 0.03 * (rng if np.isfinite(rng) and rng > 0 else 1.0)
    for x, m, s in zip(xs, means, stds):
        if np.isfinite(m) and np.isfinite(s):
            ax.text(x, m + s + offset, f"{m:.2f}±{s:.2f}",
                    ha="center", va="bottom", fontsize=7)

def plot_mean_std_pair_color(metric, unit, out_prefix,
                             gt_mean, gt_std, gen_mean, gen_std):
    """
    彩色、紧凑、单栏宽度（3.25 in）的 mean±std 点图。
    - y 轴从 0 开始
    - PNG(600dpi) + PDF
    """
    fig_w, fig_h = 3.25, 1.8  # 单栏且紧凑
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    labels = ["GT", "Generated"]
    xs     = np.arange(len(labels))
    means  = [gt_mean, gen_mean]
    stds   = [gt_std,  gen_std]

    # 颜色（符合 Matplotlib 默认调色板且打印友好）
    colors = ["tab:blue", "tab:orange"]

    # 绘制两类点+误差棒
    for i, (x, m, s) in enumerate(zip(xs, means, stds)):
        ax.errorbar([x], [m], yerr=[s],
                    fmt="o", markersize=4.0,
                    capsize=3, linewidth=1.0, elinewidth=0.9,
                    color=colors[i], mec=colors[i], mfc=colors[i])

    # 坐标轴与网格
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ylabel = f"{metric} [{unit}]" if unit else metric
    ax.set_ylabel(ylabel)

    # y 轴从 0 开始；上限留 5% 余量以容纳标注
    top = max([m + s for m, s in zip(means, stds)])
    ax.set_ylim(bottom=0.0, top=top * 1.05 if np.isfinite(top) else None)

    # 仅 y 方向虚线网格；轴在下层
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    # 干净边框
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(width=0.8, length=2)

    # 数值标注
    _annotate_values(ax, xs, means, stds)

    # 紧凑布局与输出
    plt.tight_layout(pad=0.25)
    fig.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_prefix}.png / {out_prefix}.pdf")


# ===================== Main =====================
def main(generated_csv='generated_distribution.csv'):
    # 1) 读取生成分布
    gen_speed_dist, gen_dist_dist = load_generated_from_csv(generated_csv)

    # 2) 读取 GT 分布
    gt_speed_dist = load_gt_speed_distribution()
    gt_dist_dist  = load_gt_distance_distribution()

    # 3) 统计量
    gt_speed_mean, gt_speed_std   = compute_weighted_mean_std(gt_speed_dist)
    gen_speed_mean, gen_speed_std = compute_weighted_mean_std(gen_speed_dist)

    gt_dist_mean, gt_dist_std     = compute_weighted_mean_std(gt_dist_dist)
    gen_dist_mean, gen_dist_std   = compute_weighted_mean_std(gen_dist_dist)

    # 4) 打印
    print("\n=== Mean ± Std (GT vs Generated) ===")
    print(f"Speed    : GT = {gt_speed_mean:.3f} ± {gt_speed_std:.3f} ; "
          f"Generated = {gen_speed_mean:.3f} ± {gen_speed_std:.3f}")
    print(f"Distance : GT = {gt_dist_mean:.3f} ± {gt_dist_std:.3f} ; "
          f"Generated = {gen_dist_mean:.3f} ± {gen_dist_std:.3f}")

    # 5) 绘图（彩色 + y 轴从 0 开始 + 紧凑单栏）
    plot_mean_std_pair_color(
        metric="Speed", unit="m/s",
        out_prefix=f"speed_mean_std_color_compact_GEN{generate_vehicle_number}",
        gt_mean=gt_speed_mean, gt_std=gt_speed_std,
        gen_mean=gen_speed_mean, gen_std=gen_speed_std
    )


    plot_mean_std_pair_color(
        metric="Adjacent Distance", unit="m",
        out_prefix=f"distance_mean_std_color_compact_GEN{generate_vehicle_number}",
        gt_mean=gt_dist_mean, gt_std=gt_dist_std,
        gen_mean=gen_dist_mean, gen_std=gen_dist_std
    )

    # 6) 保存 CSV 汇总
    rows = [
        ["Speed",    "GT",        f"{gt_speed_mean:.6f}", f"{gt_speed_std:.6f}"],
        ["Speed",    "Generated", f"{gen_speed_mean:.6f}", f"{gen_speed_std:.6f}"],
        ["Distance", "GT",        f"{gt_dist_mean:.6f}",  f"{gt_dist_std:.6f}"],
        ["Distance", "Generated", f"{gen_dist_mean:.6f}", f"{gen_dist_std:.6f}"],
    ]
    save_summary_csv(rows, out_csv=f"mean_std_summary_GEN{generate_vehicle_number}.csv")


if __name__ == "__main__":
    # 若 CSV 不在当前目录，可传绝对/相对路径
    generate_vehicle_number = 1  # ← 按需修改
    main(f"generated_distribution_GEN{generate_vehicle_number}.csv")
