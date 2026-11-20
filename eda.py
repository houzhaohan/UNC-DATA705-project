"""
eda.py
------
探索性数据分析模块。
输出：
  - Table 1 描述统计表 (CSV)
  - 图1：失业率直方图 + Q-Q 图
  - 图2：失业率时间序列（全州 + 3 个代表县）
  - 图3：相关系数热力图
  - 图4：月度季节性箱线图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import data_loader

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # STIX 与 Times New Roman 风格一致
plt.rcParams["axes.unicode_minus"] = False


def descriptive_table(wide, output_dir):
    """Table 1: 四个核心指标的描述统计（全州 + 县级合计）。"""
    metrics = ["Labor_Force", "Employment", "Unemployment", "Unemployment_Rate"]

    # 按 Area Type 分组统计
    rows = []
    for atype, sub in wide.groupby("Area Type"):
        for m in metrics:
            vals = sub[m].dropna()
            rows.append({
                "Area_Type": atype,
                "Variable": m,
                "N": len(vals),
                "Mean": round(vals.mean(), 2),
                "SD": round(vals.std(), 2),
                "Min": round(vals.min(), 2),
                "Median": round(vals.median(), 2),
                "Max": round(vals.max(), 2),
            })

    tbl = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "table1_descriptive.csv")
    tbl.to_csv(out_path, index=False)
    print(f"[eda] 描述统计表已保存: {out_path}")
    print(tbl.to_string(index=False))
    return tbl


def plot_unemployment_hist(model_df, output_dir):
    """图1：失业率直方图 + Q-Q 图（并排）。"""
    y = model_df["Unemployment_Rate"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图
    axes[0].hist(y, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Unemployment Rate (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) Distribution of Unemployment Rate")

    # Q-Q 图
    stats.probplot(y, dist="norm", plot=axes[1])
    axes[1].set_title("(b) Normal Q-Q Plot")
    axes[1].get_lines()[0].set_markerfacecolor("steelblue")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig1_unemployment_hist_qq.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eda] 图1已保存: {out_path}")


def plot_time_series(model_df, output_dir):
    """图2：失业率时间序列（全州 + 3 个代表县）。"""
    # 选 3 个代表县：一个大县（Wake）、一个小县（Tyrrell）、一个中等（Durham）
    sample_counties = ["Wake County", "Tyrrell County", "Durham County"]
    available = [c for c in sample_counties if c in model_df["County"].unique()]

    fig, ax = plt.subplots(figsize=(12, 5))

    for county in available:
        sub = model_df[model_df["County"] == county].sort_values("time_index")
        ax.plot(sub["time_index"], sub["Unemployment_Rate"], linewidth=0.8, label=county)

    # 标记 2008 金融危机和 2020 COVID
    ax.axvline(x=2008.5, color="red", linestyle="--", alpha=0.6, label="2008 Crisis")
    ax.axvline(x=2020.0, color="green", linestyle="--", alpha=0.6, label="2020 COVID")

    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("Unemployment Rate Over Time (Selected Counties)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig2_time_series.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eda] 图2已保存: {out_path}")


def plot_correlation(model_df, output_dir):
    """图3：数值特征相关系数热力图。"""
    cols = ["Unemployment_Rate", "Labor_Force", "Employment", "Unemployment", "time_index"]
    corr = model_df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # 标注数值
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig3_correlation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eda] 图3已保存: {out_path}")


def plot_seasonality(model_df, output_dir):
    """图4：月度失业率箱线图（季节性）。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = [model_df[model_df["Month"] == m]["Unemployment_Rate"].values for m in range(1, 13)]

    bp = ax.boxplot(data, labels=month_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("Month")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("Monthly Distribution of Unemployment Rate (Seasonality)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig4_seasonality.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eda] 图4已保存: {out_path}")


def run(model_df, wide, output_dir):
    """运行全部 EDA。"""
    descriptive_table(wide, output_dir)
    plot_unemployment_hist(model_df, output_dir)
    plot_time_series(model_df, output_dir)
    plot_correlation(model_df, output_dir)
    plot_seasonality(model_df, output_dir)
    print("[eda] 全部 EDA 完成")


if __name__ == "__main__":
    model_df, wide = data_loader.run()
    run(model_df, wide, data_loader.OUTPUT_DIR)
