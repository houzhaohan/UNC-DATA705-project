"""
data_loader.py
-------------
数据加载、清洗与宽格式转换模块。
原始数据为长格式 (Area Name; Area Type; Year; Month; Variable; Value; date)，
需要 pivot 成宽格式，每行对应 (Area, Year, Month) 组合。
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "monthly-labor-force-linc.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_raw(path=DATA_PATH):
    """加载原始 CSV（分号分隔，UTF-8 BOM）。"""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    print(f"[data_loader] 原始数据 shape: {df.shape}")
    return df


def clean(df):
    """
    基础清洗：
    - 去掉 Area Type 为 NaN 的行
    - 去掉 Value 为 NaN 的行
    - 将 Year/Month 转为 int
    - 过滤掉不完整的 2026 年数据（只保留 <=2025）
    """
    df = df.dropna(subset=["Area Type", "Value"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df = df[df["Year"] <= 2025].copy()
    print(f"[data_loader] 清洗后 shape: {df.shape}")
    return df


def pivot(df):
    """
    长格式 → 宽格式。
    输出列：Area Name, Area Type, Year, Month, Labor Force, Employment, Unemployment, Unemployment Rate
    """
    rename_map = {
        "Monthly Labor Force (Place of Residence)": "Labor_Force",
        "Monthly Employment (Place of Residence)": "Employment",
        "Monthly Unemployment (Place of Residence)": "Unemployment",
        "Monthly Unemployment Rate (Place of Residence)(Percent)": "Unemployment_Rate",
    }
    df = df[df["Variable"].isin(rename_map.keys())].copy()
    df["Variable"] = df["Variable"].map(rename_map)

    wide = df.pivot_table(
        index=["Area Name", "Area Type", "Year", "Month"],
        columns="Variable",
        values="Value",
        aggfunc="first",
    ).reset_index()

    # 重置列名（去掉 Variable 层级）
    wide.columns.name = None

    # 在宽表层面也过滤异常失业率（录入错误：把人数填成了百分比）
    if "Unemployment_Rate" in wide.columns:
        n_before = len(wide)
        wide = wide[(wide["Unemployment_Rate"] <= 25.0) |
                     (wide["Unemployment_Rate"].isna())].copy()
        n_after = len(wide)
        if n_before != n_after:
            print(f"[data_loader] wide 表过滤异常 Unemployment_Rate > 25%: "
                  f"{n_before - n_after} 行移除")

    print(f"[data_loader] pivot 后 shape: {wide.shape}")
    return wide


def build_model_dataset(wide):
    """
    构建建模用数据集：
    - 过滤掉缺失 Unemployment_Rate 的行（outcome 不能缺失）
    - 过滤 Unemployment_Rate 的异常值（>25%，数据录入错误）
    - 添加 time_index（Year + Month/12.0）捕捉时间趋势
    - 添加 County dummy（字符串形式，建模时 one-hot）
    """
    model_df = wide.dropna(subset=["Unemployment_Rate"]).copy()

    # 异常值过滤：失业率不可能 >25%（2025-12 数据录入错误：把人数填成了百分比）
    n_before = len(model_df)
    model_df = model_df[model_df["Unemployment_Rate"] <= 25.0].copy()
    n_after = len(model_df)
    if n_before != n_after:
        print(f"[data_loader] 过滤异常 Unemployment_Rate > 25%: {n_before - n_after} 行移除")

    model_df["time_index"] = model_df["Year"] + (model_df["Month"] - 1) / 12.0

    # 去掉全州汇总，只用 county 级数据（面板结构更统一）
    model_df = model_df[model_df["Area Type"] == "County"].copy()
    model_df = model_df.rename(columns={"Area Name": "County"})

    print(f"[data_loader] 建模数据集 shape: {model_df.shape}")
    print(f"[data_loader] County 数量: {model_df['County'].nunique()}")
    print(f"[data_loader] Year 范围: {model_df['Year'].min()} - {model_df['Year'].max()}")
    print(f"[data_loader] Unemployment_Rate 范围: "
          f"{model_df['Unemployment_Rate'].min():.2f} - "
          f"{model_df['Unemployment_Rate'].max():.2f}")
    return model_df


def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def run():
    """完整数据加载管线，返回建模用 DataFrame。"""
    make_output_dir()
    raw = load_raw()
    cleaned = clean(raw)
    wide = pivot(cleaned)
    model_df = build_model_dataset(wide)
    return model_df, wide


if __name__ == "__main__":
    model_df, wide = run()
    print("\n建模数据集预览：")
    print(model_df.head())
    print("\n建模数据集列：", model_df.columns.tolist())
