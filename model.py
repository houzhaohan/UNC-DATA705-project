"""
model.py
--------
Gaussian GLM + Elastic Net 正则化建模模块。

流程：
  1. 按时间切分 train（<=2020）/ test（>=2021）
  2. 构造设计矩阵（Year + Month dummy + County dummy）
  3. 拟合无正则化 OLS（statsmodels）
  4. 拟合 Elastic Net（sklearn）+ 10-fold CV 同时调 alpha 和 l1_ratio
  5. 评估 + 保存模型系数
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import data_loader

# 抑制 Ridge 坐标下降警告（我们用 RidgeCV 单独处理 Ridge）
warnings.filterwarnings("ignore", message="Coordinate descent without L1")
warnings.filterwarnings("ignore", message="Objective did not converge")


FEATURE_COLS = ["Year", "Month"]  # 基础特征；County 自动 one-hot
OUTCOME_COL = "Unemployment_Rate"
SPLIT_YEAR = 2020  # 训练集 <= 该年，测试集 > 该年


def time_split(model_df, split_year=SPLIT_YEAR):
    """按时间切分 train/test。"""
    train = model_df[model_df["Year"] <= split_year].copy()
    test = model_df[model_df["Year"] > split_year].copy()
    print(f"[model] Train: {train.shape[0]} obs (Year <= {split_year})")
    print(f"[model] Test:  {test.shape[0]} obs (Year >  {split_year})")
    return train, test


def build_design_matrix(df, county_categories=None):
    """
    构造设计矩阵：
      - Year（连续）+ Month（one-hot, drop first）+ County（one-hot, drop first）
    返回 X (DataFrame) 和 y (Series)。
    """
    # Month one-hot
    month_dummies = pd.get_dummies(df["Month"], prefix="Month", drop_first=True, dtype=float)

    # County one-hot（训练集的 categories 固定，测试集对齐）
    if county_categories is None:
        county_categories = sorted(df["County"].unique())
    county_series = pd.Categorical(df["County"], categories=county_categories)
    county_dummies = pd.get_dummies(county_series, prefix="County", drop_first=True, dtype=float)

    X = pd.concat([df[["Year"]].reset_index(drop=True),
                   month_dummies.reset_index(drop=True),
                   county_dummies.reset_index(drop=True)], axis=1)
    y = df[OUTCOME_COL].reset_index(drop=True)
    return X, y, county_categories


def fit_ols(X_train, y_train, X_test, y_test):
    """无正则化 Gaussian GLM（即 OLS），用 statsmodels 方便看系数显著性。"""
    X_tr = sm.add_constant(X_train)
    X_te = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_tr).fit()
    y_pred = model.predict(X_te)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print("\n[model] OLS (no regularization) 测试集表现:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics


def fit_elastic_net(X_train, y_train, X_test, y_test):
    """
    Elastic Net + 10-fold CV。
    同时调 l1_ratio (0 < l1_ratio <= 1，不包含纯 Ridge) 和 regularization strength。
    """
    # 标准化（Elastic Net 需要）
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    # l1_ratio 网格：去掉 0（纯 Ridge 用 RidgeCV 单独处理）
    l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    # 手动 alpha 网格（对数间距），覆盖从弱到强的正则化
    alphas = np.logspace(-4, 1, 60)

    enet = ElasticNetCV(
        l1_ratio=l1_ratio_grid,
        alphas=alphas,
        cv=10,
        random_state=42,
        max_iter=50000,
        tol=1e-4,
    )
    enet.fit(X_tr_scaled, y_train)
    y_pred = enet.predict(X_te_scaled)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print(f"\n[model] Elastic Net 测试集表现:")
    print(f"  最优 l1_ratio: {enet.l1_ratio_}")
    print(f"  最优 alpha (lambda): {enet.alpha_:.6f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return enet, scaler, metrics


def fit_ridge(X_train, y_train, X_test, y_test):
    """
    Ridge 回归 (纯 L2 正则) + 10-fold CV。
    RidgeCV 用解析解，不受坐标下降限制。
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    alphas = np.logspace(-4, 1, 50)
    ridge = RidgeCV(alphas=alphas, cv=10)
    ridge.fit(X_tr_scaled, y_train)
    y_pred = ridge.predict(X_te_scaled)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print(f"\n[model] Ridge 测试集表现:")
    print(f"  最优 alpha: {ridge.alpha_:.6f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return ridge, scaler, metrics


def save_ols_coefficients(ols_model, output_dir):
    """保存 OLS 系数表（含 p-value）。"""
    coef_df = pd.DataFrame({
        "feature": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "t_value": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    })
    out_path = os.path.join(output_dir, "table2_ols_coefficients.csv")
    coef_df.to_csv(out_path, index=False)
    print(f"[model] OLS 系数表已保存: {out_path}")
    return coef_df


def save_enet_coefficients(enet, X_train, output_dir):
    """保存 Elastic Net 非零系数。"""
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef_original_scale": enet.coef_,
    })
    nonzero = coef_df[coef_df["coef_original_scale"].abs() > 1e-5].sort_values(
        "coef_original_scale", key=abs, ascending=False
    )
    out_path = os.path.join(output_dir, "table3_enet_coefficients.csv")
    nonzero.to_csv(out_path, index=False)
    print(f"[model] Elastic Net 非零系数已保存: {out_path} ({len(nonzero)}/{len(coef_df)} 非零)")
    return nonzero


def save_metrics(ols_metrics, ridge_metrics, enet_metrics, output_dir):
    """保存模型对比表。"""
    rows = [
        {"Model": "OLS (no penalty)", **ols_metrics},
        {"Model": "Ridge (CV, L2)", **ridge_metrics},
        {"Model": "Elastic Net (CV)", **enet_metrics},
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "table4_model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[model] 模型对比表已保存: {out_path}")
    print(df.to_string(index=False))


def run(model_df, output_dir):
    """完整建模管线。"""
    train, test = time_split(model_df)

    X_train, y_train, county_cats = build_design_matrix(train)
    X_test, y_test, _ = build_design_matrix(test, county_categories=county_cats)

    print(f"[model] 设计矩阵: {X_train.shape[1]} 特征")

    ols_model, ols_metrics = fit_ols(X_train, y_train, X_test, y_test)
    ridge, ridge_scaler, ridge_metrics = fit_ridge(X_train, y_train, X_test, y_test)
    enet, enet_scaler, enet_metrics = fit_elastic_net(X_train, y_train, X_test, y_test)

    save_ols_coefficients(ols_model, output_dir)
    save_enet_coefficients(enet, X_train, output_dir)
    save_metrics(ols_metrics, ridge_metrics, enet_metrics, output_dir)

    return {
        "ols_model": ols_model,
        "ridge": ridge,
        "ridge_scaler": ridge_scaler,
        "enet": enet,
        "enet_scaler": enet_scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "county_categories": county_cats,
    }


if __name__ == "__main__":
    model_df, _ = data_loader.run()
    run(model_df, data_loader.OUTPUT_DIR)
