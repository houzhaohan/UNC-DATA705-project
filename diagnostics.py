"""
diagnostics.py
--------------
模型诊断模块。
输出：
  - 图5：OLS 残差诊断三联图（Residuals vs Fitted | Normal Q-Q | Scale-Location）
  - 图6：Elastic Net CV 误差曲线（按 l1_ratio 分组）
  - 图7：测试集预测 vs 实际值散点图（OLS vs Ridge vs Elastic Net）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import data_loader
import model as model_module

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # STIX 与 Times New Roman 风格一致
plt.rcParams["axes.unicode_minus"] = False


def plot_ols_residuals(ols_model, X_train, y_train, output_dir):
    """图5：OLS 残差诊断三联图。"""
    fitted = ols_model.fittedvalues
    residuals = ols_model.resid
    std_resid = residuals / np.sqrt(ols_model.mse_resid)  # studentized residuals

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.4, s=10)
    axes[0].axhline(y=0, color="red", linestyle="--")
    # LOWESS 平滑线
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        z = lowess(residuals, fitted, frac=0.3)
        axes[0].plot(z[:, 0], z[:, 1], color="blue", linewidth=2)
    except Exception:
        pass
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("(a) Residuals vs Fitted")

    # (b) Normal Q-Q
    stats.probplot(std_resid, dist="norm", plot=axes[1])
    axes[1].set_title("(b) Normal Q-Q Plot (Std Residuals)")

    # (c) Scale-Location
    axes[2].scatter(fitted, np.abs(std_resid), alpha=0.4, s=10)
    axes[2].axhline(y=np.mean(np.abs(std_resid)), color="red", linestyle="--")
    axes[2].set_xlabel("Fitted values")
    axes[2].set_ylabel("|Standardized Residuals|")
    axes[2].set_title("(c) Scale-Location")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fig5_ols_diagnostics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[diagnostics] 图5已保存: {out_path}")


def plot_enet_cv(enet, output_dir):
    """
    图6：Elastic Net CV 误差曲线。
    mse_path_ shape: (n_l1_ratio, n_alphas, n_folds)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    l1_ratios = list(enet.l1_ratio) if hasattr(enet.l1_ratio, "__iter__") else [enet.l1_ratio]
    # mse_path_ shape: (n_l1_ratio, n_alphas, n_folds) → 平均 folds → (n_l1_ratio, n_alphas)
    mse_path = np.mean(enet.mse_path_, axis=-1)
    alphas_arr = np.array(enet.alphas)
    log_alphas = -np.log10(alphas_arr + 1e-10)

    for i, lr in enumerate(l1_ratios):
        ax.plot(log_alphas, mse_path[i], marker="o", markersize=3,
                label=f"l1_ratio={lr}")

    # 标注最优 alpha
    best_log = -np.log10(enet.alpha_ + 1e-10)
    ax.axvline(x=best_log, color="black", linestyle=":", alpha=0.7,
               label=f"Best α={enet.alpha_:.2e}")

    ax.set_xlabel("-log10(Alpha)")
    ax.set_ylabel("CV MSE (averaged over folds)")
    ax.set_title("Elastic Net CV Error vs Regularization Strength")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig6_enet_cv.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[diagnostics] 图6已保存: {out_path}")


def plot_test_predictions(ols_model, ridge, ridge_scaler, enet, enet_scaler,
                          X_test, y_test, output_dir):
    """图7：测试集预测 vs 实际（OLS / Ridge / Elastic Net 三模型对比）。"""
    import statsmodels.api as sm

    X_te_ols = sm.add_constant(X_test)
    y_ols_pred = ols_model.predict(X_te_ols)

    X_te_ridge = ridge_scaler.transform(X_test)
    y_ridge_pred = ridge.predict(X_te_ridge)

    X_te_enet = enet_scaler.transform(X_test)
    y_enet_pred = enet.predict(X_te_enet)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    all_preds = [y_ols_pred, y_ridge_pred, y_enet_pred]
    max_val = max(y_test.max(), max(p.max() for p in all_preds))

    titles = [
        r"OLS ($R^2$=" + f"{r2_score(y_test, y_ols_pred):.3f})",
        r"Ridge ($R^2$=" + f"{r2_score(y_test, y_ridge_pred):.3f})",
        r"Elastic Net ($R^2$=" + f"{r2_score(y_test, y_enet_pred):.3f})",
    ]

    for ax, y_pred, title in zip(axes, all_preds, titles):
        ax.scatter(y_test, y_pred, alpha=0.4, s=10, color="steelblue")
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="y = x")
        ax.set_xlabel("Actual Unemployment Rate (%)")
        ax.set_ylabel("Predicted Unemployment Rate (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Test Set: Predicted vs Actual", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "fig7_pred_vs_actual.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[diagnostics] 图7已保存: {out_path}")


def run(model_dict, output_dir):
    """运行全部诊断图。"""
    plot_ols_residuals(
        model_dict["ols_model"],
        model_dict["X_train"],
        model_dict["y_train"],
        output_dir,
    )
    plot_enet_cv(model_dict["enet"], output_dir)
    plot_test_predictions(
        model_dict["ols_model"],
        model_dict["ridge"],
        model_dict["ridge_scaler"],
        model_dict["enet"],
        model_dict["enet_scaler"],
        model_dict["X_test"],
        model_dict["y_test"],
        output_dir,
    )
    print("[diagnostics] 全部诊断图完成")


if __name__ == "__main__":
    model_df, _ = data_loader.run()
    model_dict = model_module.run(model_df, data_loader.OUTPUT_DIR)
    run(model_dict, data_loader.OUTPUT_DIR)
