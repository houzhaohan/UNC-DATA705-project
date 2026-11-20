"""
run_all.py
----------
主入口：串联 data_loader → eda → model → diagnostics 全部模块。
输出全部保存在 ./output/ 目录。
"""

import os
import sys
import data_loader
import eda
import model as model_module
import diagnostics


def main():
    print("=" * 60)
    print(" GLM Assignment — North Carolina Unemployment Analysis")
    print("=" * 60)

    output_dir = data_loader.OUTPUT_DIR

    # Step 0: 加载 + 清洗数据
    print("\n" + "─" * 40)
    print("Step 0: Data Loading & Cleaning")
    print("─" * 40)
    model_df, wide = data_loader.run()

    # Step 1-2: EDA
    print("\n" + "─" * 40)
    print("Step 1-2: Exploratory Data Analysis")
    print("─" * 40)
    eda.run(model_df, wide, output_dir)

    # Step 3-4: 建模
    print("\n" + "─" * 40)
    print("Step 3-4: Model Fitting")
    print("─" * 40)
    model_dict = model_module.run(model_df, output_dir)

    # Step 5: 诊断
    print("\n" + "─" * 40)
    print("Step 5: Model Diagnostics")
    print("─" * 40)
    diagnostics.run(model_dict, output_dir)

    # 汇总输出
    print("\n" + "=" * 60)
    print(" 全部完成！输出文件列表：")
    print("=" * 60)
    for f in sorted(os.listdir(output_dir)):
        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  {f:40s} ({size_kb:7.1f} KB)")


if __name__ == "__main__":
    main()
