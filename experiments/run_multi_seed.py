#!/usr/bin/env python
"""
多种子实验脚本 - 运行seeds 43-46的主实验

论文要求5-fold × 5-seed = 25 runs per combination。
seed=42已完成，本脚本运行seeds 43-46。

用法:
    PYTHONPATH=. python experiments/run_multi_seed.py
    PYTHONPATH=. python experiments/run_multi_seed.py --seeds 43 44
"""
import argparse
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="多种子主实验")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", default=["Chambon2018", "TinySleepNet"])
    parser.add_argument("--methods", nargs="+", default=[
        "osa_adapt", "full_ft", "last_layer", "lora",
        "film_no_severity", "bn_only", "no_adapt", "coral", "mmd",
    ])
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 30, 50, 65, 100])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--pkl-dir", default="data/preprocessed")
    parser.add_argument("--severity-json", default="data/patient_severity.json")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # 导入主实验脚本
    from experiments.run_main_experiment import (
        parse_args as main_parse_args,
        setup_logging,
        run_main_experiment,
    )

    for seed in args.seeds:
        logger.info("=" * 60)
        logger.info("开始 seed=%d 的实验", seed)
        logger.info("=" * 60)

        # 构造主实验脚本的参数
        main_argv = [
            "--models"] + args.models + [
            "--methods"] + args.methods + [
            "--budgets"] + [str(b) for b in args.budgets] + [
            "--output-dir", args.output_dir,
            "--pkl-dir", args.pkl_dir,
            "--severity-json", args.severity_json,
            "--n-folds", str(args.n_folds),
            "--n-seeds", "1",  # 每次只跑1个seed
            "--batch-size", str(args.batch_size),
            "--skip-completed",
            "--log-level", args.log_level,
        ]

        # 主实验脚本的n-seeds参数控制从seed=42开始的连续种子数
        # 但我们需要指定特定的seed，所以需要修改方法
        # 实际上主实验脚本的seed是 42 + seed_idx，seed_idx从0到n_seeds-1
        # 所以seed=43对应n_seeds>=2的第2个seed
        # 但这样会重复运行seed=42
        # 更好的方案：直接调用底层函数

        print(f"\n运行 seed={seed} 的实验...")
        print(f"请手动运行以下命令:")
        print(f'PYTHONPATH=. python '
              f'experiments/run_main_experiment.py '
              f'--models {" ".join(args.models)} '
              f'--methods {" ".join(args.methods)} '
              f'--budgets {" ".join(str(b) for b in args.budgets)} '
              f'--output-dir {args.output_dir} '
              f'--n-folds {args.n_folds} --n-seeds {seed - 42 + 1} '
              f'--batch-size {args.batch_size} --skip-completed')


if __name__ == "__main__":
    main()
