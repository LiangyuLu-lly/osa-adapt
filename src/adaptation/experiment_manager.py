"""
实验管理器：ExperimentManager

管理实验配置生成（笛卡尔积）、执行调度、结果收集和断点续跑。
所有实验结果保存在 output_dir/results/ 下。
"""

import json
import logging
import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .models import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    实验管理器

    管理实验配置生成、执行调度、结果收集和可复现性保证。
    所有实验结果保存在 output_dir/results/ 下。
    """

    def __init__(self, output_dir: str = "experiments/paper3_osa_adapt"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_configs(
        self,
        models: List[str],
        methods: List[str],
        budgets: List[int],
        n_folds: int = 5,
        n_seeds: int = 5,
    ) -> List[ExperimentConfig]:
        """
        生成所有实验配置的笛卡尔积。

        对 models × methods × budgets × folds × seeds 做笛卡尔积，
        每个组合生成一个 ExperimentConfig，experiment_name 唯一标识。

        Returns:
            所有实验配置的列表
        """
        configs: List[ExperimentConfig] = []
        for model, method, budget in itertools.product(models, methods, budgets):
            for fold in range(n_folds):
                for seed_idx in range(n_seeds):
                    seed = 42 + seed_idx
                    name = f"{method}_{model}_budget{budget}_fold{fold}_seed{seed}"
                    config = ExperimentConfig(
                        experiment_name=name,
                        model_name=model,
                        adaptation_method=method,
                        data_budget=budget,
                        fold=fold,
                        seed=seed,
                    )
                    configs.append(config)
        return configs

    def is_completed(self, config: ExperimentConfig) -> bool:
        """检查实验是否已完成（结果文件存在）"""
        result_path = self.results_dir / f"{config.experiment_name}.json"
        return result_path.exists()

    def save_result(self, config: ExperimentConfig, result: Dict) -> Path:
        """
        保存实验结果到 JSON 文件。

        Args:
            config: 实验配置
            result: 实验结果字典

        Returns:
            保存的文件路径
        """
        result_path = self.results_dir / f"{config.experiment_name}.json"
        data = {
            "config": asdict(config),
            "result": result,
        }
        result_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"结果已保存: {result_path}")
        return result_path

    def run_experiment(self, config: ExperimentConfig) -> Dict:
        """
        执行单个实验并返回结果。

        这是一个骨架实现：创建输出目录、保存配置、返回占位结果。
        实际训练逻辑由实验脚本提供。

        Args:
            config: 实验配置

        Returns:
            结果字典
        """
        # 断点续跑：跳过已完成的实验
        if self.is_completed(config):
            logger.info(f"实验已完成，跳过: {config.experiment_name}")
            result_path = self.results_dir / f"{config.experiment_name}.json"
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return data.get("result", {})

        # 创建实验输出目录
        exp_dir = self.output_dir / config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = exp_dir / "config.json"
        config_path.write_text(config.to_json(), encoding="utf-8")

        # 占位结果（实际训练逻辑在实验脚本中实现）
        result: Dict = {
            "status": "placeholder",
            "experiment_name": config.experiment_name,
            "model_name": config.model_name,
            "adaptation_method": config.adaptation_method,
            "data_budget": config.data_budget,
            "fold": config.fold,
            "seed": config.seed,
        }

        # 保存结果
        self.save_result(config, result)
        return result

    def collect_results(self) -> pd.DataFrame:
        """
        收集所有已完成实验的结果为 DataFrame。

        扫描 results/ 目录下的所有 JSON 文件，加载并合并为一个 DataFrame。
        每行包含配置字段和结果字段。

        Returns:
            包含所有实验结果的 DataFrame，无结果时返回空 DataFrame
        """
        records: List[Dict] = []
        if not self.results_dir.exists():
            return pd.DataFrame()

        for result_file in sorted(self.results_dir.glob("*.json")):
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                record: Dict = {}
                # 展平配置字段
                if "config" in data:
                    record.update(data["config"])
                # 展平结果字段
                if "result" in data:
                    for k, v in data["result"].items():
                        if k not in record:
                            record[k] = v
                records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"跳过损坏的结果文件 {result_file}: {e}")
                continue

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def get_pending_configs(
        self, configs: List[ExperimentConfig]
    ) -> List[ExperimentConfig]:
        """
        从配置列表中筛选出尚未完成的实验。

        Args:
            configs: 全部实验配置列表

        Returns:
            未完成的实验配置列表
        """
        return [c for c in configs if not self.is_completed(c)]

    def run_all(
        self,
        configs: List[ExperimentConfig],
        skip_completed: bool = True,
    ) -> List[Dict]:
        """
        批量执行实验，支持断点续跑。

        Args:
            configs: 实验配置列表
            skip_completed: 是否跳过已完成的实验

        Returns:
            所有实验结果列表
        """
        results: List[Dict] = []
        pending = self.get_pending_configs(configs) if skip_completed else configs
        total = len(configs)
        skipped = total - len(pending)

        if skipped > 0:
            logger.info(f"跳过 {skipped}/{total} 个已完成实验")

        for i, config in enumerate(pending):
            logger.info(
                f"执行实验 [{i + 1}/{len(pending)}]: {config.experiment_name}"
            )
            result = self.run_experiment(config)
            results.append(result)

        return results
