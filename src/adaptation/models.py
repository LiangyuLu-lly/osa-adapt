"""
核心数据模型：ExperimentConfig 和 AdaptationResult

定义实验配置和适应结果的数据类，支持JSON序列化/反序列化。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class ExperimentConfig:
    """实验配置（可序列化为JSON）"""

    experiment_name: str
    model_name: str
    adaptation_method: str  # 'osa_adapt', 'full_ft', 'last_layer', 'lora', 'film_no_severity', 'bn_only', 'no_adapt'
    data_budget: int  # 适应集患者数
    fold: int  # CV折数 (0-4)
    seed: int  # 随机种子
    # FiLM超参数（已优化）
    condition_dim: int = 64
    lr: float = 5e-5  # 修复：从1e-3降低到5e-5
    max_epochs: int = 50
    patience: int = 5
    # N1 Loss超参数
    gamma_n1_base: float = 2.5
    gamma_n1_increment: float = 0.5
    n1_weight_multiplier: float = 2.0
    # LoRA超参数（仅LoRA基线使用）
    lora_rank: int = 4

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class AdaptationResult:
    """单次适应实验的结果"""

    config: ExperimentConfig
    baseline_metrics: Dict[str, float]
    adapted_metrics: Dict[str, float]
    training_history: List[Dict]
    patient_results: List[Dict]
    adaptation_time_seconds: float
    total_trainable_params: int

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "AdaptationResult":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        # 嵌套的 config 需要从字典还原为 ExperimentConfig
        data["config"] = ExperimentConfig(**data["config"])
        return cls(**data)
