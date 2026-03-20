"""
OSA-Adapt: 严重程度感知的临床睡眠分期域适应框架

本模块包含OSA-Adapt框架的所有适应组件，包括：
- FiLM适配器（特征调制层）
- 严重程度条件化模块
- N1感知适应损失
- 渐进式适应协议
- 分层采样与交叉验证
- 实验管理与临床分析

核心类将在后续任务中实现并在此导出。
"""

# 核心数据模型
from .models import ExperimentConfig, AdaptationResult

# FiLM适配器
from .film_adapter import FiLMAdapter

# 严重程度条件化模块
from .severity_conditioner import SeverityConditioner

# 严重程度感知N1损失
from .severity_aware_loss import SeverityAwareN1Loss

# 分层采样器
from .stratified_sampler import SeverityStratifiedFewShotSampler

# 交叉验证器
from .cross_validator import CrossValidator

# 模型包装器
from .wrapped_models import (
    FiLMWrappedChambon,
    FiLMWrappedTinySleepNet,
    ONNXFeatureAdapter,
)

# 域内预训练器
from .indomain_pretrainer import (
    InDomainPretrainer,
    AugmentationConfig,
    CosineAnnealingWithWarmup,
    AugmentedDataset,
    apply_augmentation,
    create_class_balanced_sampler,
)

# U-Sleep 集成器
from .usleep_integrator import (
    USleepIntegrator,
    resample_signal,
    pkl_to_eeg_array,
)

# 渐进式适应协议
from .progressive_adapter import ProgressiveAdapter

# AHI估计器（Two-Pass Inference）
from .ahi_estimator import AHIEstimator

# 实验管理器
from .experiment_manager import ExperimentManager

# 基线适应方法
from .baselines import (
    BaseAdaptationMethod,
    NoAdaptation,
    FullFinetune,
    LastLayerFinetune,
    LoRAAdaptation,
    StandardFiLM,
    BNOnlyAdaptation,
    CORALAdaptation,
    MMDAdaptation,
    create_baseline,
    BASELINE_METHODS,
)

# 临床影响分析器
from .clinical_analyzer import ClinicalAnalyzer

# 消融实验运行器
from .ablation_runner import AblationRunner, AblationResult

# 论文级可视化
from .visualizer import AdaptationVisualizer

# 统计检验
from .statistical_tests import (
    bootstrap_ci,
    wilcoxon_test as adaptation_wilcoxon_test,
    bonferroni_correction as adaptation_bonferroni_correction,
    cohens_d,
    patient_level_wilcoxon,
    patient_level_bootstrap_comparison,
)

__all__ = [
    "ExperimentConfig",
    "AdaptationResult",
    "FiLMAdapter",
    "SeverityConditioner",
    "SeverityAwareN1Loss",
    "SeverityStratifiedFewShotSampler",
    "CrossValidator",
    "FiLMWrappedChambon",
    "FiLMWrappedTinySleepNet",
    "ONNXFeatureAdapter",
    "InDomainPretrainer",
    "AugmentationConfig",
    "CosineAnnealingWithWarmup",
    "AugmentedDataset",
    "apply_augmentation",
    "create_class_balanced_sampler",
    "ProgressiveAdapter",
    # U-Sleep 集成器
    "USleepIntegrator",
    "resample_signal",
    "pkl_to_eeg_array",
    # AHI估计器
    "AHIEstimator",
    # 实验管理器
    "ExperimentManager",
    # 基线适应方法
    "BaseAdaptationMethod",
    "NoAdaptation",
    "FullFinetune",
    "LastLayerFinetune",
    "LoRAAdaptation",
    "StandardFiLM",
    "BNOnlyAdaptation",
    "CORALAdaptation",
    "MMDAdaptation",
    "create_baseline",
    "BASELINE_METHODS",
    # 临床影响分析器
    "ClinicalAnalyzer",
    # 消融实验运行器
    "AblationRunner",
    "AblationResult",
    # 论文级可视化
    "AdaptationVisualizer",
    # 统计检验
    "bootstrap_ci",
    "adaptation_wilcoxon_test",
    "adaptation_bonferroni_correction",
    "cohens_d",
    "patient_level_wilcoxon",
    "patient_level_bootstrap_comparison",
]
