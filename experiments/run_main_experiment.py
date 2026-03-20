#!/usr/bin/env python
"""
OSA-Adapt 主实验脚本

串联所有组件：数据加载 → CV划分 → 分层采样 → 适应 → 评估 → 结果聚合
支持命令行参数指定模型、方法、数据预算，支持断点续跑。

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.1, 8.2, 8.3

用法示例:
    # 运行全部默认实验
    PYTHONPATH=. python experiments/run_main_experiment.py

    # 指定模型和方法
    PYTHONPATH=. python experiments/run_main_experiment.py \
        --models Chambon2018 --methods osa_adapt full_ft --budgets 5 10 20

    # 指定输出目录和种子数
    PYTHONPATH=. python experiments/run_main_experiment.py \
        --output-dir results \
        --n-seeds 3 --n-folds 5
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认常量
# ---------------------------------------------------------------------------
DEFAULT_BUDGETS = [5, 10, 20, 30, 50, 65, 100]
DEFAULT_MODELS = ["Chambon2018", "TinySleepNet"]
DEFAULT_METHODS = [
    "osa_adapt", "full_ft", "last_layer", "lora",
    "film_no_severity", "bn_only", "no_adapt", "coral", "mmd",
]
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_PKL_DIR = "data/preprocessed"
DEFAULT_SEVERITY_JSON = "data/patient_severity.json"
DEFAULT_N_FOLDS = 5
DEFAULT_N_SEEDS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_PRETRAINED_DIR = "weights/pretrained"

# OSA 严重程度映射
SEVERITY_MAP = {
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# 方法名映射（CLI参数 → 内部基线名）
METHOD_TO_BASELINE = {
    "full_ft": "full_finetune",
    "last_layer": "last_layer_finetune",
    "lora": "lora_adaptation",
    "film_no_severity": "standard_film",
    "bn_only": "bn_only_adaptation",
    "no_adapt": "no_adaptation",
    "coral": "coral_adaptation",
    "mmd": "mmd_adaptation",
}


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="OSA-Adapt 主实验：多数据预算下的域适应评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"待评估的模型列表 (默认: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help=f"待评估的适应方法列表 (默认: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS,
        help=f"数据预算列表（患者数） (默认: {DEFAULT_BUDGETS})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"结果输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pkl-dir", type=str, default=DEFAULT_PKL_DIR,
        help=f"PKL数据目录 (默认: {DEFAULT_PKL_DIR})",
    )
    parser.add_argument(
        "--severity-json", type=str, default=DEFAULT_SEVERITY_JSON,
        help=f"患者严重程度数据JSON (默认: {DEFAULT_SEVERITY_JSON})",
    )
    parser.add_argument(
        "--n-folds", type=int, default=DEFAULT_N_FOLDS,
        help=f"交叉验证折数 (默认: {DEFAULT_N_FOLDS})",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS,
        help=f"随机种子数量 (默认: {DEFAULT_N_SEEDS})",
    )
    parser.add_argument(
        "--pretrained-dir", type=str, default=DEFAULT_PRETRAINED_DIR,
        help=f"域内预训练检查点目录 (默认: {DEFAULT_PRETRAINED_DIR})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"默认 batch size (默认: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--skip-completed", action="store_true", default=True,
        help="跳过已完成的实验（断点续跑，默认开启）",
    )
    parser.add_argument(
        "--no-skip-completed", dest="skip_completed", action="store_false",
        help="不跳过已完成的实验（重新运行全部）",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str, output_dir: str) -> None:
    """配置日志：同时输出到控制台和文件。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "experiment.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_severity_data(json_path: str) -> Dict[str, Dict]:
    """加载患者严重程度数据。

    Args:
        json_path: patient_severity.json 路径

    Returns:
        patient_id -> {patient_id, osa_severity, ahi, ...} 字典
    """
    path = Path(json_path)
    if not path.exists():
        logger.warning("严重程度数据文件不存在: %s，将使用空字典", json_path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("已加载 %d 名患者的严重程度数据", len(data))
    return data


def get_patient_ids_from_pkl(pkl_dir: str) -> List[str]:
    """从PKL目录获取所有患者ID。"""
    pkl_path = Path(pkl_dir)
    if not pkl_path.exists():
        logger.warning("PKL目录不存在: %s", pkl_dir)
        return []

    patient_ids = set()
    for f in sorted(pkl_path.glob("*.pkl")):
        parts = f.stem.split("_", 1)
        if parts:
            pid = f"patient_{parts[0].zfill(3)}"
            patient_ids.add(pid)
    return sorted(patient_ids)


def get_severity_labels(
    patient_ids: List[str],
    severity_data: Dict[str, Dict],
) -> List[int]:
    """获取患者的OSA严重程度标签。"""
    labels = []
    for pid in patient_ids:
        if pid in severity_data:
            sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
            labels.append(SEVERITY_MAP.get(sev_str, 0))
        else:
            # 基于哈希的确定性模拟标签 (0-3)
            labels.append(hash(pid) % 4)
    return labels


# ---------------------------------------------------------------------------
# DataLoader 辅助函数
# ---------------------------------------------------------------------------
def create_dataloader(
    dataset: "PSGDataset",
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """从 PSGDataset 创建 DataLoader。

    DataLoader 产出 (signal, stage_label, patient_features_dict) 三元组。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def create_subset_dataloader(
    dataset: "PSGDataset",
    patient_ids: List[str],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """为指定患者子集创建 DataLoader。"""
    indices = []
    for pid in patient_ids:
        indices.extend(dataset.get_patient_epoch_indices(pid))
    if not indices:
        # 返回空 DataLoader
        return DataLoader(Subset(dataset, []), batch_size=batch_size)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# FiLM 包装辅助
# ---------------------------------------------------------------------------
def wrap_model_with_film(
    base_model: "nn.Module",
    model_name: str,
    condition_dim: int = 64,
) -> Tuple["nn.Module", "SeverityConditioner"]:
    """根据模型名称选择对应的 FiLM 包装器。

    Returns:
        (wrapped_model, conditioner)
    """
    from src.adaptation.severity_conditioner import SeverityConditioner
    from src.adaptation.wrapped_models import FiLMWrappedChambon, FiLMWrappedTinySleepNet

    conditioner = SeverityConditioner(condition_dim=condition_dim)

    if model_name == "Chambon2018":
        wrapped = FiLMWrappedChambon(base_model, conditioner)
    elif model_name == "TinySleepNet":
        wrapped = FiLMWrappedTinySleepNet(base_model, conditioner)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

    return wrapped, conditioner


# ---------------------------------------------------------------------------
# Per-patient 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_patient(
    model: "nn.Module",
    dataset: "PSGDataset",
    indices: List[int],
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """对单个患者的所有 epoch 进行预测。

    Returns:
        (y_true, y_pred) 两个 numpy 数组
    """
    model.eval()
    all_true = []
    all_pred = []

    # 分批处理
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        signals = []
        labels = []
        features_list = {k: [] for k in ["ahi", "severity", "age", "sex", "bmi"]}

        for idx in batch_indices:
            signal, label, pf = dataset[idx]
            signals.append(signal)
            labels.append(label)
            for k in features_list:
                features_list[k].append(pf[k])

        x = torch.stack(signals).to(device)
        patient_features = {
            k: torch.stack(v).to(device) for k, v in features_list.items()
        }
        # SeverityConditioner 需要 severity 和 sex 为 long 类型（embedding 查找）
        patient_features["severity"] = patient_features["severity"].long()
        patient_features["sex"] = patient_features["sex"].long()

        outputs = model(x, patient_features)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_true.extend(labels)
        all_pred.extend(preds.tolist())

    return np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# 修正 patient_features 类型的辅助函数
# ---------------------------------------------------------------------------
def fix_patient_features_types(patient_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """确保 severity 和 sex 为 long 类型（embedding 查找需要）。"""
    pf = dict(patient_features)
    pf["severity"] = pf["severity"].long()
    pf["sex"] = pf["sex"].long()
    return pf


# ---------------------------------------------------------------------------
# 类型修正的 DataLoader 包装器
# ---------------------------------------------------------------------------
class TypeFixingDataLoader:
    """包装 DataLoader，自动将 severity/sex 转为 long 类型。

    ProgressiveAdapter 和 baselines 的 adapt() 内部直接迭代 DataLoader，
    而 SeverityConditioner 的 embedding 层需要 long 类型输入。
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            if len(batch) == 3:
                x, targets, pf = batch
                pf = fix_patient_features_types(pf)
                yield x, targets, pf
            elif len(batch) == 2:
                x, pf = batch
                pf = fix_patient_features_types(pf)
                yield x, pf
            else:
                yield batch

    def __len__(self):
        return len(self.loader)


def run_single_experiment_cached(
    config: "ExperimentConfig",
    train_dataset: "PSGDataset",
    test_dataset: "PSGDataset",
    train_ids: List[str],
    test_ids: List[str],
    severity_data: Dict[str, Dict],
    severity_labels_map: Dict[str, int],
    batch_size: int = DEFAULT_BATCH_SIZE,
    pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
) -> Dict:
    """执行单次实验（使用预加载的 Dataset，避免重复 PKL 加载）。

    与 run_single_experiment 相同的训练流程，但接收已加载的 Dataset。
    """
    from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if len(train_dataset) == 0:
        logger.warning("训练集为空，跳过实验 %s", config.experiment_name)
        return {"status": "skipped", "reason": "empty_train_dataset"}

    if len(test_dataset) == 0:
        logger.warning("测试集为空，跳过实验 %s", config.experiment_name)
        return {"status": "skipped", "reason": "empty_test_dataset"}

    # 分层采样选择适应集
    train_severities = [severity_labels_map.get(pid, 0) for pid in train_ids]
    sampler = SeverityStratifiedFewShotSampler(seed=config.seed)
    adaptation_ids = sampler.sample(
        patient_ids=train_ids,
        severity_labels=train_severities,
        budget=config.data_budget,
    )

    logger.info(
        "  适应集: %d/%d 名患者 (预算=%d)",
        len(adaptation_ids), len(train_ids), config.data_budget,
    )

    # 将适应集分为训练和验证（80/20 split）
    n_adapt = len(adaptation_ids)
    n_val = max(1, n_adapt // 5)
    rng = np.random.RandomState(config.seed)
    shuffled_adapt = list(adaptation_ids)
    rng.shuffle(shuffled_adapt)
    val_ids = shuffled_adapt[:n_val]
    adapt_train_ids = shuffled_adapt[n_val:]

    if not adapt_train_ids:
        adapt_train_ids = list(adaptation_ids)
        val_ids = list(adaptation_ids)

    # OOM 重试循环
    current_batch_size = batch_size
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            fold_metrics = _execute_training(
                config=config,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_ids=train_ids,
                adapt_train_ids=adapt_train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                severity_data=severity_data,
                severity_labels_map=severity_labels_map,
                device=device,
                batch_size=current_batch_size,
                pretrained_dir=pretrained_dir,
            )
            break
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM (attempt %d/%d, batch_size=%d)，减半重试",
                attempt + 1, max_retries + 1, current_batch_size,
            )
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            if attempt == max_retries:
                logger.error(
                    "CUDA OOM 重试耗尽 (batch_size=%d)，实验失败: %s",
                    current_batch_size, config.experiment_name,
                )
                raise

    elapsed = time.time() - start_time
    fold_metrics["adaptation_time_seconds"] = elapsed
    fold_metrics["batch_size_used"] = current_batch_size
    fold_metrics["status"] = "completed"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_metrics


# ---------------------------------------------------------------------------
# 单次实验执行（原始版本，每次重新加载数据）
# ---------------------------------------------------------------------------
def run_single_experiment(
    config: "ExperimentConfig",
    train_ids: List[str],
    test_ids: List[str],
    severity_data: Dict[str, Dict],
    pkl_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
) -> Dict:
    """执行单次实验（一个 fold × seed × budget × method 组合）。

    完整流程:
    1. 创建 PSGDataset 和 DataLoader
    2. 分层采样选择适应集
    3. 构建模型 + 加载权重
    4. FiLM 包装
    5. 根据方法执行适应
    6. Per-patient 评估
    7. GPU 内存清理

    Args:
        config: 实验配置
        train_ids: 训练折患者ID
        test_ids: 测试折患者ID
        severity_data: 患者严重程度数据字典
        pkl_dir: PKL数据目录
        batch_size: 初始 batch size

    Returns:
        fold 级别指标字典
    """
    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader
    from src.adaptation.severity_conditioner import SeverityConditioner
    from src.adaptation.wrapped_models import FiLMWrappedChambon, FiLMWrappedTinySleepNet
    from src.adaptation.progressive_adapter import ProgressiveAdapter
    from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
    from src.adaptation.baselines import create_baseline
    from src.adaptation.evaluator import SleepStageEvaluator
    from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    demographics_gen = DemographicsGenerator()

    # ------------------------------------------------------------------
    # 1. 创建 Dataset
    # ------------------------------------------------------------------
    train_dataset = PSGDataset(
        patient_ids=train_ids,
        pkl_dir=pkl_dir,
        severity_data=severity_data,
        demographics_generator=demographics_gen,
    )
    test_dataset = PSGDataset(
        patient_ids=test_ids,
        pkl_dir=pkl_dir,
        severity_data=severity_data,
        demographics_generator=demographics_gen,
    )

    if len(train_dataset) == 0:
        logger.warning("训练集为空，跳过实验 %s", config.experiment_name)
        return {"status": "skipped", "reason": "empty_train_dataset"}

    if len(test_dataset) == 0:
        logger.warning("测试集为空，跳过实验 %s", config.experiment_name)
        return {"status": "skipped", "reason": "empty_test_dataset"}

    # ------------------------------------------------------------------
    # 2. 分层采样选择适应集
    # ------------------------------------------------------------------
    severity_labels_map = {}
    for pid in train_ids:
        if pid in severity_data:
            sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
            severity_labels_map[pid] = SEVERITY_MAP.get(sev_str, 0)
        else:
            severity_labels_map[pid] = hash(pid) % 4

    train_severities = [severity_labels_map.get(pid, 0) for pid in train_ids]
    sampler = SeverityStratifiedFewShotSampler(seed=config.seed)
    adaptation_ids = sampler.sample(
        patient_ids=train_ids,
        severity_labels=train_severities,
        budget=config.data_budget,
    )

    logger.info(
        "  适应集: %d/%d 名患者 (预算=%d)",
        len(adaptation_ids), len(train_ids), config.data_budget,
    )

    # 将适应集分为训练和验证（80/20 split）
    n_adapt = len(adaptation_ids)
    n_val = max(1, n_adapt // 5)
    rng = np.random.RandomState(config.seed)
    shuffled_adapt = list(adaptation_ids)
    rng.shuffle(shuffled_adapt)
    val_ids = shuffled_adapt[:n_val]
    adapt_train_ids = shuffled_adapt[n_val:]

    # 如果适应训练集为空（budget 很小），用全部适应集做训练和验证
    if not adapt_train_ids:
        adapt_train_ids = list(adaptation_ids)
        val_ids = list(adaptation_ids)

    # ------------------------------------------------------------------
    # OOM 重试循环 (Req 8.2)
    # ------------------------------------------------------------------
    current_batch_size = batch_size
    max_retries = 2  # 最多重试 2 次（共 3 次尝试）

    for attempt in range(max_retries + 1):
        try:
            fold_metrics = _execute_training(
                config=config,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_ids=train_ids,
                adapt_train_ids=adapt_train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                severity_data=severity_data,
                severity_labels_map=severity_labels_map,
                device=device,
                batch_size=current_batch_size,
                pretrained_dir=pretrained_dir,
            )
            break  # 成功则跳出重试循环
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM (attempt %d/%d, batch_size=%d)，减半重试",
                attempt + 1, max_retries + 1, current_batch_size,
            )
            # 清理 GPU 内存 (Req 8.1)
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            if attempt == max_retries:
                logger.error(
                    "CUDA OOM 重试耗尽 (batch_size=%d)，实验失败: %s",
                    current_batch_size, config.experiment_name,
                )
                raise

    elapsed = time.time() - start_time
    fold_metrics["adaptation_time_seconds"] = elapsed
    fold_metrics["batch_size_used"] = current_batch_size
    fold_metrics["status"] = "completed"

    # ------------------------------------------------------------------
    # 7. GPU 内存清理 (Req 8.1)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_metrics


def _execute_training(
    config: "ExperimentConfig",
    train_dataset: "PSGDataset",
    test_dataset: "PSGDataset",
    train_ids: List[str],
    adapt_train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    severity_data: Dict[str, Dict],
    severity_labels_map: Dict[str, int],
    device: torch.device,
    batch_size: int,
    pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
) -> Dict:
    """执行实际训练流程（可被 OOM 重试调用）。

    Returns:
        fold 级别指标字典
    """
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader
    from src.adaptation.progressive_adapter import ProgressiveAdapter
    from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
    from src.adaptation.baselines import create_baseline
    from src.adaptation.evaluator import SleepStageEvaluator

    # ------------------------------------------------------------------
    # 3. 构建模型 + 加载权重
    # ------------------------------------------------------------------
    base_model = build_model(config.model_name)
    weight_meta = WeightLoader.load_weights(
        base_model, config.model_name, fold=config.fold,
        pretrained_dir=pretrained_dir,
    )
    logger.info(
        "  权重加载: loaded=%s, source=%s",
        weight_meta["loaded"], weight_meta["source"],
    )

    # ------------------------------------------------------------------
    # 4. FiLM 包装
    # ------------------------------------------------------------------
    wrapped_model, conditioner = wrap_model_with_film(
        base_model, config.model_name, condition_dim=config.condition_dim,
    )
    wrapped_model = wrapped_model.to(device)

    # ------------------------------------------------------------------
    # 创建 DataLoader
    # ------------------------------------------------------------------
    # 未标注数据（全部训练折）用于 Phase 1 BN 适应
    unlabeled_loader = TypeFixingDataLoader(
        create_dataloader(train_dataset, batch_size=batch_size, shuffle=False)
    )
    # 适应训练集
    adapt_train_loader = TypeFixingDataLoader(
        create_subset_dataloader(train_dataset, adapt_train_ids, batch_size=batch_size)
    )
    # 验证集
    val_loader = TypeFixingDataLoader(
        create_subset_dataloader(train_dataset, val_ids, batch_size=batch_size, shuffle=False)
    )

    # ------------------------------------------------------------------
    # 5. 根据方法执行适应
    # ------------------------------------------------------------------
    adaptation_result = {}

    if config.adaptation_method == "osa_adapt":
        # OSA-Adapt: ProgressiveAdapter + SeverityAwareN1Loss (Req 4.1, 4.3)
        loss_fn = SeverityAwareN1Loss(
            gamma_n1_base=config.gamma_n1_base,
            gamma_n1_increment=config.gamma_n1_increment,
            n1_weight_multiplier=config.n1_weight_multiplier,
        )
        adapter = ProgressiveAdapter(
            model=wrapped_model,
            conditioner=conditioner,
            loss_fn=loss_fn,
            lr=config.lr,
            patience=config.patience,
            max_epochs=config.max_epochs,
            bn_momentum=0.01,  # 降低 BN momentum,使统计量更新更温和
        )
        # Phase 1: BN 适应（无标签）
        phase1_result = adapter.phase1_bn_adapt(unlabeled_loader)
        logger.info("  Phase 1 完成: %d samples", phase1_result["num_samples"])

        # Phase 2: FiLM 微调（有标签）
        phase2_result = adapter.phase2_film_finetune(adapt_train_loader, val_loader)
        logger.info(
            "  Phase 2 完成: %d epochs, best_val_acc=%.4f",
            phase2_result["total_epochs"],
            phase2_result["best_val_accuracy"],
        )
        adaptation_result = {
            "phase1": phase1_result,
            "phase2": phase2_result,
        }
        model_to_eval = wrapped_model

    elif config.adaptation_method == "no_adapt":
        # 不做任何适应，直接评估
        logger.info("  no_adapt: 跳过适应")
        model_to_eval = wrapped_model

    else:
        # 基线方法：使用 baselines.py 的 adapt() 接口 (Req 4.2)
        # 基线方法需要原始 base_model（而非 FiLM 包装后的模型），
        # 因为它们有自己的包装/修改逻辑（如 StandardFiLM 会自行添加 FiLM 层，
        # LoRA 会替换线性层）。传入已包装的模型会导致双重包装或 device 问题。
        baseline_name = METHOD_TO_BASELINE.get(config.adaptation_method)
        if baseline_name is None:
            raise ValueError(
                f"未知的适应方法: {config.adaptation_method}"
            )

        # 为基线方法创建独立的 base_model（避免与 wrapped_model 共享状态）
        baseline_base_model = build_model(config.model_name)
        WeightLoader.load_weights(
            baseline_base_model, config.model_name, fold=config.fold,
            pretrained_dir=pretrained_dir,
        )
        baseline_base_model = baseline_base_model.to(device)

        baseline = create_baseline(baseline_name)
        adaptation_result = baseline.adapt(
            baseline_base_model, adapt_train_loader, val_loader,
        )
        logger.info(
            "  基线 %s 完成: %d epochs",
            baseline_name,
            adaptation_result.get("total_epochs", 0),
        )
        # 某些基线方法（如 StandardFiLM）会创建新的包装模型用于评估
        model_to_eval = adaptation_result.pop("adapted_model", baseline_base_model)

    # ------------------------------------------------------------------
    # 6. Per-patient 评估
    # ------------------------------------------------------------------
    evaluator = SleepStageEvaluator()
    patient_results = []

    # 获取测试集中实际加载成功的患者
    loaded_test_ids = [
        pid for pid in test_ids
        if test_dataset.get_patient_epoch_indices(pid)
    ]

    for pid in loaded_test_ids:
        indices = test_dataset.get_patient_epoch_indices(pid)
        if not indices:
            continue

        severity = severity_labels_map.get(pid, 0)
        # 也从 severity_data 获取
        if pid in severity_data:
            sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
            severity = SEVERITY_MAP.get(sev_str, 0)

        y_true, y_pred = predict_patient(
            model_to_eval, test_dataset, indices, device, batch_size=batch_size,
        )
        metrics = evaluator.evaluate_patient(y_true, y_pred, severity)
        patient_results.append(metrics)

    fold_metrics = evaluator.evaluate_fold(patient_results)

    # 附加元信息
    fold_metrics["n_test_patients"] = len(loaded_test_ids)
    fold_metrics["n_patient_results"] = len(patient_results)
    fold_metrics["weight_source"] = weight_meta["source"]

    # 清理模型 (Req 8.1)
    del model_to_eval, wrapped_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_metrics


# ---------------------------------------------------------------------------
# 环境元数据收集
# ---------------------------------------------------------------------------
def scan_completed_runs(output_dir: str) -> Dict[str, Dict]:
    """扫描输出目录中已完成的独立结果 JSON 文件。

    扫描 output_dir 下所有 *_result.json 文件，解析其中的
    model/method/budget/fold/seed 信息，返回已完成运行的字典。

    Args:
        output_dir: 结果输出目录路径

    Returns:
        {run_key: result_dict} 字典，run_key 格式为
        "{method}_{model}_budget{budget}_fold{fold}_seed{seed}"
    """
    completed = {}
    out_path = Path(output_dir)
    if not out_path.exists():
        return completed

    for result_file in sorted(out_path.glob("*_result.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            # 从结果 JSON 中提取组合标识
            model = data.get("model", "")
            method = data.get("method", "")
            budget = data.get("budget", "")
            fold = data.get("fold", "")
            seed = data.get("seed", "")
            if model and method and budget != "" and fold != "" and seed != "":
                run_key = f"{method}_{model}_budget{budget}_fold{fold}_seed{seed}"
                completed[run_key] = data
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("跳过损坏的结果文件 %s: %s", result_file, e)
            continue

    return completed


def get_incomplete_combinations(
    all_configs: List["ExperimentConfig"],
    completed_runs: Dict[str, Dict],
) -> List["ExperimentConfig"]:
    """从完整实验矩阵中筛选出未完成的组合。

    通过比较 experiment_name 与已完成运行的 key 来判断。

    Args:
        all_configs: 完整实验矩阵的所有配置
        completed_runs: scan_completed_runs() 返回的已完成运行字典

    Returns:
        未完成的实验配置列表
    """
    return [c for c in all_configs if c.experiment_name not in completed_runs]


def generate_completion_summary(
    total_runs: int,
    completed_count: int,
    failed_count: int,
    skipped_count: int,
    elapsed_seconds: float,
    output_dir: str,
) -> Dict:
    """生成实验完成摘要（需求 3.6）。

    Args:
        total_runs: 总运行数（完整实验矩阵大小）
        completed_count: 本次成功完成的运行数
        failed_count: 本次失败的运行数
        skipped_count: 跳过的已完成运行数
        elapsed_seconds: 本次执行总耗时（秒）
        output_dir: 结果输出目录

    Returns:
        完成摘要字典
    """
    from datetime import datetime

    summary = {
        "total_runs": total_runs,
        "completed": completed_count + skipped_count,
        "newly_completed": completed_count,
        "failed": failed_count,
        "skipped_existing": skipped_count,
        "remaining": total_runs - completed_count - skipped_count - failed_count,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_minutes": round(elapsed_seconds / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # 保存摘要 JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "completion_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("完成摘要已保存: %s", summary_path)

    # 日志输出摘要
    logger.info("=" * 60)
    logger.info("实验完成摘要")
    logger.info("=" * 60)
    logger.info("  总运行数:     %d", summary["total_runs"])
    logger.info("  已完成:       %d (本次 %d, 之前 %d)",
                summary["completed"], summary["newly_completed"],
                summary["skipped_existing"])
    logger.info("  失败:         %d", summary["failed"])
    logger.info("  剩余:         %d", summary["remaining"])
    logger.info("  总耗时:       %.1f 秒 (%.1f 分钟)",
                summary["elapsed_seconds"], summary["elapsed_minutes"])
    logger.info("=" * 60)

    return summary


def collect_metadata() -> Dict:
    """收集运行环境元数据（PyTorch 版本、CUDA 版本、GPU 型号等）。"""
    from datetime import datetime

    metadata = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.version.cuda else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }
    return metadata


def _save_independent_result_json(
    config: "ExperimentConfig",
    result: Dict,
    output_dir: str,
) -> None:
    """将单次运行结果保存为独立 JSON 文件（设计文档指定格式）。

    文件保存到 output_dir/ 下，文件名为 {experiment_name}_result.json。
    格式包含 model、method、budget、fold、seed、metrics、metadata。

    Requirements: 3.4, 8.1, 8.2
    """
    if result.get("status") != "completed":
        return

    env_metadata = collect_metadata()
    env_metadata["training_time_seconds"] = result.get("adaptation_time_seconds", 0.0)

    # 构建 metrics 字典
    metrics = {}
    for key in ["acc", "kappa", "macro_f1", "n1_f1"]:
        if key in result:
            metrics[key.replace("acc", "accuracy") if key == "acc" else key] = result[key]

    # per_stage_f1
    if "per_stage_f1" in result:
        metrics["per_stage_f1"] = result["per_stage_f1"]

    # confusion_matrix
    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        # 确保可序列化（numpy array → list）
        if hasattr(cm, "tolist"):
            cm = cm.tolist()
        metrics["confusion_matrix"] = cm

    # severity_breakdown
    if "severity_breakdown" in result:
        metrics["severity_breakdown"] = result["severity_breakdown"]

    # 补充 severity 相关指标
    for key in ["severe_acc", "severe_n1_f1"]:
        if key in result:
            metrics[key] = result[key]

    independent_result = {
        "model": config.model_name,
        "method": config.adaptation_method,
        "budget": config.data_budget,
        "fold": config.fold,
        "seed": config.seed,
        "metrics": metrics,
        "metadata": env_metadata,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{config.experiment_name}_result.json"
    result_file.write_text(
        json.dumps(independent_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("独立结果 JSON 已保存: %s", result_file)


# ---------------------------------------------------------------------------
# 结果聚合 (Requirements: 6.1, 6.2, 6.3, 6.4)
# ---------------------------------------------------------------------------
# 评估器输出的指标键（与 ResultsAggregator.BASE_METRICS 一致）
_METRIC_KEYS = ["acc", "kappa", "macro_f1", "n1_f1", "severe_acc", "severe_n1_f1"]


def _aggregate_and_save_results(
    results_df: "pd.DataFrame",
    args: argparse.Namespace,
) -> None:
    """从实验结果 DataFrame 聚合并保存 JSON 文件。

    生成两个文件:
    - results/main_results_summary.json
    - results/per_budget_results.json

    Args:
        results_df: manager.collect_results() 返回的 DataFrame
        args: 命令行参数（用于获取模型/方法列表）
    """
    from src.adaptation.results_aggregator import ResultsAggregator

    if results_df.empty:
        logger.warning("结果 DataFrame 为空，跳过聚合")
        return

    # 检查必需列是否存在
    required_cols = {"model_name", "adaptation_method", "status"}
    missing = required_cols - set(results_df.columns)
    if missing:
        logger.warning("结果 DataFrame 缺少必需列: %s，跳过聚合", missing)
        return

    # 仅保留成功完成的实验
    completed_df = results_df[results_df["status"] == "completed"].copy()
    if completed_df.empty:
        logger.warning("无已完成的实验结果，跳过聚合")
        return

    # 检查指标列是否存在（至少需要 acc）
    available_metrics = [k for k in _METRIC_KEYS if k in completed_df.columns]
    if not available_metrics:
        logger.warning("结果中无可用指标列，跳过聚合")
        return

    aggregator = ResultsAggregator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 按 model → method 分组，生成 main_results_summary.json
    # ------------------------------------------------------------------
    all_models_aggregated = {}

    for model_name in completed_df["model_name"].unique():
        model_df = completed_df[completed_df["model_name"] == model_name]

        # 构建 results_by_method: {method: [list of metric dicts]}
        results_by_method = {}
        for method in model_df["adaptation_method"].unique():
            method_df = model_df[model_df["adaptation_method"] == method]
            run_results = []
            for _, row in method_df.iterrows():
                metrics = {}
                for k in _METRIC_KEYS:
                    metrics[k] = float(row[k]) if k in row.index and not _is_nan(row.get(k)) else 0.0
                run_results.append(metrics)
            results_by_method[method] = run_results

        # 使用 aggregate_all_methods 自动处理 no_adapt 基线和 delta 计算
        model_aggregated = aggregator.aggregate_all_methods(results_by_method)
        all_models_aggregated[model_name] = model_aggregated

    # 保存 main_results_summary.json
    main_output = str(output_dir / "main_results_summary.json")
    aggregator.save_json(all_models_aggregated, main_output)
    logger.info("主结果汇总已保存: %s", main_output)

    # ------------------------------------------------------------------
    # 2. 按 model → method → budget 分组，生成 per_budget_results.json
    # ------------------------------------------------------------------
    if "data_budget" not in completed_df.columns:
        logger.warning("结果中无 data_budget 列，跳过 per-budget 聚合")
        return

    per_budget_all = {}

    for model_name in completed_df["model_name"].unique():
        model_df = completed_df[completed_df["model_name"] == model_name]
        per_budget_model = {}

        # 获取该模型的 no_adapt 基线（跨所有 budget 聚合）
        no_adapt_df = model_df[model_df["adaptation_method"] == "no_adapt"]
        no_adapt_baseline = {k: 0.0 for k in _METRIC_KEYS}
        if not no_adapt_df.empty:
            for k in _METRIC_KEYS:
                if k in no_adapt_df.columns:
                    vals = no_adapt_df[k].dropna()
                    if len(vals) > 0:
                        no_adapt_baseline[k] = float(vals.mean())

        for method in model_df["adaptation_method"].unique():
            method_df = model_df[model_df["adaptation_method"] == method]

            # 按 budget 分组
            results_by_budget = {}
            for budget in sorted(method_df["data_budget"].unique()):
                budget_df = method_df[method_df["data_budget"] == budget]
                run_results = []
                for _, row in budget_df.iterrows():
                    metrics = {}
                    for k in _METRIC_KEYS:
                        metrics[k] = float(row[k]) if k in row.index and not _is_nan(row.get(k)) else 0.0
                    run_results.append(metrics)
                results_by_budget[int(budget)] = run_results

            per_budget_model[method] = aggregator.aggregate_per_budget(
                results_by_budget, no_adapt_baseline,
            )

        per_budget_all[model_name] = per_budget_model

    # 保存 per_budget_results.json（budget 键转为字符串以兼容 JSON）
    per_budget_serializable = {}
    for model_name, methods in per_budget_all.items():
        per_budget_serializable[model_name] = {}
        for method, budgets in methods.items():
            per_budget_serializable[model_name][method] = {
                str(b): metrics for b, metrics in budgets.items()
            }

    budget_output = str(output_dir / "per_budget_results.json")
    aggregator.save_json(per_budget_serializable, budget_output)
    logger.info("Per-budget 结果已保存: %s", budget_output)


def _is_nan(value) -> bool:
    """检查值是否为 NaN（兼容 None 和非数值类型）。"""
    if value is None:
        return True
    try:
        return np.isnan(float(value))
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# 主实验流程
# ---------------------------------------------------------------------------
def run_main_experiment(args: argparse.Namespace) -> None:
    """主实验入口：遍历所有配置组合并执行实验。

    流程:
    1. 加载数据和严重程度信息
    2. 创建交叉验证划分
    3. 生成所有实验配置（models × methods × budgets × folds × seeds）
    4. 对每个配置执行实验（支持断点续跑）
    5. 收集结果并生成汇总统计
    """
    from src.adaptation.cross_validator import CrossValidator
    from src.adaptation.experiment_manager import ExperimentManager
    from src.adaptation.models import ExperimentConfig

    logger.info("=" * 60)
    logger.info("OSA-Adapt 主实验启动")
    logger.info("=" * 60)
    logger.info("模型: %s", args.models)
    logger.info("方法: %s", args.methods)
    logger.info("数据预算: %s", args.budgets)
    logger.info("折数: %d, 种子数: %d", args.n_folds, args.n_seeds)
    logger.info("输出目录: %s", args.output_dir)
    logger.info("预训练检查点目录: %s", args.pretrained_dir)
    logger.info("Batch size: %d", args.batch_size)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("GPU: 不可用，使用 CPU")

    # 1. 加载数据
    severity_data = load_severity_data(args.severity_json)

    # 检查 PKL 目录
    pkl_path = Path(args.pkl_dir)
    if not pkl_path.exists():
        logger.warning(
            "PKL 数据目录不存在: %s，无法执行真实训练", args.pkl_dir,
        )
        return

    patient_ids = get_patient_ids_from_pkl(args.pkl_dir)

    if not patient_ids:
        logger.warning(
            "未找到患者数据（PKL目录: %s），无法执行实验",
            args.pkl_dir,
        )
        return

    severity_labels = get_severity_labels(patient_ids, severity_data)
    severity_map = dict(zip(patient_ids, severity_labels))

    logger.info("患者总数: %d", len(patient_ids))
    severity_counts = {}
    for s in severity_labels:
        severity_counts[s] = severity_counts.get(s, 0) + 1
    logger.info("严重程度分布: %s", severity_counts)

    # 2. 交叉验证划分
    cv = CrossValidator(n_folds=args.n_folds, seed=42)
    folds = cv.split(patient_ids, severity_labels)
    logger.info("交叉验证: %d 折划分完成", len(folds))

    # 3. 实验管理器
    manager = ExperimentManager(output_dir=Path(args.output_dir).parent.as_posix())
    manager.results_dir = Path(args.output_dir)
    manager.results_dir.mkdir(parents=True, exist_ok=True)

    # 4. 生成实验配置
    configs = manager.generate_configs(
        models=args.models,
        methods=args.methods,
        budgets=args.budgets,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
    )
    total_configs = len(configs)
    logger.info("总实验配置数: %d", total_configs)

    # 5. 断点续跑：扫描已有结果文件（需求 3.5）
    #    同时检查 ExperimentManager 的结果和独立结果 JSON
    skipped_count = 0
    if args.skip_completed:
        # 扫描独立结果 JSON 文件
        completed_runs = scan_completed_runs(args.output_dir)
        # 合并 ExperimentManager 的完成状态
        pending = []
        for c in configs:
            if manager.is_completed(c) or c.experiment_name in completed_runs:
                skipped_count += 1
            else:
                pending.append(c)
        if skipped_count > 0:
            logger.info(
                "断点续跑: 跳过 %d 个已完成实验 (%d 个来自结果目录扫描)",
                skipped_count, len(completed_runs),
            )
    else:
        pending = configs

    logger.info("待执行实验: %d", len(pending))

    # 6. 按 fold 分组执行实验（同一 fold 共享 Dataset，避免重复加载 PKL）
    experiment_start_time = time.time()

    from collections import defaultdict
    fold_groups = defaultdict(list)
    for config in pending:
        fold_groups[config.fold % len(folds)].append(config)

    completed_count = 0
    failed_count = 0
    exp_idx = 0

    for fold_idx in sorted(fold_groups.keys()):
        fold_configs = fold_groups[fold_idx]
        train_ids, test_ids = folds[fold_idx]

        logger.info(
            "--- Fold %d: %d 个实验，加载数据集 ---",
            fold_idx, len(fold_configs),
        )

        # 为该 fold 预加载 Dataset（一次性加载，所有实验共享）
        from src.adaptation.psg_dataset import PSGDataset
        from src.adaptation.demographics_generator import DemographicsGenerator
        demographics_gen = DemographicsGenerator()

        fold_start = time.time()
        train_dataset = PSGDataset(
            patient_ids=train_ids,
            pkl_dir=args.pkl_dir,
            severity_data=severity_data,
            demographics_generator=demographics_gen,
        )
        test_dataset = PSGDataset(
            patient_ids=test_ids,
            pkl_dir=args.pkl_dir,
            severity_data=severity_data,
            demographics_generator=demographics_gen,
        )
        load_time = time.time() - fold_start
        logger.info(
            "  Fold %d 数据加载完成: 训练 %d epochs, 测试 %d epochs (%.1f秒)",
            fold_idx, len(train_dataset), len(test_dataset), load_time,
        )

        # 预计算 severity labels map
        severity_labels_map = {}
        for pid in train_ids:
            if pid in severity_data:
                sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
                severity_labels_map[pid] = SEVERITY_MAP.get(sev_str, 0)
            else:
                severity_labels_map[pid] = hash(pid) % 4
        for pid in test_ids:
            if pid in severity_data:
                sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
                severity_labels_map[pid] = SEVERITY_MAP.get(sev_str, 0)
            else:
                severity_labels_map[pid] = hash(pid) % 4

        for config in fold_configs:
            exp_idx += 1
            logger.info(
                "[%d/%d] %s", exp_idx, len(pending), config.experiment_name,
            )

            try:
                result = run_single_experiment_cached(
                    config=config,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    severity_data=severity_data,
                    severity_labels_map=severity_labels_map,
                    batch_size=args.batch_size,
                    pretrained_dir=args.pretrained_dir,
                )
                result["status"] = "completed"
                completed_count += 1
            except Exception as e:
                logger.error(
                    "实验失败 %s: %s\n%s",
                    config.experiment_name, e, traceback.format_exc(),
                )
                result = {
                    "status": "failed",
                    "error": str(e),
                    "experiment_name": config.experiment_name,
                }
                failed_count += 1

            # 保存到 ExperimentManager（兼容现有聚合逻辑）
            manager.save_result(config, result)

            # 保存独立结果 JSON
            _save_independent_result_json(config, result, args.output_dir)

        # Fold 完成后清理 Dataset 内存
        del train_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7. 收集结果并生成汇总
    total_elapsed = time.time() - experiment_start_time

    # 生成完成摘要（需求 3.6）
    generate_completion_summary(
        total_runs=total_configs,
        completed_count=completed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        elapsed_seconds=total_elapsed,
        output_dir=args.output_dir,
    )

    results_df = manager.collect_results()

    if not results_df.empty:
        summary_path = Path(args.output_dir) / "experiment_summary.csv"
        results_df.to_csv(str(summary_path), index=False)
        logger.info("结果汇总已保存: %s (共 %d 条)", summary_path, len(results_df))

        # 按方法和预算汇总
        if "adaptation_method" in results_df.columns and "data_budget" in results_df.columns:
            summary = results_df.groupby(
                ["adaptation_method", "data_budget"]
            ).size().reset_index(name="n_runs")
            logger.info("实验汇总:\n%s", summary.to_string(index=False))
    else:
        logger.warning("未收集到任何实验结果")

    # ------------------------------------------------------------------
    # 7.5 结果聚合：生成 main_results_summary.json 和 per_budget_results.json
    #     (Requirements: 6.1, 6.2, 6.3, 6.4)
    # ------------------------------------------------------------------
    _aggregate_and_save_results(results_df, args)

    logger.info("=" * 60)
    logger.info("OSA-Adapt 主实验完成")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """脚本入口。"""
    args = parse_args(argv)
    setup_logging(args.log_level, args.output_dir)
    run_main_experiment(args)


if __name__ == "__main__":
    main()
