#!/usr/bin/env python
"""
实验1：数据效率曲线
目的：展示OSA-Adapt在不同数据量下相对于Full Fine-tuning的优势
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptation.model_builder import build_model
from src.adaptation.weight_loader import WeightLoader
from src.adaptation.psg_dataset import PSGDataset
from src.adaptation.demographics_generator import DemographicsGenerator
from src.adaptation.wrapped_models import FiLMWrappedChambon
from src.adaptation.severity_conditioner import SeverityConditioner
from src.adaptation.progressive_adapter import ProgressiveAdapter
from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
from src.adaptation.cross_validator import CrossValidator
from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler
from src.adaptation.baselines import FullFineTuneBaseline
from torch.utils.data import DataLoader, Subset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/data_efficiency/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_severity_data(json_path="data/patient_severity.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_patient_ids():
    pkl_dir = Path("data/preprocessed")
    patient_ids = set()
    for f in sorted(pkl_dir.glob("*.pkl")):
        parts = f.stem.split("_", 1)
        if parts:
            pid = f"patient_{parts[0].zfill(3)}"
            patient_ids.add(pid)
    return sorted(patient_ids)

def create_subset_dataloader(dataset, patient_ids, batch_size=128, shuffle=False):
    indices = []
    for pid in patient_ids:
        indices.extend(dataset.get_patient_epoch_indices(pid))
    if not indices:
        return None
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        if len(batch) != 3:
            continue
            
        x, targets, pf = batch
        x = x.to(device)
        targets = targets.to(device)
        pf = {k: v.to(device) for k, v in pf.items()}
        
        outputs = model(x, pf)
        preds = outputs.argmax(dim=1)
        
        valid_mask = targets >= 0
        if valid_mask.any():
            all_preds.append(preds[valid_mask].cpu().numpy())
            all_targets.append(targets[valid_mask].cpu().numpy())
    
    if not all_preds:
        return {"accuracy": 0.0, "kappa": 0.0}
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    accuracy = (all_preds == all_targets).mean()
    
    # 计算Cohen's kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    return {"accuracy": accuracy, "kappa": kappa}

def run_single_experiment(
    model_name, budget, fold, seed, method,
    train_dataset, test_dataset, train_ids, test_ids,
    train_severities, device
):
    """运行单个实验配置"""
    logger.info(f"开始: {model_name}, budget={budget}, fold={fold}, seed={seed}, method={method}")
    
    # 采样适应集
    sampler = SeverityStratifiedFewShotSampler(seed=seed)
    adaptation_ids = sampler.sample(
        patient_ids=train_ids,
        severity_labels=train_severities,
        budget=budget,
    )
    
    n_val = max(2, len(adaptation_ids) // 5)
    rng = np.random.RandomState(seed)
    shuffled_adapt = list(adaptation_ids)
    rng.shuffle(shuffled_adapt)
    val_ids = shuffled_adapt[:n_val]
    adapt_train_ids = shuffled_adapt[n_val:]
    
    # 加载模型
    base_model = build_model(model_name)
    WeightLoader.load_weights(
        base_model, model_name, fold=fold,
        pretrained_dir="weights/rescue_pretrained"
    )
    base_model = base_model.to(device)
    
    # 创建测试集加载器
    test_loader = create_subset_dataloader(test_dataset, test_ids, batch_size=128)
    
    if method == "no_adapt":
        # 无适应基线
        conditioner = SeverityConditioner(condition_dim=64)
        wrapped_model = FiLMWrappedChambon(base_model, conditioner)
        wrapped_model = wrapped_model.to(device)
        
        result = evaluate_model(wrapped_model, test_loader, device)
        result.update({
            "model": model_name,
            "budget": budget,
            "fold": fold,
            "seed": seed,
            "method": method,
            "training_epochs": 0,
        })
        
    elif method == "osa_adapt":
        # OSA-Adapt
        conditioner = SeverityConditioner(condition_dim=64)
        wrapped_model = FiLMWrappedChambon(base_model, conditioner)
        wrapped_model = wrapped_model.to(device)
        
        # 创建数据加载器
        adapt_train_loader = create_subset_dataloader(
            train_dataset, adapt_train_ids, batch_size=32, shuffle=True
        )
        val_loader = create_subset_dataloader(train_dataset, val_ids, batch_size=32)
        
        # 训练
        loss_fn = SeverityAwareN1Loss()
        adapter = ProgressiveAdapter(
            model=wrapped_model,
            conditioner=conditioner,
            loss_fn=loss_fn,
            lr=5e-5,
            patience=5,
            max_epochs=30,
        )
        
        phase2_result = adapter.phase2_film_finetune(adapt_train_loader, val_loader)
        
        # 评估
        result = evaluate_model(wrapped_model, test_loader, device)
        result.update({
            "model": model_name,
            "budget": budget,
            "fold": fold,
            "seed": seed,
            "method": method,
            "training_epochs": phase2_result["total_epochs"],
            "best_val_accuracy": phase2_result["best_val_accuracy"],
        })
        
    elif method == "full_ft":
        # Full Fine-tuning
        conditioner = SeverityConditioner(condition_dim=64)
        wrapped_model = FiLMWrappedChambon(base_model, conditioner)
        wrapped_model = wrapped_model.to(device)
        
        # 创建数据加载器
        adapt_train_loader = create_subset_dataloader(
            train_dataset, adapt_train_ids, batch_size=32, shuffle=True
        )
        val_loader = create_subset_dataloader(train_dataset, val_ids, batch_size=32)
        
        # 训练（使用Full Fine-tune baseline）
        loss_fn = SeverityAwareN1Loss()
        baseline = FullFineTuneBaseline(
            model=wrapped_model,
            loss_fn=loss_fn,
            lr=1e-3,  # 标准fine-tuning学习率
            patience=5,
            max_epochs=30,
        )
        
        ft_result = baseline.train(adapt_train_loader, val_loader)
        
        # 评估
        result = evaluate_model(wrapped_model, test_loader, device)
        result.update({
            "model": model_name,
            "budget": budget,
            "fold": fold,
            "seed": seed,
            "method": method,
            "training_epochs": ft_result["total_epochs"],
            "best_val_accuracy": ft_result["best_val_accuracy"],
        })
    
    logger.info(f"完成: accuracy={result['accuracy']:.4f}, kappa={result['kappa']:.4f}")
    
    # 清理
    del base_model, wrapped_model
    if method != "no_adapt":
        if method == "osa_adapt":
            del adapter
        else:
            del baseline
    torch.cuda.empty_cache()
    
    return result

def main():
    logger.info("="*60)
    logger.info("实验1：数据效率曲线")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path("results/data_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    severity_data = load_severity_data()
    all_patient_ids = get_patient_ids()
    
    severity_map = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
    severity_labels = []
    for pid in all_patient_ids:
        if pid in severity_data:
            sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
            severity_labels.append(severity_map.get(sev_str, 0))
        else:
            severity_labels.append(0)
    
    # 交叉验证划分
    cv = CrossValidator(n_folds=5, seed=42)
    splits = cv.split(all_patient_ids, severity_labels)
    
    # 实验配置
    model_name = "Chambon2018"
    budgets = [5, 10, 20, 50, 100]
    methods = ["no_adapt", "osa_adapt", "full_ft"]
    seeds = [42, 43, 44, 45, 46]
    
    # 运行所有实验
    all_results = []
    total_experiments = len(budgets) * len(methods) * len(splits) * len(seeds)
    current = 0
    
    for budget in budgets:
        for fold_idx, (train_ids, test_ids) in enumerate(splits):
            # 创建数据集
            demographics_gen = DemographicsGenerator()
            train_dataset = PSGDataset(
                patient_ids=train_ids,
                pkl_dir="data/preprocessed",
                severity_data=severity_data,
                demographics_generator=demographics_gen,
            )
            test_dataset = PSGDataset(
                patient_ids=test_ids,
                pkl_dir="data/preprocessed",
                severity_data=severity_data,
                demographics_generator=demographics_gen,
            )
            
            train_severities = [
                severity_labels[all_patient_ids.index(pid)] for pid in train_ids
            ]
            
            for seed in seeds:
                for method in methods:
                    current += 1
                    logger.info(f"\n进度: {current}/{total_experiments}")
                    
                    try:
                        result = run_single_experiment(
                            model_name=model_name,
                            budget=budget,
                            fold=fold_idx,
                            seed=seed,
                            method=method,
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            train_ids=train_ids,
                            test_ids=test_ids,
                            train_severities=train_severities,
                            device=device,
                        )
                        all_results.append(result)
                        
                        # 保存中间结果
                        with open(output_dir / "intermediate_results.json", "w") as f:
                            json.dump(all_results, f, indent=2)
                            
                    except Exception as e:
                        logger.error(f"实验失败: {e}")
                        continue
    
    # 保存最终结果
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 生成汇总表格
    df = pd.DataFrame(all_results)
    
    # 按budget和method分组统计
    summary = df.groupby(["budget", "method"])["accuracy"].agg(["mean", "std", "count"])
    summary.to_csv(output_dir / "summary_table1.csv")
    
    logger.info("\n" + "="*60)
    logger.info("实验完成！")
    logger.info("="*60)
    logger.info(f"\n汇总结果:\n{summary}")
    logger.info(f"\n结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
