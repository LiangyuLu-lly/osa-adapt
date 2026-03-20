#!/usr/bin/env python
"""
现实数据量实验：测试budget=50,100,150
验证OSA-Adapt在中等数据量下的优势
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

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
from src.adaptation.simple_baselines import FullFineTuneBaseline
from torch.utils.data import DataLoader, Subset

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
        return 0.0
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return (all_preds == all_targets).mean()

def run_experiment(budget, fold, seed, device, all_patient_ids, severity_labels, 
                   severity_data, demographics_gen):
    """运行单个实验配置"""
    
    # 交叉验证划分
    cv = CrossValidator(n_folds=5, seed=42)
    splits = cv.split(all_patient_ids, severity_labels)
    train_ids, test_ids = splits[fold]
    
    # 创建数据集
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
    
    test_loader = create_subset_dataloader(test_dataset, test_ids, batch_size=128)
    
    # 采样适应集
    train_severities = [severity_labels[all_patient_ids.index(pid)] for pid in train_ids]
    sampler = SeverityStratifiedFewShotSampler(seed=seed)
    adaptation_ids = sampler.sample(
        patient_ids=train_ids,
        severity_labels=train_severities,
        budget=budget,
    )
    
    # 划分训练/验证集 (80/20)
    n_val = max(5, len(adaptation_ids) // 5)  # 至少5个患者用于验证
    rng = np.random.RandomState(seed)
    shuffled_adapt = list(adaptation_ids)
    rng.shuffle(shuffled_adapt)
    val_ids = shuffled_adapt[:n_val]
    adapt_train_ids = shuffled_adapt[n_val:]
    
    results = {}
    
    # 方法1: No Adaptation (baseline)
    base_model = build_model("Chambon2018")
    WeightLoader.load_weights(base_model, "Chambon2018", fold=fold, 
                             pretrained_dir="weights/rescue_pretrained")
    base_model = base_model.to(device)
    
    conditioner = SeverityConditioner(condition_dim=64)
    wrapped_model = FiLMWrappedChambon(base_model, conditioner)
    wrapped_model = wrapped_model.to(device)
    
    no_adapt_acc = evaluate_model(wrapped_model, test_loader, device)
    results["no_adapt"] = no_adapt_acc
    
    del base_model, conditioner, wrapped_model
    torch.cuda.empty_cache()
    
    # 方法2: OSA-Adapt
    base_model = build_model("Chambon2018")
    WeightLoader.load_weights(base_model, "Chambon2018", fold=fold,
                             pretrained_dir="weights/rescue_pretrained")
    base_model = base_model.to(device)
    
    conditioner = SeverityConditioner(condition_dim=64)
    wrapped_model = FiLMWrappedChambon(base_model, conditioner)
    wrapped_model = wrapped_model.to(device)
    
    adapt_train_loader = create_subset_dataloader(train_dataset, adapt_train_ids, 
                                                  batch_size=32, shuffle=True)
    val_loader = create_subset_dataloader(train_dataset, val_ids, batch_size=32)
    
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
    osa_adapt_acc = evaluate_model(wrapped_model, test_loader, device)
    results["osa_adapt"] = {
        "test_accuracy": osa_adapt_acc,
        "val_accuracy": phase2_result["best_val_accuracy"],
        "epochs": phase2_result["total_epochs"],
    }
    
    del base_model, conditioner, wrapped_model, adapter
    torch.cuda.empty_cache()
    
    # 方法3: Full Fine-tuning
    base_model = build_model("Chambon2018")
    WeightLoader.load_weights(base_model, "Chambon2018", fold=fold,
                             pretrained_dir="weights/rescue_pretrained")
    base_model = base_model.to(device)
    
    conditioner = SeverityConditioner(condition_dim=64)
    wrapped_model = FiLMWrappedChambon(base_model, conditioner)
    wrapped_model = wrapped_model.to(device)
    
    loss_fn = SeverityAwareN1Loss()
    baseline = FullFineTuneBaseline(
        model=wrapped_model,
        loss_fn=loss_fn,
        lr=1e-3,
        patience=5,
        max_epochs=30,
    )
    
    ft_result = baseline.train(adapt_train_loader, val_loader)
    full_ft_acc = evaluate_model(wrapped_model, test_loader, device)
    results["full_ft"] = {
        "test_accuracy": full_ft_acc,
        "val_accuracy": ft_result["best_val_accuracy"],
        "epochs": ft_result["total_epochs"],
    }
    
    del base_model, conditioner, wrapped_model, baseline
    torch.cuda.empty_cache()
    
    return results

def main():
    print("="*60)
    print("现实数据量实验")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
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
    
    demographics_gen = DemographicsGenerator()
    
    # 实验配置
    budgets = [50, 100, 150]
    folds = [0, 1, 2]  # 3个fold
    seeds = [42, 123, 456]  # 3个seed
    
    total_experiments = len(budgets) * len(folds) * len(seeds) * 3  # 3个方法
    print(f"总实验数: {total_experiments}")
    print(f"预计时间: {total_experiments * 2 / 60:.1f} 小时\n")
    
    all_results = []
    output_dir = Path("results/realistic_budgets_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_count = 0
    start_time = datetime.now()
    
    for budget in budgets:
        for fold in folds:
            for seed in seeds:
                exp_count += 1
                print(f"\n[{exp_count}/{len(budgets)*len(folds)*len(seeds)}] "
                      f"Budget={budget}, Fold={fold}, Seed={seed}")
                print("-" * 60)
                
                try:
                    results = run_experiment(
                        budget, fold, seed, device,
                        all_patient_ids, severity_labels,
                        severity_data, demographics_gen
                    )
                    
                    # 保存结果
                    result_record = {
                        "budget": budget,
                        "fold": fold,
                        "seed": seed,
                        "no_adapt": results["no_adapt"],
                        "osa_adapt_test": results["osa_adapt"]["test_accuracy"],
                        "osa_adapt_val": results["osa_adapt"]["val_accuracy"],
                        "osa_adapt_epochs": results["osa_adapt"]["epochs"],
                        "full_ft_test": results["full_ft"]["test_accuracy"],
                        "full_ft_val": results["full_ft"]["val_accuracy"],
                        "full_ft_epochs": results["full_ft"]["epochs"],
                    }
                    all_results.append(result_record)
                    
                    # 打印结果
                    print(f"No Adapt:   {results['no_adapt']:.2%}")
                    print(f"OSA-Adapt:  {results['osa_adapt']['test_accuracy']:.2%} "
                          f"(val: {results['osa_adapt']['val_accuracy']:.2%}, "
                          f"epochs: {results['osa_adapt']['epochs']})")
                    print(f"Full FT:    {results['full_ft']['test_accuracy']:.2%} "
                          f"(val: {results['full_ft']['val_accuracy']:.2%}, "
                          f"epochs: {results['full_ft']['epochs']})")
                    
                    # 保存中间结果
                    with open(output_dir / "intermediate_results.json", "w") as f:
                        json.dump(all_results, f, indent=2)
                    
                except Exception as e:
                    print(f"❌ 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 计算统计
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    for budget in budgets:
        budget_results = [r for r in all_results if r["budget"] == budget]
        if not budget_results:
            continue
        
        no_adapt_mean = np.mean([r["no_adapt"] for r in budget_results])
        osa_test_mean = np.mean([r["osa_adapt_test"] for r in budget_results])
        osa_val_mean = np.mean([r["osa_adapt_val"] for r in budget_results])
        ft_test_mean = np.mean([r["full_ft_test"] for r in budget_results])
        ft_val_mean = np.mean([r["full_ft_val"] for r in budget_results])
        
        osa_test_std = np.std([r["osa_adapt_test"] for r in budget_results])
        ft_test_std = np.std([r["full_ft_test"] for r in budget_results])
        
        print(f"\nBudget={budget}:")
        print(f"  No Adapt:   {no_adapt_mean:.2%}")
        print(f"  OSA-Adapt:  {osa_test_mean:.2%} ± {osa_test_std:.2%} "
              f"(val: {osa_val_mean:.2%})")
        print(f"  Full FT:    {ft_test_mean:.2%} ± {ft_test_std:.2%} "
              f"(val: {ft_val_mean:.2%})")
        print(f"  Δ (OSA - FT): {osa_test_mean - ft_test_mean:+.2%}")
    
    # 保存最终结果
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = datetime.now() - start_time
    print(f"\n总耗时: {elapsed}")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
