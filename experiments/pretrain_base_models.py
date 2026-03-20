#!/usr/bin/env python
"""
预训练基础模型脚本

在我们的临床PSG数据上训练Chambon2018和TinySleepNet的基础权重。
使用5-fold CV，每个fold用训练折数据训练一个基础模型，保存checkpoint。
这些checkpoint将作为adaptation实验的"预训练权重"。

策略：对于每个CV fold，用该fold的训练集（不含adaptation子集）训练基础模型。
这确保了预训练权重不会泄露测试集信息。

用法:
    PYTHONPATH=. python experiments/pretrain_base_models.py
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_PKL_DIR = "data/preprocessed"
DEFAULT_SEVERITY_JSON = "data/patient_severity.json"
DEFAULT_OUTPUT_DIR = "weights/pretrained"
DEFAULT_N_FOLDS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 7

SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def parse_args():
    parser = argparse.ArgumentParser(description="预训练基础睡眠分期模型")
    parser.add_argument("--models", nargs="+", default=["Chambon2018", "TinySleepNet"])
    parser.add_argument("--pkl-dir", default=DEFAULT_PKL_DIR)
    parser.add_argument("--severity-json", default=DEFAULT_SEVERITY_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="指定要训练的fold（默认全部）")
    return parser.parse_args()


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(output_dir) / "pretrain.log"), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers, force=True,
    )


def load_severity_data(json_path: str) -> Dict[str, Dict]:
    path = Path(json_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "patients" in raw:
        return {p["patient_id"]: p for p in raw["patients"]}
    return raw



def get_patient_ids_and_labels(pkl_dir: str, severity_data: Dict) -> Tuple[List[str], List[int]]:
    """获取所有患者ID和严重程度标签。"""
    pkl_path = Path(pkl_dir)
    patient_ids = set()
    for f in sorted(pkl_path.glob("*.pkl")):
        parts = f.stem.split("_", 1)
        if parts:
            pid = f"patient_{parts[0].zfill(3)}"
            patient_ids.add(pid)
    patient_ids = sorted(patient_ids)

    labels = []
    for pid in patient_ids:
        if pid in severity_data:
            sev_str = str(severity_data[pid].get("osa_severity", "normal")).lower()
            labels.append(SEVERITY_MAP.get(sev_str, 0))
        else:
            labels.append(hash(pid) % 4)
    return patient_ids, labels


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 7,
) -> Dict:
    """训练一个fold的基础模型。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(max_epochs):
        # 训练
        model.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0

        for batch in train_loader:
            x, targets, pf = batch
            x = x.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # 模型需要 patient_features 参数
            pf_device = {k: v.to(device) for k, v in pf.items()}
            pf_device["severity"] = pf_device["severity"].long()
            pf_device["sex"] = pf_device["sex"].long()
            outputs = model(x, pf_device)

            valid = targets >= 0
            if valid.any():
                loss = loss_fn(outputs[valid], targets[valid])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                preds = outputs[valid].argmax(dim=1)
                n_correct += (preds == targets[valid]).sum().item()
                n_total += valid.sum().item()

        train_acc = n_correct / max(n_total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_n1_correct = 0
        val_n1_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x, targets, pf = batch
                x = x.to(device)
                targets = targets.to(device)
                pf_device = {k: v.to(device) for k, v in pf.items()}
                pf_device["severity"] = pf_device["severity"].long()
                pf_device["sex"] = pf_device["sex"].long()
                outputs = model(x, pf_device)

                valid = targets >= 0
                if valid.any():
                    preds = outputs[valid].argmax(dim=1)
                    val_correct += (preds == targets[valid]).sum().item()
                    val_total += valid.sum().item()
                    n1_mask = targets[valid] == 1
                    if n1_mask.any():
                        val_n1_correct += (preds[n1_mask] == 1).sum().item()
                        val_n1_total += n1_mask.sum().item()

        val_acc = val_correct / max(val_total, 1)
        val_n1_recall = val_n1_correct / max(val_n1_total, 1)
        scheduler.step(val_acc)

        history.append({
            "epoch": epoch, "train_loss": avg_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_n1_recall": val_n1_recall,
            "lr": optimizer.param_groups[0]["lr"],
        })

        logger.info(
            "Epoch %d/%d: loss=%.4f train_acc=%.4f val_acc=%.4f n1_recall=%.4f lr=%.6f",
            epoch + 1, max_epochs, avg_loss, train_acc, val_acc, val_n1_recall,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("早停: %d epochs无改善", no_improve)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "total_epochs": len(history),
        "history": history,
    }



def main():
    args = parse_args()
    setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("预训练基础模型")
    logger.info("模型: %s", args.models)
    logger.info("Folds: %s", args.folds or list(range(args.n_folds)))
    logger.info("Max epochs: %d, LR: %g, Patience: %d", args.max_epochs, args.lr, args.patience)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    # 加载数据
    severity_data = load_severity_data(args.severity_json)
    patient_ids, severity_labels = get_patient_ids_and_labels(args.pkl_dir, severity_data)
    logger.info("患者数: %d", len(patient_ids))

    if len(patient_ids) == 0:
        logger.error("无患者数据")
        return

    # CV 划分
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(patient_ids, severity_labels))

    folds_to_run = args.folds if args.folds else list(range(args.n_folds))

    # 延迟导入
    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator
    from src.adaptation.model_builder import build_model

    demographics_gen = DemographicsGenerator()

    for model_name in args.models:
        logger.info("\n" + "=" * 60)
        logger.info("模型: %s", model_name)
        logger.info("=" * 60)

        for fold_idx in folds_to_run:
            train_indices, test_indices = folds[fold_idx]
            train_pids = [patient_ids[i] for i in train_indices]
            test_pids = [patient_ids[i] for i in test_indices]

            # 检查是否已有checkpoint
            ckpt_path = Path(args.output_dir) / f"{model_name}_fold{fold_idx}_best.pt"
            if ckpt_path.exists():
                logger.info("Fold %d: checkpoint已存在，跳过: %s", fold_idx, ckpt_path)
                continue

            logger.info("\nFold %d: 训练集 %d 患者, 测试集 %d 患者",
                        fold_idx, len(train_pids), len(test_pids))

            # 从训练集中划出10%作为验证集
            n_val = max(1, len(train_pids) // 10)
            rng = np.random.RandomState(42 + fold_idx)
            shuffled = list(train_pids)
            rng.shuffle(shuffled)
            val_pids = shuffled[:n_val]
            actual_train_pids = shuffled[n_val:]

            logger.info("  实际训练: %d, 验证: %d", len(actual_train_pids), len(val_pids))

            # 创建 Dataset
            train_dataset = PSGDataset(
                patient_ids=actual_train_pids,
                pkl_dir=args.pkl_dir,
                severity_data=severity_data,
                demographics_generator=demographics_gen,
            )
            val_dataset = PSGDataset(
                patient_ids=val_pids,
                pkl_dir=args.pkl_dir,
                severity_data=severity_data,
                demographics_generator=demographics_gen,
            )

            logger.info("  训练样本: %d, 验证样本: %d", len(train_dataset), len(val_dataset))

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logger.warning("  数据集为空，跳过")
                continue

            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available(),
            )

            # 构建模型
            model = build_model(model_name)

            # 训练
            start_time = time.time()
            result = train_one_fold(
                model, train_loader, val_loader, device,
                max_epochs=args.max_epochs, lr=args.lr, patience=args.patience,
            )
            elapsed = time.time() - start_time

            logger.info(
                "Fold %d 完成: val_acc=%.4f, epochs=%d, 耗时=%.1fs",
                fold_idx, result["best_val_acc"], result["total_epochs"], elapsed,
            )

            # 保存 checkpoint
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "fold": fold_idx,
                "best_val_acc": result["best_val_acc"],
                "total_epochs": result["total_epochs"],
                "history": result["history"],
            }, str(ckpt_path))
            logger.info("  Checkpoint 保存: %s", ckpt_path)

            # 清理
            del model, train_dataset, val_dataset, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("\n预训练完成！")


if __name__ == "__main__":
    main()
