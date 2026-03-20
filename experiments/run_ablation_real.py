#!/usr/bin/env python
"""
OSA-Adapt 真实消融实验脚本

复用主实验的GPU训练管线，系统性移除OSA-Adapt的各个组件。
消融变体：
  - full_model: 完整OSA-Adapt（基线）
  - no_severity_conditioning: 用固定零向量代替severity conditioning
  - no_n1_loss: 用标准CrossEntropy代替SeverityAwareN1Loss
  - no_progressive: 跳过Phase 1 BN适应
  - no_stratified: 用随机采样代替分层采样

用法:
    PYTHONPATH=. python experiments/run_ablation_real.py
"""
import argparse, copy, json, logging, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS = [5, 10, 20, 30, 50, 65, 100]
DEFAULT_MODELS = ["Chambon2018", "TinySleepNet"]
DEFAULT_OUTPUT_DIR = "results/ablation"
DEFAULT_PKL_DIR = "data/preprocessed"
DEFAULT_SEVERITY_JSON = "data/patient_severity.json"
SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}

ABLATION_CONFIGS = {
    "full_model": dict(use_severity=True, use_n1_loss=True, use_stratified=True, use_bn=True),
    "no_severity_conditioning": dict(use_severity=False, use_n1_loss=True, use_stratified=True, use_bn=True),
    "no_n1_loss": dict(use_severity=True, use_n1_loss=False, use_stratified=True, use_bn=True),
    "no_progressive": dict(use_severity=True, use_n1_loss=True, use_stratified=True, use_bn=False),
    "no_stratified": dict(use_severity=True, use_n1_loss=True, use_stratified=False, use_bn=True),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OSA-Adapt ablation")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pkl-dir", default=DEFAULT_PKL_DIR)
    p.add_argument("--severity-json", default=DEFAULT_SEVERITY_JSON)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    p.add_argument("--skip-completed", action="store_true", default=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)

def setup_logging(level, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(str(Path(outdir)/"ablation.log"), encoding="utf-8")],
        force=True)

def load_severity_data(path):
    p = Path(path)
    if not p.exists(): return {}
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def get_patient_ids(pkl_dir):
    p = Path(pkl_dir)
    if not p.exists(): return []
    ids = set()
    for f in sorted(p.glob("*.pkl")):
        parts = f.stem.split("_", 1)
        if parts: ids.add(f"patient_{parts[0].zfill(3)}")
    return sorted(ids)

def get_sev_labels(pids, sev_data):
    out = []
    for pid in pids:
        if pid in sev_data:
            s = str(sev_data[pid].get("osa_severity","normal")).lower()
            out.append(SEVERITY_MAP.get(s, 0))
        else:
            out.append(hash(pid) % 4)
    return out

class TypeFixDL:
    def __init__(self, loader): self.loader = loader; self.dataset = loader.dataset
    def __iter__(self):
        for batch in self.loader:
            if len(batch) == 3:
                x, t, pf = batch
                pf["severity"] = pf["severity"].long(); pf["sex"] = pf["sex"].long()
                yield x, t, pf
            elif len(batch) == 2:
                x, pf = batch
                pf["severity"] = pf["severity"].long(); pf["sex"] = pf["sex"].long()
                yield x, pf
            else: yield batch
    def __len__(self): return len(self.loader)

def make_dl(ds, bs, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0,
                      drop_last=False, pin_memory=torch.cuda.is_available())

def make_subset_dl(ds, pids, bs, shuffle=True):
    idx = []
    for pid in pids: idx.extend(ds.get_patient_epoch_indices(pid))
    if not idx: return DataLoader(Subset(ds, []), batch_size=bs)
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=shuffle,
                      num_workers=0, drop_last=False, pin_memory=torch.cuda.is_available())


class ZeroConditioner(nn.Module):
    """输出固定零向量的conditioner，用于no_severity_conditioning消融。"""
    def __init__(self, dim=64):
        super().__init__()
        self.condition_dim = dim
        # 需要与SeverityConditioner相同的接口
        self.severity_embedding = nn.Embedding(4, 16)  # dummy
        self.sex_embedding = nn.Embedding(2, 16)  # dummy
    def forward(self, ahi, severity, age, sex, bmi):
        bs = ahi.shape[0]
        dev = ahi.device
        return torch.zeros(bs, self.condition_dim, device=dev)

class IgnoreSeverityLoss(nn.Module):
    """包装CrossEntropyLoss，忽略第三个severity参数，并过滤无效标签。"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, inputs, targets, severity=None):
        # 过滤超出范围的标签（保留0-4的有效睡眠分期标签）
        valid_mask = (targets >= 0) & (targets < inputs.shape[1])
        if valid_mask.all():
            return self.ce(inputs, targets)
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        return self.ce(inputs[valid_mask], targets[valid_mask])

@torch.no_grad()
def predict_patient(model, ds, indices, device, bs=256):
    model.eval()
    yt, yp = [], []
    for s in range(0, len(indices), bs):
        bi = indices[s:s+bs]
        sigs, labs = [], []
        fl = {k: [] for k in ["ahi","severity","age","sex","bmi"]}
        for i in bi:
            sig, lab, pf = ds[i]
            sigs.append(sig); labs.append(lab)
            for k in fl: fl[k].append(pf[k])
        x = torch.stack(sigs).to(device)
        pf = {k: torch.stack(v).to(device) for k,v in fl.items()}
        pf["severity"] = pf["severity"].long(); pf["sex"] = pf["sex"].long()
        out = model(x, pf)
        pred = out.argmax(dim=1).cpu().numpy()
        yt.extend(labs); yp.extend(pred.tolist())
    return np.array(yt), np.array(yp)

def run_one(model_name, budget, abl_name, abl_cfg, train_ds, test_ds,
            train_ids, test_ids, sev_data, sev_map, seed, bs, device, fold=None):
    """执行单次消融实验。"""
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader
    from src.adaptation.progressive_adapter import ProgressiveAdapter
    from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
    from src.adaptation.severity_conditioner import SeverityConditioner
    from src.adaptation.wrapped_models import FiLMWrappedChambon, FiLMWrappedTinySleepNet
    from src.adaptation.evaluator import SleepStageEvaluator
    from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

    torch.manual_seed(seed); np.random.seed(seed)

    # 采样
    train_sevs = [sev_map.get(p, 0) for p in train_ids]
    if abl_cfg["use_stratified"]:
        sampler = SeverityStratifiedFewShotSampler(seed=seed)
        adapt_ids = sampler.sample(train_ids, train_sevs, budget)
    else:
        rng = np.random.RandomState(seed)
        n = min(budget, len(train_ids))
        idx = rng.choice(len(train_ids), size=n, replace=False)
        adapt_ids = [train_ids[i] for i in sorted(idx)]

    n_val = max(1, len(adapt_ids) // 5)
    rng = np.random.RandomState(seed)
    sh = list(adapt_ids); rng.shuffle(sh)
    val_ids = sh[:n_val]; at_ids = sh[n_val:] or list(adapt_ids)

    # 模型
    base = build_model(model_name)
    WeightLoader.load_weights(base, model_name, fold=fold)
    cond = SeverityConditioner(condition_dim=64)
    if model_name == "Chambon2018":
        wrapped = FiLMWrappedChambon(base, cond)
    else:
        wrapped = FiLMWrappedTinySleepNet(base, cond)
    wrapped = wrapped.to(device)

    # DataLoaders
    ul = TypeFixDL(make_dl(train_ds, bs, shuffle=False))
    tl = TypeFixDL(make_subset_dl(train_ds, at_ids, bs))
    vl = TypeFixDL(make_subset_dl(train_ds, val_ids, bs, shuffle=False))

    # 损失
    if abl_cfg["use_n1_loss"]:
        loss_fn = SeverityAwareN1Loss(gamma_n1_base=2.5, gamma_n1_increment=0.5, n1_weight_multiplier=2.0)
    else:
        loss_fn = IgnoreSeverityLoss()

    # 消融: no_severity_conditioning
    if not abl_cfg["use_severity"]:
        zc = ZeroConditioner(64).to(device)
        wrapped.conditioner = zc
        cond = zc

    adapter = ProgressiveAdapter(model=wrapped, conditioner=cond,
                                  loss_fn=loss_fn, lr=1e-3, patience=5, max_epochs=50)
    if abl_cfg["use_bn"]:
        adapter.phase1_bn_adapt(ul)
    adapter.phase2_film_finetune(tl, vl)

    # 评估
    evaluator = SleepStageEvaluator()
    pr = []
    for pid in test_ids:
        idx = test_ds.get_patient_epoch_indices(pid)
        if not idx: continue
        sev = sev_map.get(pid, 0)
        if pid in sev_data:
            s = str(sev_data[pid].get("osa_severity","normal")).lower()
            sev = SEVERITY_MAP.get(s, 0)
        yt, yp = predict_patient(wrapped, test_ds, idx, device, bs)
        pr.append(evaluator.evaluate_patient(yt, yp, sev))

    fm = evaluator.evaluate_fold(pr)
    fm["n_test_patients"] = len(pr)
    fm["ablation"] = abl_name

    del wrapped, base
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return fm


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

def run_main(args):
    from src.adaptation.cross_validator import CrossValidator
    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator

    logger.info("=" * 60)
    logger.info("OSA-Adapt 消融实验（真实GPU训练）")
    logger.info("模型: %s | 预算: %s | 消融: %s", args.models, args.budgets, args.ablations)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    sev_data = load_severity_data(args.severity_json)
    pids = get_patient_ids(args.pkl_dir)
    if not pids:
        logger.error("未找到患者数据"); return

    sev_labels = get_sev_labels(pids, sev_data)
    sev_map_global = dict(zip(pids, sev_labels))
    logger.info("患者: %d", len(pids))

    cv = CrossValidator(n_folds=args.n_folds, seed=42)
    folds = cv.split(pids, sev_labels)

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    all_results = []
    total = len(args.models) * len(args.budgets) * len(args.ablations) * args.n_folds
    ei = 0

    for mn in args.models:
        for fi, (tr_ids, te_ids) in enumerate(folds):
            logger.info("--- %s Fold %d ---", mn, fi)
            dg = DemographicsGenerator()
            tr_ds = PSGDataset(patient_ids=tr_ids, pkl_dir=args.pkl_dir,
                               severity_data=sev_data, demographics_generator=dg)
            te_ds = PSGDataset(patient_ids=te_ids, pkl_dir=args.pkl_dir,
                               severity_data=sev_data, demographics_generator=dg)
            logger.info("  训练: %d epochs, 测试: %d epochs", len(tr_ds), len(te_ds))

            sm = {}
            for pid in tr_ids + te_ids:
                if pid in sev_data:
                    s = str(sev_data[pid].get("osa_severity","normal")).lower()
                    sm[pid] = SEVERITY_MAP.get(s, 0)
                else:
                    sm[pid] = hash(pid) % 4

            for bud in args.budgets:
                for abl in args.ablations:
                    ei += 1
                    key = f"{abl}_{mn}_budget{bud}_fold{fi}_seed{args.seed}"
                    rf = outdir / f"{key}.json"

                    if args.skip_completed and rf.exists():
                        logger.info("[%d/%d] 跳过: %s", ei, total, key)
                        with open(rf) as f: all_results.append(json.load(f))
                        continue

                    logger.info("[%d/%d] %s", ei, total, key)
                    try:
                        t0 = time.time()
                        m = run_one(mn, bud, abl, ABLATION_CONFIGS[abl],
                                    tr_ds, te_ds, tr_ids, te_ids, sev_data, sm,
                                    args.seed, args.batch_size, device, fold=fi)
                        el = time.time() - t0
                        m.update(status="completed", elapsed_seconds=el,
                                 model_name=mn, data_budget=bud, fold=fi,
                                 seed=args.seed, experiment_name=key)
                        logger.info("  acc=%.4f n1_f1=%.4f %.1fs",
                                    m.get("acc",0), m.get("n1_f1",0), el)
                    except Exception as e:
                        logger.error("失败 %s: %s\n%s", key, e, traceback.format_exc())
                        m = dict(status="failed", error=str(e), model_name=mn,
                                 data_budget=bud, fold=fi, seed=args.seed,
                                 ablation=abl, experiment_name=key)

                    with open(rf, "w", encoding="utf-8") as f:
                        json.dump(m, f, ensure_ascii=False, indent=2, default=_json_default)
                    all_results.append(m)

            del tr_ds, te_ds
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 聚合
    completed = [r for r in all_results if r.get("status") == "completed"]
    logger.info("完成: %d/%d", len(completed), len(all_results))

    if completed:
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        mkeys = ["acc","kappa","macro_f1","n1_f1","severe_acc","severe_n1_f1"]
        for r in completed:
            grouped[r["model_name"]][r["ablation"]][r["data_budget"]].append(r)

        summary = {}
        for model, abls in grouped.items():
            summary[model] = {}
            for abl, buds in abls.items():
                summary[model][abl] = {}
                for bud, runs in buds.items():
                    ag = {"n_runs": len(runs)}
                    for k in mkeys:
                        vs = [r.get(k,0) for r in runs if r.get(k) is not None]
                        ag[k] = float(np.mean(vs)) if vs else 0.0
                        ag[f"{k}_std"] = float(np.std(vs)) if vs else 0.0
                    summary[model][abl][str(bud)] = ag

        with open(outdir/"ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

        # 跨budget总汇总
        overall = {}
        for model, abls in grouped.items():
            overall[model] = {}
            for abl, buds in abls.items():
                ar = [r for runs in buds.values() for r in runs]
                ag = {"n_runs": len(ar)}
                for k in mkeys:
                    vs = [r.get(k,0) for r in ar if r.get(k) is not None]
                    ag[k] = float(np.mean(vs)) if vs else 0.0
                    ag[f"{k}_std"] = float(np.std(vs)) if vs else 0.0
                overall[model][abl] = ag

        with open(outdir/"ablation_overall.json", "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2, default=_json_default)

        logger.info("结果已保存到 %s", outdir)

    logger.info("=" * 60)
    logger.info("消融实验完成")

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level, args.output_dir)
    run_main(args)
