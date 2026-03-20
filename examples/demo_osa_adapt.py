#!/usr/bin/env python
"""
OSA-Adapt Demo — run the full adaptation pipeline on synthetic data.

This script demonstrates the core OSA-Adapt workflow without requiring
any real clinical data. It generates synthetic PSG-like signals and
patient features, then runs:
  1. Base model forward pass
  2. FiLM wrapping with severity conditioning
  3. Two-phase progressive adaptation (BN adapt + FiLM fine-tune)
  4. Two-pass inference (AHI estimation from staging)

Usage:
    python examples/demo_osa_adapt.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.film_adapter import FiLMAdapter
from src.adaptation.severity_conditioner import SeverityConditioner
from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
from src.adaptation.model_builder import build_model
from src.adaptation.ahi_estimator import AHIEstimator
from src.adaptation.wrapped_models import FiLMWrappedChambon


def generate_synthetic_data(n_patients=20, epochs_per_patient=50, seed=42):
    """Generate synthetic PSG-like data for demonstration.

    Returns:
        signals: [total_epochs, 1, 3000] tensor
        labels: [total_epochs] tensor (sleep stages 0-4)
        patient_features: dict of tensors
    """
    rng = np.random.RandomState(seed)
    all_signals, all_labels = [], []
    all_ahi, all_severity, all_age, all_sex, all_bmi = [], [], [], [], []

    for i in range(n_patients):
        # Simulate patient characteristics
        severity = rng.randint(0, 4)  # 0=Normal, 1=Mild, 2=Moderate, 3=Severe
        ahi = [rng.uniform(0, 5), rng.uniform(5, 15),
               rng.uniform(15, 30), rng.uniform(30, 60)][severity]
        age = rng.uniform(30, 70)
        sex = rng.randint(0, 2)
        bmi = rng.uniform(20, 40)

        for _ in range(epochs_per_patient):
            # Synthetic 30-second EEG signal at 100 Hz
            signal = rng.randn(3000).astype(np.float32)
            # More severe OSA -> more N1/Wake, less N3
            stage_probs = [0.15, 0.10, 0.45, 0.15, 0.15]  # base
            if severity >= 2:
                stage_probs = [0.25, 0.20, 0.35, 0.05, 0.15]
            label = rng.choice(5, p=stage_probs)

            all_signals.append(signal)
            all_labels.append(label)
            all_ahi.append(ahi)
            all_severity.append(severity)
            all_age.append(age)
            all_sex.append(sex)
            all_bmi.append(bmi)

    signals = torch.tensor(np.array(all_signals)).unsqueeze(1)  # [N, 1, 3000]
    labels = torch.tensor(all_labels, dtype=torch.long)
    pf = {
        "ahi": torch.tensor(all_ahi, dtype=torch.float32),
        "severity": torch.tensor(all_severity, dtype=torch.long),
        "age": torch.tensor(all_age, dtype=torch.float32),
        "sex": torch.tensor(all_sex, dtype=torch.long),
        "bmi": torch.tensor(all_bmi, dtype=torch.float32),
    }
    return signals, labels, pf


class SyntheticPFDataset(torch.utils.data.Dataset):
    """Wraps signals + labels + patient_features into a Dataset."""

    def __init__(self, signals, labels, pf):
        self.signals = signals
        self.labels = labels
        self.pf = pf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.signals[idx],
            self.labels[idx],
            {k: v[idx] for k, v in self.pf.items()},
        )


def main():
    print("=" * 60)
    print("OSA-Adapt Demo: Synthetic Data Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- 1. Generate synthetic data ---
    print("[1/5] Generating synthetic data...")
    signals, labels, pf = generate_synthetic_data(n_patients=20, epochs_per_patient=50)
    n_total = len(labels)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train_ds = SyntheticPFDataset(signals[:n_train], labels[:n_train],
                                   {k: v[:n_train] for k, v in pf.items()})
    val_ds = SyntheticPFDataset(signals[n_train:n_train+n_val], labels[n_train:n_train+n_val],
                                 {k: v[n_train:n_train+n_val] for k, v in pf.items()})
    test_ds = SyntheticPFDataset(signals[n_train+n_val:], labels[n_train+n_val:],
                                  {k: v[n_train+n_val:] for k, v in pf.items()})

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # --- 2. Build base model ---
    print("\n[2/5] Building Chambon2018 base model...")
    base_model = build_model("Chambon2018", use_physioex=False).to(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Base model parameters: {n_params:,}")

    # --- 3. Wrap with FiLM ---
    print("\n[3/5] Wrapping with FiLM + SeverityConditioner...")
    conditioner = SeverityConditioner(condition_dim=64).to(device)
    wrapped = FiLMWrappedChambon(base_model, conditioner=conditioner).to(device)
    film_params = sum(p.numel() for p in wrapped.get_trainable_params())
    print(f"  FiLM trainable parameters: {film_params:,} "
          f"({film_params/n_params*100:.1f}% of base)")

    # --- 4. Two-phase adaptation ---
    print("\n[4/5] Running two-phase progressive adaptation...")
    from src.adaptation.progressive_adapter import ProgressiveAdapter

    loss_fn = SeverityAwareN1Loss(
        gamma_n1_base=2.5, gamma_n1_increment=0.5, n1_weight_multiplier=2.0,
    ).to(device)

    adapter = ProgressiveAdapter(
        model=wrapped, conditioner=wrapped.conditioner,
        loss_fn=loss_fn, lr=1e-3, patience=3, max_epochs=5,
        bn_momentum=0.01,
    )

    # Phase 1: BN adaptation (label-free)
    print("  Phase 1: BN adaptation...")
    p1_stats = adapter.phase1_bn_adapt(train_loader)
    print(f"    Processed {p1_stats['num_samples']} samples")

    # Phase 2: FiLM fine-tuning
    print("  Phase 2: FiLM fine-tuning...")
    p2_stats = adapter.phase2_film_finetune(train_loader, val_loader)
    print(f"    Best val accuracy: {p2_stats['best_val_accuracy']:.4f} "
          f"({p2_stats['total_epochs']} epochs)")

    # --- 5. Evaluate ---
    print("\n[5/5] Evaluating on test set...")
    wrapped.eval()
    correct = total = 0
    with torch.no_grad():
        for x, tgt, feat in test_loader:
            x, tgt = x.to(device), tgt.to(device)
            feat = {k: v.to(device) for k, v in feat.items()}
            pred = wrapped(x, feat).argmax(dim=1)
            correct += (pred == tgt).sum().item()
            total += len(tgt)
    print(f"  Test accuracy: {correct/total:.4f} ({correct}/{total})")

    # --- Bonus: AHI Estimator demo ---
    print("\n[Bonus] AHI Estimator demo...")
    estimator = AHIEstimator()
    rng = np.random.RandomState(0)
    demo_preds = [rng.randint(0, 5, 100) for _ in range(10)]
    demo_ahi = rng.uniform(0, 50, 10)
    stats = estimator.fit(demo_preds, demo_ahi)
    print(f"  Fitted on 10 patients: R2={stats['r_squared']:.3f}, MAE={stats['mae']:.1f}")
    est = estimator.estimate(rng.randint(0, 5, 100))
    sev = estimator.ahi_to_severity(est)
    sev_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    print(f"  Estimated AHI: {est:.1f} -> {sev_names[sev]}")

    print("\n" + "=" * 60)
    print("Demo complete. All core components verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
