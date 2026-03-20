"""Core module tests for OSA-Adapt framework."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFiLMAdapter:
    """Test FiLM conditioning layer."""

    def test_identity_init(self):
        """FiLM should be identity mapping at initialization."""
        from src.adaptation.film_adapter import FiLMAdapter

        film = FiLMAdapter(feature_dim=64, condition_dim=32)
        x = torch.randn(4, 64)
        c = torch.zeros(4, 32)
        out = film(x, c)
        assert torch.allclose(x, out, atol=1e-5)

    def test_3d_input(self):
        """FiLM should handle [B, C, T] inputs."""
        from src.adaptation.film_adapter import FiLMAdapter

        film = FiLMAdapter(feature_dim=64, condition_dim=32)
        x = torch.randn(4, 64, 100)
        c = torch.randn(4, 32)
        out = film(x, c)
        assert out.shape == (4, 64, 100)

    def test_gradient_flow(self):
        """Gradients should flow through FiLM."""
        from src.adaptation.film_adapter import FiLMAdapter

        film = FiLMAdapter(feature_dim=64, condition_dim=32)
        x = torch.randn(4, 64, requires_grad=True)
        c = torch.randn(4, 32, requires_grad=True)
        out = film(x, c)
        out.sum().backward()
        assert x.grad is not None
        assert c.grad is not None


class TestSeverityConditioner:
    """Test patient feature encoder."""

    def test_output_shape(self):
        """Conditioner should produce [B, condition_dim] output."""
        from src.adaptation.severity_conditioner import SeverityConditioner

        cond = SeverityConditioner(condition_dim=64)
        ahi = torch.tensor([5.0, 15.0, 30.0, 50.0])
        severity = torch.tensor([0, 1, 2, 3])
        age = torch.tensor([45.0, 50.0, 55.0, 60.0])
        sex = torch.tensor([0, 1, 0, 1])
        bmi = torch.tensor([25.0, 28.0, 31.0, 35.0])
        out = cond(ahi, severity, age, sex, bmi)
        assert out.shape == (4, 64)

    def test_nan_handling(self):
        """Conditioner should handle NaN values gracefully."""
        from src.adaptation.severity_conditioner import SeverityConditioner

        cond = SeverityConditioner(condition_dim=64)
        ahi = torch.tensor([float("nan"), 15.0])
        severity = torch.tensor([0, 1])
        age = torch.tensor([45.0, float("nan")])
        sex = torch.tensor([0, 1])
        bmi = torch.tensor([25.0, 28.0])
        out = cond(ahi, severity, age, sex, bmi)
        assert not torch.isnan(out).any()


class TestSeverityAwareN1Loss:
    """Test severity-aware focal loss."""

    def test_basic_forward(self):
        """Loss should compute without errors."""
        from src.adaptation.severity_aware_loss import SeverityAwareN1Loss

        loss_fn = SeverityAwareN1Loss(num_classes=5)
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        severity = torch.randint(0, 4, (8,))
        loss = loss_fn(inputs, targets, severity)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_invalid_labels(self):
        """Loss should return zero for all-invalid labels."""
        from src.adaptation.severity_aware_loss import SeverityAwareN1Loss

        loss_fn = SeverityAwareN1Loss(num_classes=5)
        inputs = torch.randn(4, 5)
        targets = torch.full((4,), -1, dtype=torch.long)
        severity = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(inputs, targets, severity)
        assert loss.item() == 0.0

    def test_n1_weight_constraint(self):
        """N1 weight should be >= multiplier * mean of other weights."""
        from src.adaptation.severity_aware_loss import SeverityAwareN1Loss

        loss_fn = SeverityAwareN1Loss(n1_weight_multiplier=2.0)
        counts = torch.tensor([1000, 50, 3000, 500, 800], dtype=torch.float)
        loss_fn.set_class_weights(counts)
        w = loss_fn.class_weights
        other_mean = (w[0] + w[2] + w[3] + w[4]) / 4
        assert w[1] >= other_mean * 2.0 - 1e-5


class TestCrossValidator:
    """Test patient-level stratified cross-validation."""

    def test_no_patient_leakage(self):
        """Same patient should not appear in both train and test."""
        from src.adaptation.cross_validator import CrossValidator

        cv = CrossValidator(n_folds=5, seed=42)
        ids = [f"p{i:03d}" for i in range(50)]
        sevs = [i % 4 for i in range(50)]
        folds = cv.split(ids, sevs)
        assert len(folds) == 5
        for train_ids, test_ids in folds:
            assert len(set(train_ids) & set(test_ids)) == 0

    def test_all_patients_covered(self):
        """All patients should appear in exactly one test fold."""
        from src.adaptation.cross_validator import CrossValidator

        cv = CrossValidator(n_folds=5, seed=42)
        ids = [f"p{i:03d}" for i in range(50)]
        sevs = [i % 4 for i in range(50)]
        folds = cv.split(ids, sevs)
        all_test = []
        for _, test_ids in folds:
            all_test.extend(test_ids)
        assert sorted(all_test) == sorted(ids)


class TestStratifiedSampler:
    """Test severity-stratified few-shot sampler."""

    def test_budget_respected(self):
        """Sampler should return exactly budget patients."""
        from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

        sampler = SeverityStratifiedFewShotSampler(seed=42)
        ids = [f"p{i:03d}" for i in range(100)]
        sevs = [i % 4 for i in range(100)]
        selected = sampler.sample(ids, sevs, budget=20)
        assert len(selected) == 20

    def test_all_groups_represented(self):
        """All severity groups should be represented when budget allows."""
        from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

        sampler = SeverityStratifiedFewShotSampler(seed=42)
        ids = [f"p{i:03d}" for i in range(100)]
        sevs = [i % 4 for i in range(100)]
        selected = sampler.sample(ids, sevs, budget=20)
        selected_sevs = set()
        id_to_sev = dict(zip(ids, sevs))
        for pid in selected:
            selected_sevs.add(id_to_sev[pid])
        assert len(selected_sevs) == 4


class TestAHIEstimator:
    """Test AHI estimation from staging results."""

    def test_fit_and_estimate(self):
        """Estimator should fit and produce non-negative AHI."""
        from src.adaptation.ahi_estimator import AHIEstimator

        est = AHIEstimator()
        rng = np.random.RandomState(42)
        preds_list = [rng.randint(0, 5, 100) for _ in range(20)]
        ahi_values = rng.uniform(0, 60, 20)
        stats = est.fit(preds_list, ahi_values)
        assert stats["n_patients"] == 20
        assert est.is_fitted

        ahi_est = est.estimate(rng.randint(0, 5, 100))
        assert ahi_est >= 0

    def test_severity_mapping(self):
        """AHI to severity mapping should follow AASM criteria."""
        from src.adaptation.ahi_estimator import AHIEstimator

        est = AHIEstimator()
        assert est.ahi_to_severity(3.0) == 0   # Normal
        assert est.ahi_to_severity(10.0) == 1  # Mild
        assert est.ahi_to_severity(20.0) == 2  # Moderate
        assert est.ahi_to_severity(40.0) == 3  # Severe


class TestEvaluator:
    """Test sleep staging evaluator."""

    def test_perfect_prediction(self):
        """Perfect predictions should yield accuracy=1.0."""
        from src.adaptation.evaluator import SleepStageEvaluator

        evaluator = SleepStageEvaluator()
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        result = evaluator.evaluate_patient(y, y, severity=2)
        assert result["accuracy"] == 1.0
        assert result["kappa"] == 1.0

    def test_fold_aggregation(self):
        """Fold evaluation should aggregate patient results."""
        from src.adaptation.evaluator import SleepStageEvaluator

        evaluator = SleepStageEvaluator()
        results = [
            {"accuracy": 0.8, "kappa": 0.7, "macro_f1": 0.75,
             "w_f1": 0.8, "n1_f1": 0.5, "n2_f1": 0.8, "n3_f1": 0.7, "rem_f1": 0.9,
             "severity": 0.0},
            {"accuracy": 0.6, "kappa": 0.5, "macro_f1": 0.55,
             "w_f1": 0.7, "n1_f1": 0.3, "n2_f1": 0.6, "n3_f1": 0.5, "rem_f1": 0.7,
             "severity": 3.0},
        ]
        fold_result = evaluator.evaluate_fold(results)
        assert 0 < fold_result["acc"] < 1
        assert fold_result["severe_acc"] == 0.6
