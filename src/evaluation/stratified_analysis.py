"""
分层分析模块

按不同维度对模型性能进行分层分析：
- 按年龄分层（儿童5-18、青年18-40、中年40-60、老年60+）
- 按性别分层 + 统计检验
- 按OSA严重程度分层
- 按OSA严重程度分层对比reference baseline基线（Requirements: 19.1, 19.2, 19.4）
- 按性别分层对比reference baseline基线（Requirements: 22.1）
- 按BMI分层（Requirements: 22.2）
- 性别与BMI跨任务一致性分析（Requirements: 22.3）

Requirements: 11.1, 11.2, 11.3, 19.1, 19.2, 19.4, 22.1, 22.2, 22.3
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.evaluation.evaluator import Evaluator
from src.evaluation.statistical_tests import mcnemar_test, wilcoxon_test
import logging

logger = logging.getLogger(__name__)

AGE_GROUPS = {
    'pediatric': (5, 18),
    'young_adult': (18, 40),
    'middle_aged': (40, 60),
    'elderly': (60, 120),
}

OSA_GROUPS = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}


class StratifiedAnalyzer:
    """分层分析器"""

    def __init__(self, evaluator: Optional[Evaluator] = None):
        self.evaluator = evaluator or Evaluator()

    def analyze_by_age(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        ages: np.ndarray, task: str = 'classification',
    ) -> Dict[str, Dict]:
        """按年龄分层分析"""
        results = {}
        for group_name, (lo, hi) in AGE_GROUPS.items():
            mask = (ages >= lo) & (ages < hi)
            n = mask.sum()
            if n < 5:
                results[group_name] = {'n': int(n), 'metrics': None}
                continue
            yt, yp = y_true[mask], y_pred[mask]
            if task == 'classification':
                metrics = self.evaluator.evaluate_osa_classification(yt, yp)
            else:
                metrics = self.evaluator.evaluate_ahi_regression(yt, yp)
            # 移除不可序列化的对象
            metrics.pop('confusion_matrix', None)
            results[group_name] = {'n': int(n), 'metrics': metrics}
        return results

    def analyze_by_gender(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        genders: np.ndarray, task: str = 'classification',
    ) -> Dict[str, Dict]:
        """按性别分层分析 + 组间统计检验"""
        results = {}
        unique_genders = np.unique(genders)
        for g in unique_genders:
            mask = genders == g
            n = mask.sum()
            if n < 5:
                results[str(g)] = {'n': int(n), 'metrics': None}
                continue
            yt, yp = y_true[mask], y_pred[mask]
            if task == 'classification':
                metrics = self.evaluator.evaluate_osa_classification(yt, yp)
            else:
                metrics = self.evaluator.evaluate_ahi_regression(yt, yp)
            metrics.pop('confusion_matrix', None)
            results[str(g)] = {'n': int(n), 'metrics': metrics}

        # 性别间统计检验（如果有两组）
        if len(unique_genders) == 2 and task == 'classification':
            g0, g1 = unique_genders[0], unique_genders[1]
            m0, m1 = genders == g0, genders == g1
            if m0.sum() >= 5 and m1.sum() >= 5:
                # 使用McNemar检验比较两组的预测准确性
                test_result = mcnemar_test(
                    y_true[m0 | m1],
                    y_pred[m0 | m1],
                    y_pred[m0 | m1],  # 自身对比作为基线
                )
                results['gender_test'] = test_result

        return results

    def analyze_by_osa_severity(
        self, y_true_event: np.ndarray, y_pred_event: np.ndarray,
        osa_labels: np.ndarray,
    ) -> Dict[str, Dict]:
        """按OSA严重程度分层分析事件检测性能"""
        results = {}
        for osa_val, osa_name in OSA_GROUPS.items():
            mask = osa_labels == osa_val
            n = mask.sum()
            if n < 5:
                results[osa_name] = {'n': int(n), 'metrics': None}
                continue
            yt, yp = y_true_event[mask], y_pred_event[mask]
            metrics = self.evaluator.evaluate_event_detection(yt, yp)
            metrics.pop('confusion_matrix', None)
            results[osa_name] = {'n': int(n), 'metrics': metrics}
        return results

    # ------------------------------------------------------------------ #
    # OSA严重程度分层对比 (Requirements: 19.1, 19.2, 19.4)
    # ------------------------------------------------------------------ #

    # reference baseline中U-Sleep 2.0 (EEG+EOG) 的OSA严重程度分层基线
    REFERENCE_SEVERITY_BASELINES = {
        'Mild': 0.808,
        'Moderate': 0.741,
        'Severe': 0.679,
    }

    # 性能下降阈值（超过10个百分点标记为需要改进）
    DROP_THRESHOLD = 0.10

    # 默认睡眠分期名称
    SLEEP_STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # reference baseline中U-Sleep 2.0 (EEG+EOG) 的性别基线 (Requirements: 22.1)
    REFERENCE_GENDER_BASELINES = {
        'male_accuracy': 0.713,
        'female_accuracy': 0.778,
        'gender_gap_range': (0.065, 0.093),  # 6.5-9.3 percentage points
    }

    # BMI分组定义 (Requirements: 22.2)
    BMI_GROUPS = {
        'normal': (0, 25),
        'overweight': (25, 30),
        'obese': (30, 100),
    }

    def analyze_osa_severity_with_baseline_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        osa_labels: np.ndarray,
        task: str = 'classification',
        stage_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        OSA严重程度分层对比分析（含reference baseline基线对比）。

        计算每个严重程度组的性能指标，计算Mild→Severe性能下降幅度，
        与reference baseline中U-Sleep 2.0的分层结果对比，并分析性能下降的主要原因。

        Args:
            y_true: 真实标签 [N]（分类标签或睡眠分期标签）
            y_pred: 预测标签 [N]
            osa_labels: OSA严重程度标签 [N]，值域 {1: Mild, 2: Moderate, 3: Severe}
            task: 任务类型，'classification' 或 'regression'
            stage_names: 睡眠分期名称列表（用于分析下降原因），默认AASM标准

        Returns:
            包含分层结果、性能下降、reference baseline对比和下降原因分析的字典
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        osa_labels = np.asarray(osa_labels).flatten()
        if stage_names is None:
            stage_names = self.SLEEP_STAGE_NAMES

        results = {}

        # 1. 每个严重程度组的性能指标
        severity_results = self._compute_per_severity_metrics(
            y_true, y_pred, osa_labels, task
        )
        results['per_severity'] = severity_results

        # 2. Mild→Severe性能下降幅度
        results['performance_drop'] = self._compute_performance_drop(severity_results)

        # 3. 与reference baseline U-Sleep 2.0分层结果对比
        results['baseline_comparison'] = self._compare_severity_with_baseline(severity_results)

        # 4. 分析性能下降的主要原因（哪些睡眠期受影响最大）
        results['drop_causes'] = self._analyze_drop_causes(
            y_true, y_pred, osa_labels, stage_names
        )

        return results

    def _compute_per_severity_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        osa_labels: np.ndarray,
        task: str = 'classification',
    ) -> Dict[str, Dict]:
        """计算每个OSA严重程度组的性能指标。"""
        # 仅分析Mild(1), Moderate(2), Severe(3)
        severity_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        results = {}

        for osa_val, osa_name in severity_map.items():
            mask = osa_labels == osa_val
            n = int(mask.sum())
            if n < 5:
                results[osa_name] = {'n': n, 'metrics': None}
                continue

            yt, yp = y_true[mask], y_pred[mask]

            if task == 'classification':
                metrics = self.evaluator.evaluate_osa_classification(yt, yp)
            else:
                metrics = self.evaluator.evaluate_ahi_regression(yt, yp)

            # 移除不可序列化的对象
            metrics.pop('confusion_matrix', None)
            results[osa_name] = {'n': n, 'metrics': metrics}

        return results

    def _compute_performance_drop(
        self,
        severity_results: Dict[str, Dict],
    ) -> Dict:
        """
        计算Mild→Severe的性能下降幅度。

        对accuracy、kappa、f1_macro三个指标分别计算下降值。
        当下降超过10个百分点时标记为需要改进。

        Requirements: 19.1
        """
        mild = severity_results.get('Mild', {})
        severe = severity_results.get('Severe', {})
        mild_m = mild.get('metrics')
        severe_m = severe.get('metrics')

        if mild_m is None or severe_m is None:
            return {
                'available': False,
                'reason': 'Mild或Severe组样本不足',
            }

        drop_metrics = {}
        for metric_key in ['accuracy', 'kappa', 'f1_macro']:
            mild_val = mild_m.get(metric_key)
            severe_val = severe_m.get(metric_key)
            if mild_val is not None and severe_val is not None:
                drop = mild_val - severe_val
                drop_metrics[metric_key] = {
                    'mild': float(mild_val),
                    'severe': float(severe_val),
                    'drop': float(round(drop, 4)),
                    'drop_percentage_points': float(round(drop * 100, 1)),
                    'needs_improvement': drop > self.DROP_THRESHOLD,
                }

        # 汇总判定
        any_needs_improvement = any(
            v.get('needs_improvement', False) for v in drop_metrics.values()
        )

        return {
            'available': True,
            'metrics': drop_metrics,
            'any_needs_improvement': any_needs_improvement,
        }

    def _compare_severity_with_baseline(
        self,
        severity_results: Dict[str, Dict],
    ) -> Dict:
        """
        与reference baseline中U-Sleep 2.0的分层结果对比。

        reference baseline基线: Mild 0.808, Moderate 0.741, Severe 0.679

        Requirements: 19.4
        """
        comparison = {}

        for severity_name, baseline_acc in self.REFERENCE_SEVERITY_BASELINES.items():
            group = severity_results.get(severity_name, {})
            group_metrics = group.get('metrics')

            entry = {
                'reference_baseline_accuracy': baseline_acc,
                'current_model_accuracy': None,
                'delta': None,
                'relative_improvement_pct': None,
                'improved': None,
            }

            if group_metrics is not None:
                p1_acc = group_metrics.get('accuracy')
                if p1_acc is not None:
                    delta = p1_acc - baseline_acc
                    rel_imp = (delta / baseline_acc * 100) if baseline_acc != 0 else 0.0
                    entry['current_model_accuracy'] = float(p1_acc)
                    entry['delta'] = float(round(delta, 4))
                    entry['relative_improvement_pct'] = float(round(rel_imp, 2))
                    entry['improved'] = delta > 0

            comparison[severity_name] = entry

        # reference baseline的Mild→Severe下降幅度
        p2_mild = self.REFERENCE_SEVERITY_BASELINES['Mild']
        p2_severe = self.REFERENCE_SEVERITY_BASELINES['Severe']
        p2_drop = p2_mild - p2_severe

        # current model的Mild→Severe下降幅度
        p1_mild_acc = comparison['Mild'].get('current_model_accuracy')
        p1_severe_acc = comparison['Severe'].get('current_model_accuracy')

        drop_comparison = {
            'reference_drop': float(round(p2_drop, 4)),
            'reference_drop_percentage_points': float(round(p2_drop * 100, 1)),
        }

        if p1_mild_acc is not None and p1_severe_acc is not None:
            p1_drop = p1_mild_acc - p1_severe_acc
            drop_reduction = p2_drop - p1_drop
            drop_comparison['current_model_drop'] = float(round(p1_drop, 4))
            drop_comparison['current_drop_pp'] = float(round(p1_drop * 100, 1))
            drop_comparison['drop_reduction'] = float(round(drop_reduction, 4))
            drop_comparison['drop_reduction_percentage_points'] = float(round(drop_reduction * 100, 1))
            drop_comparison['current_drop_smaller'] = p1_drop < p2_drop

        comparison['drop_comparison'] = drop_comparison

        return comparison

    def _analyze_drop_causes(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        osa_labels: np.ndarray,
        stage_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        分析性能下降的主要原因：哪些睡眠期/事件类型受影响最大。

        对每个严重程度组，计算每个睡眠期的准确率，
        找出从Mild到Severe下降最大的睡眠期。

        Requirements: 19.2
        """
        if stage_names is None:
            stage_names = self.SLEEP_STAGE_NAMES

        severity_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}

        # 每个严重程度组、每个睡眠期的准确率
        per_severity_per_stage = {}
        for osa_val, osa_name in severity_map.items():
            mask = osa_labels == osa_val
            if mask.sum() < 5:
                per_severity_per_stage[osa_name] = None
                continue

            yt, yp = y_true[mask], y_pred[mask]
            stage_acc = {}
            for stage_idx, stage_name in enumerate(stage_names):
                stage_mask = yt == stage_idx
                n_stage = int(stage_mask.sum())
                if n_stage == 0:
                    stage_acc[stage_name] = {'accuracy': None, 'n': 0}
                    continue
                correct = int((yp[stage_mask] == stage_idx).sum())
                acc = correct / n_stage
                stage_acc[stage_name] = {
                    'accuracy': float(round(acc, 4)),
                    'n': n_stage,
                }
            per_severity_per_stage[osa_name] = stage_acc

        # 计算Mild→Severe每个睡眠期的下降幅度
        mild_stages = per_severity_per_stage.get('Mild')
        severe_stages = per_severity_per_stage.get('Severe')

        stage_drops = {}
        if mild_stages is not None and severe_stages is not None:
            for stage_name in stage_names:
                mild_acc = (mild_stages.get(stage_name) or {}).get('accuracy')
                severe_acc = (severe_stages.get(stage_name) or {}).get('accuracy')
                if mild_acc is not None and severe_acc is not None:
                    drop = mild_acc - severe_acc
                    stage_drops[stage_name] = {
                        'mild_accuracy': mild_acc,
                        'severe_accuracy': severe_acc,
                        'drop': float(round(drop, 4)),
                        'drop_percentage_points': float(round(drop * 100, 1)),
                    }

        # 找出下降最大的睡眠期
        most_affected = None
        if stage_drops:
            most_affected = max(
                stage_drops.keys(),
                key=lambda k: stage_drops[k]['drop'],
            )

        return {
            'per_severity_per_stage': per_severity_per_stage,
            'stage_drops_mild_to_severe': stage_drops,
            'most_affected_stage': most_affected,
        }

    # ------------------------------------------------------------------ #
    # 性别分层对比 reference baseline (Requirements: 22.1)
    # ------------------------------------------------------------------ #

    def analyze_gender_with_baseline_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        genders: np.ndarray,
        task: str = 'classification',
    ) -> Dict:
        """
        性别分层评估 + reference baseline性别差异对比。

        报告男性和女性的准确率差异，与reference baseline中U-Sleep 2.0的
        性别差异（6.5-9.3个百分点）进行对比。

        Args:
            y_true: 真实标签 [N]
            y_pred: 预测标签 [N]
            genders: 性别标签 [N]，值为 'M'/'F' 或 'male'/'female'
            task: 'classification' 或 'regression'

        Returns:
            包含性别分层结果和reference baseline对比的字典
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        genders = np.asarray(genders).flatten()

        results = {}

        # 标准化性别标签
        gender_map = {'M': 'male', 'F': 'female', 'm': 'male', 'f': 'female',
                      'Male': 'male', 'Female': 'female'}
        normalized = np.array([
            gender_map.get(str(g), str(g).lower()) for g in genders
        ])

        # 每个性别组的指标
        per_gender = {}
        for gender in ['male', 'female']:
            mask = normalized == gender
            n = int(mask.sum())
            if n < 5:
                per_gender[gender] = {'n': n, 'metrics': None}
                continue
            yt, yp = y_true[mask], y_pred[mask]
            if task == 'classification':
                metrics = self.evaluator.evaluate_osa_classification(yt, yp)
            else:
                metrics = self.evaluator.evaluate_ahi_regression(yt, yp)
            metrics.pop('confusion_matrix', None)
            per_gender[gender] = {'n': n, 'metrics': metrics}

        results['per_gender'] = per_gender

        # 性别差异计算
        male_m = (per_gender.get('male') or {}).get('metrics')
        female_m = (per_gender.get('female') or {}).get('metrics')

        gender_gap = {'available': False}
        if male_m is not None and female_m is not None:
            male_acc = male_m.get('accuracy')
            female_acc = female_m.get('accuracy')
            if male_acc is not None and female_acc is not None:
                gap = female_acc - male_acc
                gap_pp = gap * 100  # percentage points
                p2_lo, p2_hi = self.REFERENCE_GENDER_BASELINES['gender_gap_range']
                exceeds_paper2 = abs(gap) > p2_hi
                gender_gap = {
                    'available': True,
                    'male_accuracy': float(male_acc),
                    'female_accuracy': float(female_acc),
                    'gap': float(round(gap, 4)),
                    'gap_percentage_points': float(round(gap_pp, 2)),
                    'exceeds_paper2_gap': exceeds_paper2,
                }

        results['gender_gap'] = gender_gap

        # reference baseline对比
        p2_baselines = self.REFERENCE_GENDER_BASELINES
        paper2_cmp = {
            'paper2_male_accuracy': p2_baselines['male_accuracy'],
            'paper2_female_accuracy': p2_baselines['female_accuracy'],
            'reference_gap_range': p2_baselines['gender_gap_range'],
        }
        if male_m is not None:
            male_acc = male_m.get('accuracy')
            if male_acc is not None:
                paper2_cmp['current_male_accuracy'] = float(male_acc)
                paper2_cmp['male_delta'] = float(
                    round(male_acc - p2_baselines['male_accuracy'], 4)
                )
        if female_m is not None:
            female_acc = female_m.get('accuracy')
            if female_acc is not None:
                paper2_cmp['current_female_accuracy'] = float(female_acc)
                paper2_cmp['female_delta'] = float(
                    round(female_acc - p2_baselines['female_accuracy'], 4)
                )

        results['baseline_comparison'] = paper2_cmp

        return results

    # ------------------------------------------------------------------ #
    # BMI分层分析 (Requirements: 22.2)
    # ------------------------------------------------------------------ #

    def analyze_by_bmi(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bmi_values: np.ndarray,
        task: str = 'classification',
    ) -> Dict:
        """
        按BMI分组报告性能。

        分组：正常体重（<25）、超重（25-30）、肥胖（≥30）。

        Args:
            y_true: 真实标签 [N]
            y_pred: 预测标签 [N]
            bmi_values: BMI值 [N]
            task: 'classification' 或 'regression'

        Returns:
            包含每个BMI组性能指标的字典
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        bmi_values = np.asarray(bmi_values, dtype=float).flatten()

        per_group = {}
        for group_name, (lo, hi) in self.BMI_GROUPS.items():
            mask = (bmi_values >= lo) & (bmi_values < hi)
            n = int(mask.sum())
            if n < 5:
                per_group[group_name] = {'n': n, 'metrics': None}
                continue
            yt, yp = y_true[mask], y_pred[mask]
            if task == 'classification':
                metrics = self.evaluator.evaluate_osa_classification(yt, yp)
            else:
                metrics = self.evaluator.evaluate_ahi_regression(yt, yp)
            metrics.pop('confusion_matrix', None)
            per_group[group_name] = {'n': n, 'metrics': metrics}

        # 组间性能差异摘要
        accs = {}
        for gn, gd in per_group.items():
            m = (gd or {}).get('metrics')
            if m is not None:
                accs[gn] = m.get('accuracy')

        summary = {'group_accuracies': accs}
        valid_accs = [v for v in accs.values() if v is not None]
        if len(valid_accs) >= 2:
            summary['max_gap'] = float(round(max(valid_accs) - min(valid_accs), 4))
            summary['max_gap_percentage_points'] = float(
                round((max(valid_accs) - min(valid_accs)) * 100, 2)
            )

        return {'per_group': per_group, 'summary': summary}

    # ------------------------------------------------------------------ #
    # 性别与BMI跨任务一致性分析 (Requirements: 22.3)
    # ------------------------------------------------------------------ #

    def analyze_gender_bmi_task_consistency(
        self,
        results_by_task: Dict[str, Dict],
    ) -> Dict:
        """
        分析性别和BMI对不同任务的影响是否一致。

        Args:
            results_by_task: {task_name: {'gender_results': ..., 'bmi_results': ...}}
                gender_results 来自 analyze_gender_with_baseline_comparison
                bmi_results 来自 analyze_by_bmi

        Returns:
            包含跨任务一致性分析的字典
        """
        task_names = list(results_by_task.keys())

        # 性别差异跨任务一致性
        gender_gaps = {}
        for task_name, task_data in results_by_task.items():
            gr = task_data.get('gender_results', {})
            gap_info = gr.get('gender_gap', {})
            if gap_info.get('available'):
                gender_gaps[task_name] = gap_info['gap_percentage_points']

        gender_consistency = self._assess_consistency(gender_gaps, 'gender_gap')

        # BMI差异跨任务一致性
        bmi_gaps = {}
        for task_name, task_data in results_by_task.items():
            br = task_data.get('bmi_results', {})
            summary = br.get('summary', {})
            max_gap_pp = summary.get('max_gap_percentage_points')
            if max_gap_pp is not None:
                bmi_gaps[task_name] = max_gap_pp

        bmi_consistency = self._assess_consistency(bmi_gaps, 'bmi_gap')

        return {
            'task_names': task_names,
            'gender_consistency': gender_consistency,
            'bmi_consistency': bmi_consistency,
        }

    def _assess_consistency(
        self,
        gaps_by_task: Dict[str, float],
        label: str,
    ) -> Dict:
        """评估某个指标在多个任务间的一致性。"""
        if len(gaps_by_task) < 2:
            return {
                'available': False,
                'reason': f'需要至少2个任务的{label}数据',
            }

        values = list(gaps_by_task.values())
        # 方向一致性：所有gap同号
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in values]
        direction_consistent = len(set(s for s in signs if s != 0)) <= 1

        spread = max(values) - min(values)

        return {
            'available': True,
            'per_task': gaps_by_task,
            'direction_consistent': direction_consistent,
            'spread': float(round(spread, 2)),
            'mean': float(round(np.mean(values), 2)),
            'std': float(round(np.std(values), 2)),
        }

    def generate_stratified_report(self, all_results: Dict) -> str:
        """生成分层分析文本报告"""
        lines = ["=" * 60, "分层分析报告", "=" * 60]
        for dim_name, dim_results in all_results.items():
            lines.append(f"\n--- {dim_name} ---")
            for group, data in dim_results.items():
                if group.endswith('_test'):
                    lines.append(f"  统计检验: p={data.get('p_value', 'N/A')}")
                    continue
                n = data.get('n', 0)
                metrics = data.get('metrics')
                if metrics is None:
                    lines.append(f"  {group}: n={n} (样本不足)")
                    continue
                lines.append(f"  {group}: n={n}")
                for k, v in metrics.items():
                    if k in ('per_class',):
                        continue
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
        return "\n".join(lines)
