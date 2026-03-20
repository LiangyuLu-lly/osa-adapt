"""
医学专用指标分析器

实现医学领域标准的性能指标计算：
- 多分类ROC曲线 (One-vs-Rest)
- 各睡眠阶段的Sensitivity/Specificity
- AUC置信区间 (Bootstrap)
- Precision-Recall曲线
- 医学指标详细报告

Requirements: 15.1, 15.2, 15.3, 15.4
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MedicalMetricsAnalyzer:
    """医学指标分析器"""
    
    def __init__(self, sleep_stages: Optional[List[str]] = None, confidence_level: float = 0.95, n_bootstrap: int = 1000, random_seed: int = 42):
        """
        初始化医学指标分析器
        
        Args:
            sleep_stages: 睡眠阶段名称列表，默认为AASM标准
            confidence_level: 置信区间水平，默认95%
            n_bootstrap: Bootstrap迭代次数，默认1000
            random_seed: 随机种子
        """
        self.sleep_stages = sleep_stages or ['Wake', 'N1', 'N2', 'N3', 'REM']
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)
        
    def compute_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算医学专用指标
        
        Args:
            y_true: 真实标签 [N]
            y_pred: 预测标签 [N] 
            y_pred_proba: 预测概率 [N, C]，可选
            
        Returns:
            包含医学指标的字典
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        results = {}
        
        # 基本信息
        n_classes = len(self.sleep_stages)
        results['n_classes'] = n_classes
        results['n_samples'] = len(y_true)
        results['class_names'] = self.sleep_stages.copy()
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        results['confusion_matrix'] = cm.tolist()
        
        # 计算每个类别的Sensitivity和Specificity
        sensitivity_specificity = self._compute_sensitivity_specificity(cm)
        results['sensitivity'] = sensitivity_specificity['sensitivity']
        results['specificity'] = sensitivity_specificity['specificity']
        results['ppv'] = sensitivity_specificity['ppv']  # Positive Predictive Value
        results['npv'] = sensitivity_specificity['npv']  # Negative Predictive Value
        
        # 如果有预测概率，计算ROC和PR曲线相关指标
        if y_pred_proba is not None:
            roc_results = self.generate_roc_curves(y_true, y_pred_proba)
            results['roc_curves'] = roc_results['roc_curves']
            results['roc_auc'] = roc_results['roc_auc']
            results['macro_auc'] = roc_results['macro_auc']
            results['weighted_auc'] = roc_results['weighted_auc']
            
            pr_results = self.generate_pr_curves(y_true, y_pred_proba)
            results['pr_curves'] = pr_results['pr_curves']
            results['pr_auc'] = pr_results['pr_auc']
            results['macro_ap'] = pr_results['macro_ap']
            results['weighted_ap'] = pr_results['weighted_ap']
            
            # 计算AUC置信区间
            auc_ci = self.compute_auc_confidence_intervals(y_true, y_pred_proba)
            results['auc_confidence_intervals'] = auc_ci
        
        # 整体指标
        results['overall_accuracy'] = float(np.trace(cm) / np.sum(cm))
        results['macro_sensitivity'] = float(np.mean(list(results['sensitivity'].values())))
        results['macro_specificity'] = float(np.mean(list(results['specificity'].values())))
        
        return results
    
    def _compute_sensitivity_specificity(self, confusion_matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        从混淆矩阵计算每个类别的Sensitivity、Specificity、PPV、NPV
        
        Args:
            confusion_matrix: 混淆矩阵 [n_classes, n_classes]
            
        Returns:
            包含各指标的字典
        """
        n_classes = confusion_matrix.shape[0]
        sensitivity = {}
        specificity = {}
        ppv = {}  # Positive Predictive Value (Precision)
        npv = {}  # Negative Predictive Value
        
        for i in range(n_classes):
            class_name = self.sleep_stages[i] if i < len(self.sleep_stages) else f'Class_{i}'
            
            # True Positive, False Positive, False Negative, True Negative
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            tn = confusion_matrix.sum() - tp - fp - fn
            
            # Sensitivity (Recall) = TP / (TP + FN)
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            sensitivity[class_name] = sens
            
            # Specificity = TN / (TN + FP)
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            specificity[class_name] = spec
            
            # PPV (Precision) = TP / (TP + FP)
            ppv_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            ppv[class_name] = ppv_val
            
            # NPV = TN / (TN + FN)
            npv_val = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            npv[class_name] = npv_val
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        }
    
    def generate_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        生成多分类ROC曲线 (One-vs-Rest)
        
        Args:
            y_true: 真实标签 [N]
            y_pred_proba: 预测概率 [N, C]
            
        Returns:
            包含ROC曲线数据和AUC值的字典
        """
        n_classes = y_pred_proba.shape[1]
        
        # 二值化标签 (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])
        
        roc_curves = {}
        roc_auc = {}
        
        # 计算每个类别的ROC曲线
        for i in range(n_classes):
            class_name = self.sleep_stages[i] if i < len(self.sleep_stages) else f'Class_{i}'
            
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            
            roc_curves[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc_val)
            }
            roc_auc[class_name] = float(roc_auc_val)
        
        # 计算宏平均和加权平均AUC
        class_counts = np.bincount(y_true, minlength=n_classes)
        weights = class_counts / len(y_true)
        
        macro_auc = float(np.mean(list(roc_auc.values())))
        weighted_auc = float(np.sum([roc_auc[self.sleep_stages[i]] * weights[i] for i in range(n_classes)]))
        
        return {
            'roc_curves': roc_curves,
            'roc_auc': roc_auc,
            'macro_auc': macro_auc,
            'weighted_auc': weighted_auc
        }
    
    def generate_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        生成Precision-Recall曲线
        
        Args:
            y_true: 真实标签 [N]
            y_pred_proba: 预测概率 [N, C]
            
        Returns:
            包含PR曲线数据和AP值的字典
        """
        n_classes = y_pred_proba.shape[1]
        
        # 二值化标签 (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])
        
        pr_curves = {}
        pr_auc = {}
        
        # 计算每个类别的PR曲线
        for i in range(n_classes):
            class_name = self.sleep_stages[i] if i < len(self.sleep_stages) else f'Class_{i}'
            
            precision, recall, thresholds = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
            ap_score = average_precision_score(y_bin[:, i], y_pred_proba[:, i])
            
            pr_curves[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'ap': float(ap_score)
            }
            pr_auc[class_name] = float(ap_score)
        
        # 计算宏平均和加权平均AP
        class_counts = np.bincount(y_true, minlength=n_classes)
        weights = class_counts / len(y_true)
        
        macro_ap = float(np.mean(list(pr_auc.values())))
        weighted_ap = float(np.sum([pr_auc[self.sleep_stages[i]] * weights[i] for i in range(n_classes)]))
        
        return {
            'pr_curves': pr_curves,
            'pr_auc': pr_auc,
            'macro_ap': macro_ap,
            'weighted_ap': weighted_ap
        }
    
    def compute_auc_confidence_intervals(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        使用Bootstrap方法计算AUC置信区间
        
        Args:
            y_true: 真实标签 [N]
            y_pred_proba: 预测概率 [N, C]
            
        Returns:
            包含每个类别AUC置信区间的字典
        """
        n_classes = y_pred_proba.shape[1]
        n_samples = len(y_true)
        
        # 二值化标签 (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])
        
        confidence_intervals = {}
        
        for i in range(n_classes):
            class_name = self.sleep_stages[i] if i < len(self.sleep_stages) else f'Class_{i}'
            
            # Bootstrap采样计算AUC分布
            bootstrap_aucs = []
            
            for _ in range(self.n_bootstrap):
                # 有放回采样
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                y_boot = y_bin[indices, i]
                prob_boot = y_pred_proba[indices, i]
                
                # 如果bootstrap样本中只有一个类别，跳过
                if len(np.unique(y_boot)) < 2:
                    continue
                
                try:
                    fpr, tpr, _ = roc_curve(y_boot, prob_boot)
                    bootstrap_auc = auc(fpr, tpr)
                    if np.isfinite(bootstrap_auc):
                        bootstrap_aucs.append(bootstrap_auc)
                except Exception:
                    continue
            
            if len(bootstrap_aucs) > 0:
                bootstrap_aucs = np.array(bootstrap_aucs)
                alpha = 1 - self.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = float(np.percentile(bootstrap_aucs, lower_percentile))
                ci_upper = float(np.percentile(bootstrap_aucs, upper_percentile))
                mean_auc = float(np.mean(bootstrap_aucs))
                std_auc = float(np.std(bootstrap_aucs))
                
                confidence_intervals[class_name] = {
                    'mean': mean_auc,
                    'std': std_auc,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'confidence_level': self.confidence_level,
                    'n_bootstrap_samples': len(bootstrap_aucs)
                }
            else:
                confidence_intervals[class_name] = {
                    'mean': 0.5,
                    'std': 0.0,
                    'ci_lower': 0.5,
                    'ci_upper': 0.5,
                    'confidence_level': self.confidence_level,
                    'n_bootstrap_samples': 0
                }
        
        return confidence_intervals
    
    def generate_medical_report(self, results: Dict[str, Any]) -> str:
        """
        生成医学指标文本报告
        
        Args:
            results: compute_medical_metrics的返回结果
            
        Returns:
            格式化的文本报告
        """
        lines = []
        lines.append("=" * 80)
        lines.append("医学专用指标分析报告")
        lines.append("=" * 80)
        
        # 基本信息
        lines.append(f"\n数据集信息:")
        lines.append(f"  样本数量: {results['n_samples']}")
        lines.append(f"  类别数量: {results['n_classes']}")
        lines.append(f"  类别名称: {', '.join(results['class_names'])}")
        lines.append(f"  整体准确率: {results['overall_accuracy']:.4f}")
        
        # Sensitivity和Specificity
        lines.append(f"\n各类别Sensitivity (敏感性):")
        for class_name, sens in results['sensitivity'].items():
            lines.append(f"  {class_name}: {sens:.4f}")
        lines.append(f"  宏平均: {results['macro_sensitivity']:.4f}")
        
        lines.append(f"\n各类别Specificity (特异性):")
        for class_name, spec in results['specificity'].items():
            lines.append(f"  {class_name}: {spec:.4f}")
        lines.append(f"  宏平均: {results['macro_specificity']:.4f}")
        
        # PPV和NPV
        lines.append(f"\n各类别PPV (阳性预测值):")
        for class_name, ppv_val in results['ppv'].items():
            lines.append(f"  {class_name}: {ppv_val:.4f}")
        
        lines.append(f"\n各类别NPV (阴性预测值):")
        for class_name, npv_val in results['npv'].items():
            lines.append(f"  {class_name}: {npv_val:.4f}")
        
        # ROC AUC (如果可用)
        if 'roc_auc' in results:
            lines.append(f"\n各类别ROC AUC:")
            for class_name, auc_val in results['roc_auc'].items():
                lines.append(f"  {class_name}: {auc_val:.4f}")
            lines.append(f"  宏平均AUC: {results['macro_auc']:.4f}")
            lines.append(f"  加权平均AUC: {results['weighted_auc']:.4f}")
        
        # PR AUC (如果可用)
        if 'pr_auc' in results:
            lines.append(f"\n各类别PR AUC (Average Precision):")
            for class_name, ap_val in results['pr_auc'].items():
                lines.append(f"  {class_name}: {ap_val:.4f}")
            lines.append(f"  宏平均AP: {results['macro_ap']:.4f}")
            lines.append(f"  加权平均AP: {results['weighted_ap']:.4f}")
        
        # AUC置信区间 (如果可用)
        if 'auc_confidence_intervals' in results:
            lines.append(f"\nAUC {int(results['auc_confidence_intervals'][results['class_names'][0]]['confidence_level']*100)}% 置信区间:")
            for class_name, ci_data in results['auc_confidence_intervals'].items():
                lines.append(f"  {class_name}: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}] (mean: {ci_data['mean']:.4f})")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        保存医学指标结果到JSON文件
        
        Args:
            results: compute_medical_metrics的返回结果
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 添加元数据
        results_with_metadata = {
            'metadata': {
                'analyzer': 'MedicalMetricsAnalyzer',
                'sleep_stages': self.sleep_stages,
                'confidence_level': self.confidence_level,
                'n_bootstrap': self.n_bootstrap,
                'timestamp': self._get_timestamp()
            },
            'results': results
        }
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"医学指标结果已保存到: {output_path}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def demo_medical_metrics():
    """演示医学指标分析器的使用"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # 模拟真实标签和预测
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # 添加一些错误预测
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    # 模拟预测概率
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    # 让预测概率与真实标签有一定相关性
    for i in range(n_samples):
        y_pred_proba[i, y_true[i]] += 0.3
        y_pred_proba[i] /= y_pred_proba[i].sum()
    
    # 创建分析器并计算指标
    analyzer = MedicalMetricsAnalyzer()
    results = analyzer.compute_medical_metrics(y_true, y_pred, y_pred_proba)
    
    # 生成报告
    report = analyzer.generate_medical_report(results)
    print(report)
    
    # 保存结果
    analyzer.save_results(results, 'results/medical_metrics_demo.json')


if __name__ == "__main__":
    demo_medical_metrics()