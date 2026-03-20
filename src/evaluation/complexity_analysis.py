"""
模型复杂度分析器

实现OSA-Adapt全面实验扩展规范中的任务7：模型复杂度分析
- 计算FLOPs (使用fvcore库)
- 测量GPU推理延迟 (批次大小1, 8, 32)
- 监控内存使用峰值
- 分析效率权衡关系
- 生成详细的复杂度分析报告

Requirements: 16.1, 16.2, 16.3, 16.4
Author: PSG Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from fvcore.nn import FlopCountMode, flop_count_table, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    print("警告: fvcore库未安装，FLOPs计算将不可用")
    FlopCountMode = None
    flop_count_table = None
    flop_count = None
    FVCORE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """复杂度指标数据类"""
    # 模型参数
    total_params: int
    trainable_params: int
    model_size_mb: float
    
    # FLOPs指标
    flops: Optional[int] = None
    flops_g: Optional[float] = None  # GFLOPs
    
    # 延迟指标 (毫秒)
    inference_latency_ms: float = 0.0
    inference_std_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # 内存指标 (MB)
    peak_memory_mb: float = 0.0
    training_memory_mb: float = 0.0
    
    # 效率指标
    accuracy_per_flop: Optional[float] = None
    accuracy_per_parameter: Optional[float] = None
    
    # 批次大小
    batch_size: int = 1


@dataclass
class BatchSizeAnalysis:
    """批次大小分析结果"""
    batch_size: int
    latency_ms: float
    latency_std_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    memory_efficiency: float  # samples/MB


class ModelComplexityAnalyzer:
    """
    模型复杂度分析器
    
    实现完整的模型复杂度分析功能：
    1. 参数数量统计
    2. FLOPs计算
    3. 推理延迟测量
    4. 内存使用监控
    5. 效率权衡分析
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        batch_sizes: List[int] = None
    ):
        """
        初始化复杂度分析器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
            warmup_runs: 预热运行次数
            benchmark_runs: 基准测试运行次数
            batch_sizes: 要测试的批次大小列表
        """
        self.device = torch.device(device)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.batch_sizes = batch_sizes or [1, 8, 32]
        
        # 检查CUDA可用性
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU模式")
            self.device = torch.device("cpu")
    
    def count_parameters(self, model: nn.Module) -> Tuple[int, int, float]:
        """
        统计模型参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            (总参数数, 可训练参数数, 模型大小MB)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算模型大小 (假设float32，每个参数4字节)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return total_params, trainable_params, model_size_mb
    
    def compute_flops(
        self, 
        model: nn.Module, 
        input_shape: Dict[str, Tuple[int, ...]]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        计算模型FLOPs
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典，格式: {"modality": (batch_size, ...)}
            
        Returns:
            (FLOPs数量, FLOPs表格字符串)
        """
        if not FVCORE_AVAILABLE:
            logger.warning("fvcore库不可用，使用简化的FLOPs估算")
            return self._estimate_flops_simple(model, input_shape)
        
        try:
            model = model.to(self.device)
            model.eval()
            
            # 创建输入张量
            inputs = {}
            for modality, shape in input_shape.items():
                inputs[modality] = torch.randn(shape, device=self.device)
            
            # 使用fvcore计算FLOPs
            with torch.no_grad():
                # 方法1: 尝试使用flop_count
                try:
                    flops_dict = flop_count(model, (inputs,))
                    total_flops = sum(flops_dict.values())
                    flops_table = flop_count_table(model, (inputs,))
                    return total_flops, flops_table
                except Exception as e1:
                    logger.warning(f"flop_count方法失败: {e1}")
                    
                    # 方法2: 尝试直接调用
                    try:
                        from fvcore.nn.flop_count import flop_count as fc
                        flops_dict = fc(model, inputs)
                        total_flops = sum(flops_dict.values())
                        return total_flops, str(flops_dict)
                    except Exception as e2:
                        logger.warning(f"备用FLOPs计算方法也失败: {e2}")
                        return self._estimate_flops_simple(model, input_shape)
            
        except Exception as e:
            logger.error(f"FLOPs计算失败: {e}")
            return self._estimate_flops_simple(model, input_shape)
    
    def _estimate_flops_simple(
        self, 
        model: nn.Module, 
        input_shape: Dict[str, Tuple[int, ...]]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        简化的FLOPs估算方法
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典
            
        Returns:
            (估算的FLOPs数量, 说明字符串)
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            
            # 简单估算：假设每个参数平均进行2次运算（乘法+加法）
            # 这是一个非常粗略的估算
            estimated_flops = total_params * 2
            
            explanation = f"简化估算: {total_params:,} 参数 × 2 运算/参数 = {estimated_flops:,} FLOPs"
            
            return estimated_flops, explanation
            
        except Exception as e:
            logger.error(f"简化FLOPs估算也失败: {e}")
            return None, None
    
    def measure_inference_latency(
        self,
        model: nn.Module,
        input_shape: Dict[str, Tuple[int, ...]],
        batch_size: int = 1
    ) -> Tuple[float, float, float]:
        """
        测量推理延迟
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典 (不包含batch维度)
            batch_size: 批次大小
            
        Returns:
            (平均延迟ms, 标准差ms, 吞吐量samples/sec)
        """
        model = model.to(self.device)
        model.eval()
        
        # 创建输入张量 (添加batch维度)
        inputs = {}
        for modality, shape in input_shape.items():
            full_shape = (batch_size,) + shape
            inputs[modality] = torch.randn(full_shape, device=self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        # 基准测试
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(inputs)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计量
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = (batch_size * 1000) / mean_latency  # samples/sec
        
        return mean_latency, std_latency, throughput
    
    def monitor_memory_usage(
        self,
        model: nn.Module,
        input_shape: Dict[str, Tuple[int, ...]],
        batch_size: int = 1
    ) -> Tuple[float, float]:
        """
        监控内存使用
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典
            batch_size: 批次大小
            
        Returns:
            (推理峰值内存MB, 训练峰值内存MB)
        """
        model = model.to(self.device)
        
        # 创建输入张量
        inputs = {}
        for modality, shape in input_shape.items():
            full_shape = (batch_size,) + shape
            inputs[modality] = torch.randn(full_shape, device=self.device, requires_grad=True)
        
        # 清理内存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 测量推理内存
        model.eval()
        with torch.no_grad():
            try:
                _ = model(inputs)
                if self.device.type == "cuda":
                    inference_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    # CPU内存监控 (近似)
                    process = psutil.Process()
                    inference_memory = process.memory_info().rss / (1024 * 1024)
            except Exception as e:
                logger.warning(f"推理内存测量失败 (batch_size={batch_size}): {e}")
                inference_memory = 0.0
        
        # 测量训练内存 (包含梯度) - 只在batch_size > 1时进行
        training_memory = 0.0
        if batch_size > 1:
            try:
                model.train()
                outputs = model(inputs)
                
                # 计算损失并反向传播
                if isinstance(outputs, dict):
                    loss = sum(output.sum() for output in outputs.values())
                else:
                    loss = outputs.sum()
                
                loss.backward()
                
                if self.device.type == "cuda":
                    training_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                else:
                    # CPU内存监控 (近似)
                    training_memory = process.memory_info().rss / (1024 * 1024)
            except Exception as e:
                logger.warning(f"训练内存测量失败 (batch_size={batch_size}): {e}")
                training_memory = inference_memory  # 使用推理内存作为近似
        else:
            # batch_size=1时，训练内存近似为推理内存的1.5倍
            training_memory = inference_memory * 1.5
        
        return inference_memory, training_memory
    
    def analyze_batch_size_scaling(
        self,
        model: nn.Module,
        input_shape: Dict[str, Tuple[int, ...]]
    ) -> List[BatchSizeAnalysis]:
        """
        分析不同批次大小的性能表现
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典 (不包含batch维度)
            
        Returns:
            批次大小分析结果列表
        """
        results = []
        
        for batch_size in self.batch_sizes:
            try:
                # 测量延迟
                latency, latency_std, throughput = self.measure_inference_latency(
                    model, input_shape, batch_size
                )
                
                # 测量内存
                inference_memory, _ = self.monitor_memory_usage(
                    model, input_shape, batch_size
                )
                
                # 计算内存效率
                memory_efficiency = batch_size / inference_memory if inference_memory > 0 else 0
                
                analysis = BatchSizeAnalysis(
                    batch_size=batch_size,
                    latency_ms=latency,
                    latency_std_ms=latency_std,
                    throughput_samples_per_sec=throughput,
                    memory_mb=inference_memory,
                    memory_efficiency=memory_efficiency
                )
                
                results.append(analysis)
                
                logger.info(f"批次大小 {batch_size}: 延迟={latency:.2f}ms, 吞吐量={throughput:.1f}samples/s")
                
            except Exception as e:
                logger.warning(f"批次大小 {batch_size} 分析失败: {e}")
                # 创建一个默认的分析结果，避免完全跳过
                analysis = BatchSizeAnalysis(
                    batch_size=batch_size,
                    latency_ms=0.0,
                    latency_std_ms=0.0,
                    throughput_samples_per_sec=0.0,
                    memory_mb=0.0,
                    memory_efficiency=0.0
                )
                results.append(analysis)
                continue
        
        return results
    
    def compute_efficiency_metrics(
        self,
        complexity_metrics: ComplexityMetrics,
        accuracy: float
    ) -> ComplexityMetrics:
        """
        计算效率指标
        
        Args:
            complexity_metrics: 复杂度指标
            accuracy: 模型准确率
            
        Returns:
            更新后的复杂度指标
        """
        # 计算每FLOP准确率
        if complexity_metrics.flops is not None and complexity_metrics.flops > 0:
            complexity_metrics.accuracy_per_flop = accuracy / complexity_metrics.flops
        
        # 计算每参数准确率
        if complexity_metrics.total_params > 0:
            complexity_metrics.accuracy_per_parameter = accuracy / complexity_metrics.total_params
        
        return complexity_metrics
    
    def analyze_model_complexity(
        self,
        model: nn.Module,
        input_shape: Dict[str, Tuple[int, ...]],
        accuracy: Optional[float] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        完整的模型复杂度分析
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状字典 (不包含batch维度)
            accuracy: 模型准确率 (可选)
            model_name: 模型名称
            
        Returns:
            完整的复杂度分析结果
        """
        logger.info(f"开始分析模型复杂度: {model_name}")
        
        results = {
            "model_name": model_name,
            "device": str(self.device),
            "analysis_config": {
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
                "batch_sizes": self.batch_sizes
            }
        }
        
        # 1. 参数统计
        logger.info("1. 统计模型参数...")
        total_params, trainable_params, model_size_mb = self.count_parameters(model)
        
        # 2. FLOPs计算
        logger.info("2. 计算FLOPs...")
        # 使用batch_size=1计算FLOPs
        flops_input_shape = {k: (1,) + v for k, v in input_shape.items()}
        flops, flops_table = self.compute_flops(model, flops_input_shape)
        
        # 3. 批次大小分析
        logger.info("3. 分析不同批次大小性能...")
        batch_analyses = self.analyze_batch_size_scaling(model, input_shape)
        
        # 4. 创建主要指标 (使用batch_size=1的结果)
        main_metrics = ComplexityMetrics(
            total_params=total_params,
            trainable_params=trainable_params,
            model_size_mb=model_size_mb,
            flops=flops,
            flops_g=flops / 1e9 if flops else None,
            batch_size=1
        )
        
        # 从batch_size=1的分析中获取延迟和内存数据
        if batch_analyses:
            batch_1_analysis = next((ba for ba in batch_analyses if ba.batch_size == 1), None)
            if batch_1_analysis:
                main_metrics.inference_latency_ms = batch_1_analysis.latency_ms
                main_metrics.inference_std_ms = batch_1_analysis.latency_std_ms
                main_metrics.throughput_samples_per_sec = batch_1_analysis.throughput_samples_per_sec
                main_metrics.peak_memory_mb = batch_1_analysis.memory_mb
        
        # 5. 计算效率指标
        if accuracy is not None:
            logger.info("4. 计算效率指标...")
            main_metrics = self.compute_efficiency_metrics(main_metrics, accuracy)
        
        # 6. 组装结果
        results.update({
            "main_metrics": asdict(main_metrics),
            "batch_size_analysis": [asdict(ba) for ba in batch_analyses],
            "flops_table": flops_table,
            "parameter_breakdown": self._get_parameter_breakdown(model)
        })
        
        logger.info(f"复杂度分析完成: {model_name}")
        return results
    
    def _get_parameter_breakdown(self, model: nn.Module) -> Dict[str, int]:
        """获取各模块的参数分解"""
        breakdown = {}
        
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            breakdown[name] = module_params
        
        return breakdown
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Dict[str, Tuple[int, ...]],
        accuracies: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        比较多个模型的复杂度
        
        Args:
            models: 模型字典 {name: model}
            input_shape: 输入形状字典
            accuracies: 准确率字典 {name: accuracy}
            
        Returns:
            模型比较结果
        """
        logger.info(f"开始比较 {len(models)} 个模型的复杂度")
        
        model_results = {}
        
        for model_name, model in models.items():
            accuracy = accuracies.get(model_name) if accuracies else None
            
            try:
                result = self.analyze_model_complexity(
                    model, input_shape, accuracy, model_name
                )
                model_results[model_name] = result
                
            except Exception as e:
                logger.error(f"模型 {model_name} 分析失败: {e}")
                continue
        
        # 生成比较表
        comparison_table = self._generate_comparison_table(model_results)
        
        return {
            "individual_results": model_results,
            "comparison_table": comparison_table,
            "summary": self._generate_comparison_summary(model_results)
        }
    
    def _generate_comparison_table(self, model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成模型比较表"""
        table = []
        
        for model_name, result in model_results.items():
            metrics = result["main_metrics"]
            
            row = {
                "model": model_name,
                "params_M": round(metrics["total_params"] / 1e6, 2),
                "flops_G": round(metrics["flops_g"], 2) if metrics["flops_g"] else None,
                "latency_ms": round(metrics["inference_latency_ms"], 2),
                "throughput": round(metrics["throughput_samples_per_sec"], 1),
                "memory_MB": round(metrics["peak_memory_mb"], 1),
                "model_size_MB": round(metrics["model_size_mb"], 1)
            }
            
            # 添加效率指标
            if metrics.get("accuracy_per_flop"):
                row["acc_per_flop"] = f"{metrics['accuracy_per_flop']:.2e}"
            if metrics.get("accuracy_per_parameter"):
                row["acc_per_param"] = f"{metrics['accuracy_per_parameter']:.2e}"
            
            table.append(row)
        
        return table
    
    def _generate_comparison_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成比较摘要"""
        if not model_results:
            return {}
        
        # 提取所有指标
        params = []
        flops = []
        latencies = []
        memories = []
        
        for result in model_results.values():
            metrics = result["main_metrics"]
            params.append(metrics["total_params"])
            if metrics["flops"]:
                flops.append(metrics["flops"])
            latencies.append(metrics["inference_latency_ms"])
            memories.append(metrics["peak_memory_mb"])
        
        summary = {
            "num_models": len(model_results),
            "parameter_range": {
                "min": min(params),
                "max": max(params),
                "ratio": max(params) / min(params) if min(params) > 0 else 0
            },
            "latency_range": {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "ratio": max(latencies) / min(latencies) if min(latencies) > 0 else 0
            },
            "memory_range": {
                "min_MB": min(memories),
                "max_MB": max(memories),
                "ratio": max(memories) / min(memories) if min(memories) > 0 else 0
            }
        }
        
        if flops:
            summary["flops_range"] = {
                "min": min(flops),
                "max": max(flops),
                "ratio": max(flops) / min(flops) if min(flops) > 0 else 0
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存分析结果到JSON文件
        
        Args:
            results: 分析结果
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理不可序列化的对象
        def json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializable)
        
        logger.info(f"分析结果已保存到: {output_path}")


def create_sample_input_shape() -> Dict[str, Tuple[int, ...]]:
    """创建示例输入形状 (用于测试)"""
    return {
        'eeg': (6, 3000),      # 6通道EEG，3000采样点
        'eog': (2, 3000),      # 2通道EOG，3000采样点  
        'emg': (3, 3000),      # 3通道EMG，3000采样点
        'respiratory': (4, 750), # 4通道呼吸，750采样点
        'cardiac': (3, 3000),   # 3通道心血管，3000采样点
        'spo2': (1, 30)        # 1通道SpO2，30采样点
    }


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("模型复杂度分析器测试")
    print("=" * 70)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建分析器
    analyzer = ModelComplexityAnalyzer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=5,
        benchmark_runs=20,
        batch_sizes=[1, 4, 8]
    )
    
    # 创建简单测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(6, 64, 3)
            self.conv2 = nn.Conv1d(64, 128, 3)
            self.fc = nn.Linear(128, 5)
            
        def forward(self, inputs):
            x = inputs['eeg']  # 只使用EEG
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.mean(x, dim=2)  # Global average pooling
            x = self.fc(x)
            return {'sleep_staging': x}
    
    model = SimpleModel()
    input_shape = {'eeg': (6, 100)}  # 简化的输入形状
    
    # 运行分析
    results = analyzer.analyze_model_complexity(
        model=model,
        input_shape=input_shape,
        accuracy=0.85,
        model_name="SimpleTestModel"
    )
    
    # 打印结果
    print("\n分析结果:")
    print(f"参数数量: {results['main_metrics']['total_params']:,}")
    print(f"模型大小: {results['main_metrics']['model_size_mb']:.2f} MB")
    print(f"推理延迟: {results['main_metrics']['inference_latency_ms']:.2f} ms")
    
    if results['main_metrics']['flops']:
        print(f"FLOPs: {results['main_metrics']['flops_g']:.2f} G")
    
    print("\n测试完成!")