"""
公开数据集适配器
为Sleep-EDF和MASS等公开数据集提供统一接口
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import pickle


class PublicDatasetAdapter(Dataset):
    """
    公开数据集适配器
    
    由于公开数据集没有OSA严重程度信息，我们：
    1. 使用虚拟的严重程度标签（全部设为normal）
    2. 使用虚拟的人口统计学信息（随机生成）
    3. 主要测试方法在非OSA人群上的泛化能力
    """
    
    def __init__(
        self,
        patient_ids,
        pkl_dir,
        demographics_generator=None,
        use_dummy_severity=True,
    ):
        """
        Args:
            patient_ids: 患者ID列表
            pkl_dir: PKL文件目录
            demographics_generator: 人口统计学生成器
            use_dummy_severity: 是否使用虚拟严重程度（公开数据集必须为True）
        """
        self.patient_ids = patient_ids
        self.pkl_dir = Path(pkl_dir)
        self.demographics_generator = demographics_generator
        self.use_dummy_severity = use_dummy_severity
        
        # 加载所有数据
        self.data = {}
        self.patient_epoch_map = {}
        
        total_epochs = 0
        for pid in patient_ids:
            pkl_file = self.pkl_dir / f"{pid}.pkl"
            if not pkl_file.exists():
                # 尝试不同的命名格式
                pkl_file = self.pkl_dir / f"{pid}E0.pkl"
            
            if pkl_file.exists():
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                self.data[pid] = data
                
                # 记录每个患者的epoch范围（兼容不同字段名）
                label_key = 'labels' if 'labels' in data else 'sleep_stages'
                n_epochs = len(data[label_key])
                self.patient_epoch_map[pid] = list(range(total_epochs, total_epochs + n_epochs))
                total_epochs += n_epochs
        
        self.total_epochs = total_epochs
        
        # 创建epoch到患者的映射
        self.epoch_to_patient = {}
        for pid, epoch_indices in self.patient_epoch_map.items():
            for global_idx, local_idx in enumerate(epoch_indices):
                self.epoch_to_patient[local_idx] = (pid, global_idx)
    
    def __len__(self):
        return self.total_epochs
    
    def __getitem__(self, idx):
        """
        返回: (signal, label, patient_features)
        """
        # 找到对应的患者和局部索引
        pid, local_idx = self.epoch_to_patient[idx]
        data = self.data[pid]
        
        # 获取信号和标签（兼容不同字段名）
        signal = data['signals'][local_idx]  # [T] or [C, T]
        label_key = 'labels' if 'labels' in data else 'sleep_stages'
        label = data[label_key][local_idx]
        
        # 转换为tensor，确保信号是 [C, T] 格式
        signal = torch.from_numpy(signal).float()
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # [T] -> [1, T]
        label = torch.tensor(label, dtype=torch.long)
        
        # 生成患者特征
        patient_features = self._generate_patient_features(pid)
        
        return signal, label, patient_features
    
    def _generate_patient_features(self, pid):
        """生成患者特征（公开数据集使用虚拟值）"""
        # 使用患者ID作为随机种子，确保同一患者的特征一致
        seed = hash(pid) % (2**32)
        rng = np.random.RandomState(seed)
        
        if self.use_dummy_severity:
            # 公开数据集：使用虚拟值
            # 假设都是健康人群（normal severity）
            ahi = rng.uniform(0, 4.9)  # Normal range
            severity = 0  # Normal
            age = rng.uniform(30, 60)
            sex = rng.randint(0, 2)
            bmi = rng.uniform(20, 28)
        else:
            # 如果有真实数据，使用真实值
            # （为未来扩展预留）
            raise NotImplementedError("Real severity data not available for public datasets")
        
        return {
            'ahi': torch.tensor(ahi, dtype=torch.float32),
            'severity': torch.tensor(severity, dtype=torch.long),
            'age': torch.tensor(age, dtype=torch.float32),
            'sex': torch.tensor(sex, dtype=torch.long),
            'bmi': torch.tensor(bmi, dtype=torch.float32),
        }
    
    def get_patient_epoch_indices(self, pid):
        """获取指定患者的所有epoch索引"""
        return self.patient_epoch_map.get(pid, [])


def get_sleep_edf_patient_ids(pkl_dir="data/sleep_edf"):
    """获取Sleep-EDF数据集的患者ID列表"""
    pkl_dir = Path(pkl_dir)
    patient_ids = []
    
    for pkl_file in sorted(pkl_dir.glob("*.pkl")):
        # Sleep-EDF格式: SC4001E0.pkl
        # 提取患者ID: SC4001E0
        pid = pkl_file.stem
        patient_ids.append(pid)
    
    return patient_ids


def split_sleep_edf_train_test(patient_ids, test_ratio=0.2, seed=42):
    """
    划分Sleep-EDF数据集为训练集和测试集
    
    模拟"域迁移"场景：
    - 训练集：源域（预训练模型已见过）
    - 测试集：目标域（新的临床中心，需要适应）
    """
    rng = np.random.RandomState(seed)
    
    # 随机打乱
    shuffled_ids = patient_ids.copy()
    rng.shuffle(shuffled_ids)
    
    # 划分
    n_test = int(len(shuffled_ids) * test_ratio)
    test_ids = shuffled_ids[:n_test]
    train_ids = shuffled_ids[n_test:]
    
    return train_ids, test_ids
