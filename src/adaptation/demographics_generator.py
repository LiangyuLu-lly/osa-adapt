"""基于 AHI/severity 生成确定性合成人口学数据。

用于在真实 PSG 数据缺少 age/sex/bmi 字段时，
根据患者的 AHI 和 OSA 严重程度生成临床合理的合成值。
"""

import random
from typing import Dict


class DemographicsGenerator:
    """基于 AHI/severity 生成确定性合成人口学数据。

    使用 patient_id 作为随机种子，确保跨运行可复现。
    """

    # 临床合理的分布参数: severity -> (mean, std)
    AGE_PARAMS = {
        0: (45.0, 12.0),  # normal: mean=45, std=12
        1: (50.0, 11.0),  # mild
        2: (55.0, 10.0),  # moderate
        3: (58.0, 9.0),   # severe
    }
    BMI_PARAMS = {
        0: (25.0, 4.0),   # normal
        1: (28.0, 4.5),   # mild
        2: (31.0, 5.0),   # moderate
        3: (34.0, 5.5),   # severe
    }
    MALE_RATIO = {
        0: 0.55,  # normal
        1: 0.60,  # mild
        2: 0.65,  # moderate
        3: 0.70,  # severe: ~2:1 male-to-female
    }

    def generate(self, patient_id: str, ahi: float, severity: int) -> Dict[str, float]:
        """生成确定性合成人口学数据。

        Args:
            patient_id: 患者 ID，用作随机种子
            ahi: Apnea-Hypopnea Index
            severity: OSA 严重程度等级 (0=normal, 1=mild, 2=moderate, 3=severe)

        Returns:
            包含 age, sex, bmi 的字典
        """
        rng = random.Random(hash(patient_id))

        # age: 从正态分布采样，clamp 到 [18, 90]
        age_mean, age_std = self.AGE_PARAMS[severity]
        age = rng.gauss(age_mean, age_std)
        age = max(18.0, min(90.0, age))

        # sex: 按 severity 对应的 male_ratio 采样 Bernoulli (1=M, 0=F)
        male_ratio = self.MALE_RATIO[severity]
        sex = 1 if rng.random() < male_ratio else 0

        # bmi: 从正态分布采样，clamp 到 [15, 60]
        bmi_mean, bmi_std = self.BMI_PARAMS[severity]
        bmi = rng.gauss(bmi_mean, bmi_std)
        bmi = max(15.0, min(60.0, bmi))

        return {"age": age, "sex": sex, "bmi": bmi}
