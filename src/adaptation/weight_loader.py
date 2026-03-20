"""
预训练权重加载模块。

加载策略（按优先级）：
1. 如果提供 checkpoint_path，直接加载
2. 如果模型是 PhysioEx 包装器，尝试从 HuggingFace Hub 加载预训练 checkpoint
3. 回退到随机初始化
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightLoader:
    """预训练权重加载器。"""

    @staticmethod
    def load_weights(
        model: nn.Module,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        fold: Optional[int] = None,
        pretrained_dir: str = "weights/pretrained",
    ) -> Dict[str, Any]:
        """尝试加载预训练权重。

        Args:
            model: 目标模型实例
            model_name: 模型名称（"Chambon2018" 或 "TinySleepNet"）
            checkpoint_path: 可选的 checkpoint 文件路径
            fold: 可选的 fold 编号，用于加载对应 fold 的预训练权重
            pretrained_dir: 预训练权重目录（默认 weights/pretrained）

        Returns:
            metadata dict: {loaded: bool, source: str, mismatched_keys: List[str]}
        """
        # 策略 1: 从 checkpoint 文件加载
        if checkpoint_path is not None:
            try:
                state_dict = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                mismatched = WeightLoader._load_state_dict_flexible(
                    model, state_dict
                )
                logger.info(
                    "从 checkpoint 加载权重: %s (不匹配键: %d)",
                    checkpoint_path, len(mismatched),
                )
                return {
                    "loaded": True,
                    "source": f"checkpoint:{checkpoint_path}",
                    "mismatched_keys": mismatched,
                }
            except Exception as e:
                logger.warning("从 checkpoint 加载失败: %s", e)

        # 策略 2: 从本地预训练 checkpoint 加载（我们自己训练的）
        if fold is not None:
            local_ckpt = Path(pretrained_dir) / f"{model_name}_fold{fold}_best.pt"
            if local_ckpt.exists():
                try:
                    ckpt = torch.load(str(local_ckpt), map_location="cpu", weights_only=True)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    mismatched = WeightLoader._load_state_dict_flexible(
                        model, state_dict
                    )
                    val_acc = ckpt.get("best_val_acc", "N/A")
                    logger.info(
                        "从本地预训练 checkpoint 加载权重: %s (val_acc=%.4f, 不匹配键: %d)",
                        local_ckpt, val_acc if isinstance(val_acc, float) else 0.0, len(mismatched),
                    )
                    return {
                        "loaded": True,
                        "source": f"local_pretrained:{local_ckpt}",
                        "mismatched_keys": mismatched,
                    }
                except Exception as e:
                    logger.warning("从本地预训练 checkpoint 加载失败: %s", e)

        # 策略 3: 从 PhysioEx / HuggingFace Hub 加载
        success, mismatched = WeightLoader._try_physioex_pretrained(
            model, model_name
        )
        if success:
            return {
                "loaded": True,
                "source": "physioex_pretrained",
                "mismatched_keys": mismatched,
            }

        # 策略 4: 回退到随机初始化
        logger.warning(
            "无法加载 %s 预训练权重，使用随机初始化", model_name
        )
        return {
            "loaded": False,
            "source": "random_init",
            "mismatched_keys": [],
        }

    @staticmethod
    def _try_physioex_pretrained(
        model: nn.Module, model_name: str
    ) -> tuple:
        """尝试从 PhysioEx 预训练 checkpoint 加载权重。

        PhysioEx 模型可通过 HuggingFace Hub 下载预训练权重。
        """
        try:
            import physioex  # noqa: F401
        except ImportError:
            logger.warning("PhysioEx 未安装，跳过预训练权重加载")
            return False, []

        # 检查模型是否是 PhysioEx 包装器
        from src.adaptation.model_builder import (
            PhysioExChambon2018Wrapper,
            PhysioExTinySleepNetWrapper,
        )

        is_pex_wrapper = isinstance(
            model, (PhysioExChambon2018Wrapper, PhysioExTinySleepNetWrapper)
        )

        if not is_pex_wrapper:
            logger.info("模型不是 PhysioEx 包装器，跳过 PhysioEx 权重加载")
            return False, []

        # 尝试通过 PhysioEx 的 SleepModule.load_from_checkpoint 加载
        pex_name_map = {
            "Chambon2018": "chambon2018",
            "TinySleepNet": "tinysleepnet",
        }
        pex_name = pex_name_map.get(model_name)
        if pex_name is None:
            return False, []

        try:
            # 尝试从 HuggingFace Hub 下载预训练 checkpoint
            ckpt_path = WeightLoader._download_physioex_checkpoint(pex_name)
            if ckpt_path is None:
                logger.info("未找到 PhysioEx %s 预训练 checkpoint", pex_name)
                return False, []

            # 加载 checkpoint 到内部的 _pex_net
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # PhysioEx checkpoint 通常是 Lightning checkpoint
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Lightning state_dict 的键通常有 "nn." 前缀
            # 我们需要映射到 _pex_net 的键
            pex_net = model._pex_net
            pex_sd = {}
            for k, v in state_dict.items():
                # 去掉 "nn." 前缀
                if k.startswith("nn."):
                    pex_sd[k[3:]] = v
                else:
                    pex_sd[k] = v

            mismatched = WeightLoader._load_state_dict_flexible(
                pex_net, pex_sd
            )
            logger.info(
                "从 PhysioEx 预训练 checkpoint 加载 %s 权重成功 (不匹配键: %d)",
                model_name, len(mismatched),
            )
            return True, mismatched

        except Exception as e:
            logger.warning("从 PhysioEx 加载 %s 预训练权重失败: %s", model_name, e)
            return False, []

    @staticmethod
    def _download_physioex_checkpoint(model_name: str) -> Optional[str]:
        """从本地 weights 目录查找 PhysioEx 预训练 checkpoint。

        Args:
            model_name: PhysioEx 模型名 ("chambon2018" 或 "tinysleepnet")

        Returns:
            checkpoint 文件路径，或 None
        """
        # 检查本地 weights 目录
        local_candidates = [
            Path("weights") / f"{model_name}_pretrained.ckpt",
            Path("weights") / f"{model_name}_best.ckpt",
            Path("weights") / f"physioex_{model_name}.ckpt",
            Path("weights") / "physioex_cache" / f"{model_name}" / "best.ckpt",
        ]
        for p in local_candidates:
            if p.exists():
                logger.info("找到本地 PhysioEx checkpoint: %s", p)
                return str(p)

        # 尝试从 HuggingFace Hub 下载（可能因网络问题失败）
        try:
            from huggingface_hub import hf_hub_download
            repo_id = "Jathurshan/physioex"
            possible_paths = [
                f"{model_name}/best.ckpt",
                f"{model_name}/last.ckpt",
            ]
            for fpath in possible_paths:
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=fpath,
                        cache_dir=str(Path("weights") / "physioex_cache"),
                        timeout=10,
                    )
                    logger.info("下载 PhysioEx checkpoint: %s", local_path)
                    return local_path
                except Exception:
                    continue
        except ImportError:
            pass

        return None

    @staticmethod
    def _load_state_dict_flexible(
        model: nn.Module, state_dict: Dict[str, torch.Tensor]
    ) -> List[str]:
        """灵活加载 state_dict，跳过不匹配的键。"""
        result = model.load_state_dict(state_dict, strict=False)

        mismatched_keys: List[str] = []
        if result.missing_keys:
            logger.info("缺失键 (%d): %s", len(result.missing_keys), result.missing_keys[:5])
            mismatched_keys.extend(f"missing:{k}" for k in result.missing_keys)
        if result.unexpected_keys:
            logger.info("多余键 (%d): %s", len(result.unexpected_keys), result.unexpected_keys[:5])
            mismatched_keys.extend(f"unexpected:{k}" for k in result.unexpected_keys)

        return mismatched_keys
