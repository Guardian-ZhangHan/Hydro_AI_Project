# ==============================================
# 科研级早停机制实现（水文顶刊规范版）
# 功能：可配置耐心轮次、最小改善阈值、自动保存最优权重、恢复最优权重
# 适配PyTorch训练框架，全流程可追溯
# ==============================================
import numpy as np
import torch
import logging
from datetime import datetime

class EarlyStopping:
    def __init__(
        self,
        patience: int = 150,
        min_delta: float = 1e-6,
        mode: str = "min",
        restore_best_weights: bool = True,
        save_best_model: bool = True,
        save_path: str = "./best_model.pth",
        logger: logging.Logger = None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_model = save_best_model
        self.save_path = save_path
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # 内部状态变量
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop_triggered = False
        self.metric_history = []
        self.lr_history = []  # 新增：记录学习率变化，用于画图

        # 模式校验
        if mode not in ["min", "max"]:
            raise ValueError(f"模式mode只能是'min'或'max'，当前输入：{mode}")

    def __call__(self, current_metric: float, model: torch.nn.Module, epoch: int, current_lr: float = None) -> bool:
        """
        每轮训练结束后调用，判断是否触发早停
        :param current_metric: 当前轮次的监控指标
        :param model: 当前训练的模型
        :param epoch: 当前轮次
        :param current_lr: 当前学习率，用于记录
        :return: True：触发早停；False：继续训练
        """
        self.metric_history.append(current_metric)
        if current_lr is not None:
            self.lr_history.append(current_lr)

        # 第一轮初始化
        if self.best_score is None:
            self.best_score = current_metric
            self.best_epoch = epoch
            self._save_best_model(model, epoch)
            return False

        # 计算当前指标是否有改善
        if self.mode == "min":
            is_improved = current_metric < self.best_score - self.min_delta
        else:
            is_improved = current_metric > self.best_score + self.min_delta

        # 情况1：指标有改善
        if is_improved:
            self.best_score = current_metric
            self.best_epoch = epoch
            self.counter = 0
            self._save_best_model(model, epoch)
            self.logger.info(f"【早停】新最优指标：{current_metric:.6f}，最优轮次：{epoch}，已保存最优权重")
            return False

        # 情况2：指标无改善
        else:
            self.counter += 1
            self.logger.info(f"【早停】指标无改善，连续{self.counter}/{self.patience}轮，当前最优轮次：{self.best_epoch}，最优指标：{self.best_score:.6f}")
            
            # 触发早停
            if self.counter >= self.patience:
                self.early_stop_triggered = True
                self.logger.info(f"【早停】已连续{self.patience}轮指标无改善，触发早停，终止训练")
                # 恢复最优权重
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                    self.logger.info(f"【早停】已自动恢复最优轮次（epoch={self.best_epoch}）的模型权重")
                return True
            
            return False

    def _save_best_model(self, model: torch.nn.Module, epoch: int):
        """保存最优模型权重"""
        self.best_model_state = model.state_dict().copy()
        if self.save_best_model:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.best_model_state,
                "best_metric": self.best_score,
                "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, self.save_path)

    def get_summary(self) -> dict:
        """获取早停过程总结，用于论文归档"""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_epoch": self.best_epoch,
            "best_score": float(self.best_score) if self.best_score is not None else None,
            "early_stop_triggered": self.early_stop_triggered,
            "total_epochs_run": len(self.metric_history),
            "metric_history": self.metric_history,
            "lr_history": self.lr_history
        }