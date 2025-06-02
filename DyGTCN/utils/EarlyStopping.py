import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger,
                 model_name: str = None):
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder,
                                                                   f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module):
        """
        修改后的核心逻辑：
        1. 任一指标提升即保存模型
        2. 只有当所有指标都不提升时才增加counter
        """
        has_improvement = False

        # 遍历所有指标
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            # 判断是否提升
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = metric_value
                has_improvement = True
                continue

            if higher_better:
                if metric_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
                    has_improvement = True
            else:
                if metric_value < self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
                    has_improvement = True

        # 存在提升指标
        if has_improvement:
            self.save_checkpoint(model)
            self.counter = 0
            self.logger.debug(f"Metric improved, reset counter to 0. Best metrics: {self.best_metrics}")
        # 所有指标均未提升
        else:
            self.counter += 1
            self.logger.debug(f"No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)


    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))

