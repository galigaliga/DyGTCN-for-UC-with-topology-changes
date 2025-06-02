import logging
import time
import sys
import os
import warnings
import json
import re

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入模块
from models.GCN import GCN
from models.TCN import TCN
from models.LSTM import LSTM
from models.STGCN import STGCN
from utils.DataLoader import get_other_data, get_node_classification_data_loader
from utils.load_configs import get_other_args
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer


# 加载 maintenance.csv
maintenance_df = pd.read_csv('./DG_data/ieee118/maintenance.csv')
maintenance_dict = {}
for day_col in maintenance_df.columns:
    # 使用正则表达式提取天数数字部分
    day_match = re.search(r'\d+', day_col)
    if day_match:
        day = int(day_match.group())
        units = [int(u) for u in maintenance_df[day_col] if u != 0 and not pd.isna(u)]
        maintenance_dict[day] = units
    else:
        warnings.warn(f"Invalid column name format: {day_col}")


def apply_maintenance_mask(predictions: torch.Tensor, timestamps: np.ndarray) -> torch.Tensor:
    """根据维护计划修正预测结果"""
    days = timestamps.astype(int)
    masked_predictions = predictions.clone()

    for i in range(masked_predictions.size(0)):  # 遍历每个样本
        current_day = days[i]
        if current_day in maintenance_dict:
            for unit in maintenance_dict[current_day]:  # 遍历维护机组
                if 1 <= unit <= 54:  # 确保机组编号有效
                    masked_predictions[i, unit - 1, :] = 0  # 设置为0表示停机
    return masked_predictions


def calculate_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """计算分类任务评估指标"""
    predicted_labels = (predictions > 0.5).float()

    correct = (predicted_labels == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total

    true_pos = ((predicted_labels == 1) & (labels == 1)).sum().item()
    false_pos = ((predicted_labels == 1) & (labels == 0)).sum().item()
    false_neg = ((predicted_labels == 0) & (labels == 1)).sum().item()

    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # 获取参数
    args = get_other_args(is_evaluation=False)

    # 数据加载
    train_node, val_node, test_node, edge_index, train_labels, val_labels, test_labels = \
        get_other_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    train_features = torch.FloatTensor(train_node).to(args.device)  # shape: [T, F, N]
    val_features = torch.FloatTensor(val_node).to(args.device)
    test_features = torch.FloatTensor(test_node).to(args.device)

    train_labels = torch.FloatTensor(train_labels).to(args.device)  # shape: [T, C, N]
    val_labels = torch.FloatTensor(val_labels).to(args.device)
    test_labels = torch.FloatTensor(test_labels).to(args.device)

    edge_index = torch.LongTensor(edge_index).to(args.device)

    # 构建 DataLoader
    train_loader = get_node_classification_data_loader(
        features=train_features.cpu().numpy(),
        labels=train_labels.cpu().numpy(),
        days=np.arange(train_features.size(0)),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = get_node_classification_data_loader(
        features=val_features.cpu().numpy(),
        labels=val_labels.cpu().numpy(),
        days=np.arange(val_features.size(0)),
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = get_node_classification_data_loader(
        features=test_features.cpu().numpy(),
        labels=test_labels.cpu().numpy(),
        days=np.arange(test_features.size(0)),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 日志配置
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/", exist_ok=True)
    fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 模型初始化
    def initialize_model():
        if args.model_name == 'GCN':
            model = GCN(
                num_nodes=118,
                num_features=train_features.size(1),
                hidden_dim=args.gcn_hidden_dim,
                dropout=args.dropout
            )
        elif args.model_name == 'TCN':
            model = TCN(
                num_nodes=118,
                num_features=train_features.size(1),
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            )
        elif args.model_name == 'LSTM':
            model = LSTM(
                num_nodes=118,
                seq_len=train_features.size(1),  # T (总时间步数)
                hidden_dim=args.hidden_dim,
                num_units=54,                    # 固定为54台机组
                output_steps=24,                 # 输出未来24小时
            )
        elif args.model_name == 'STGCN':
            model = STGCN(
                num_nodes=118,
                num_features=train_features.size(1),  # 输入特征维度
                hidden_dims=args.hidden_dim,  # 隐藏层维度
                output_steps=24,  # 输出时间步
                dropout=args.dropout,  # Dropout率
                edge_index=edge_index
            )
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")
        return model

    def weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # 初始化模型
    model = initialize_model()
    model.apply(weights_init)
    model = convert_to_gpu(model, device=args.device)

    # 日志输出模型信息
    logger.info(f'model -> {model}')
    logger.info(f'#parameters: {get_parameter_sizes(model) * 4 / 1024 / 1024:.2f} MB')

    # 优化器和损失函数
    optimizer = create_optimizer(model=model,
                                 optimizer_name=args.optimizer,
                                 learning_rate=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # 模型保存配置
    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/"
    os.makedirs(save_model_folder, exist_ok=True)

    # 损失函数
    bce_loss_func = nn.BCELoss()

    test_metric_all_runs = []

    # 主循环：多轮运行
    for run in range(args.num_runs):
        set_random_seed(seed=run + args.seed)
        args.save_model_name = f'{args.model_name}_seed{run + args.seed}'

        # 创建新的日志文件
        run_logger = logging.getLogger(f'run_{run}')
        run_logger.setLevel(logging.DEBUG)
        run_fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}_{str(time.time())}.log")
        run_fh.setLevel(logging.DEBUG)
        run_ch = logging.StreamHandler()
        run_ch.setLevel(logging.WARNING)
        run_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        run_fh.setFormatter(run_formatter)
        run_ch.setFormatter(run_formatter)
        run_logger.addHandler(run_fh)
        run_logger.addHandler(run_ch)

        run_start_time = time.time()
        run_logger.info(f"********** Run {run + 1} starts. **********")

        # 重新初始化模型参数
        model = initialize_model().to(args.device)
        model.apply(weights_init)
        model = convert_to_gpu(model, device=args.device)

        # 重新初始化优化器
        optimizer = create_optimizer(model=model,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        # 训练循环
        for epoch in range(args.num_epochs):
            model.train()
            train_losses = []
            train_metrics = []

            for x_batch, y_batch, day_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", ncols=120):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                if args.model_name == 'GCN':
                    predictions = model(x_batch, edge_index)
                elif args.model_name == 'STGCN':
                    # 将输入转换为 (batch, seq_len, nodes, features)
                    # 假设原始数据是 [batch, features, nodes]
                    # 需要构造时间序列输入（这里需要确认数据加载逻辑）
                    # 这里假设x_batch是 [batch, features, nodes]
                    predictions = model(x_batch)
                else:  # LSTM 或 TCN
                    predictions = model(x_batch)

                predictions = torch.sigmoid(predictions)
                predictions = apply_maintenance_mask(predictions, day_batch.numpy())

                loss = bce_loss_func(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                metrics = calculate_classification_metrics(predictions, y_batch)
                train_losses.append(loss.item())
                train_metrics.append(metrics)

            # 验证阶段
            model.eval()
            val_losses = []
            val_metrics = []

            with torch.no_grad():
                for x_batch, y_batch, day_batch in val_loader:
                    x_batch = x_batch.to(args.device)
                    y_batch = y_batch.to(args.device)

                    if args.model_name == 'GCN':
                        predictions = model(x_batch, edge_index)
                    else:
                        predictions = model(x_batch)

                    predictions = torch.sigmoid(predictions)
                    predictions = apply_maintenance_mask(predictions, day_batch.numpy())

                    loss = bce_loss_func(predictions, y_batch)

                    metrics = calculate_classification_metrics(predictions, y_batch)
                    val_losses.append(loss.item())
                    val_metrics.append(metrics)

            avg_val_loss = np.mean(val_losses)

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean([m['accuracy'] for m in train_metrics])
            avg_val_acc = np.mean([m['accuracy'] for m in val_metrics])
            avg_val_f1 = np.mean([m['f1'] for m in val_metrics])

            run_logger.info(f'Epoch {epoch + 1}:')
            run_logger.info(f'Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}')
            run_logger.info(f'Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}')
            run_logger.info(f'Val Precision: {np.mean([m["precision"] for m in val_metrics]):.4f}')
            run_logger.info(f'Val Recall: {np.mean([m["recall"] for m in val_metrics]):.4f}')
            run_logger.info(f'Val F1: {avg_val_f1:.4f}')

        # 测试阶段
        model.eval()
        test_losses = []
        test_metrics = []

        with torch.no_grad():
            for x_batch, y_batch, day_batch in test_loader:
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                if args.model_name == 'GCN':
                    predictions = model(x_batch, edge_index)
                else:
                    predictions = model(x_batch)

                predictions = torch.sigmoid(predictions)
                predictions = apply_maintenance_mask(predictions, day_batch.numpy())

                loss = bce_loss_func(predictions, y_batch)

                metrics = calculate_classification_metrics(predictions, y_batch)
                test_losses.append(loss.item())
                test_metrics.append(metrics)

        test_metric = {
            'loss': np.mean(test_losses),
            'accuracy': np.mean([m['accuracy'] for m in test_metrics]),
            'precision': np.mean([m['precision'] for m in test_metrics]),
            'recall': np.mean([m['recall'] for m in test_metrics]),
            'f1': np.mean([m['f1'] for m in test_metrics])
        }
        test_metric_all_runs.append(test_metric)

        run_logger.info(f'Test Results:')
        run_logger.info(f'Loss: {test_metric["loss"]:.4f} | Acc: {test_metric["accuracy"]:.4f}')
        run_logger.info(f'Precision: {test_metric["precision"]:.4f} | Recall: {test_metric["recall"]:.4f}')
        run_logger.info(f'F1-score: {test_metric["f1"]:.4f}')
        run_logger.info(f'Run {run + 1} cost {time.time() - run_start_time:.1f}s\n')

        run_logger.removeHandler(run_fh)
        run_logger.removeHandler(run_ch)

    # 汇总最终结果
    logger.info(f'Final Results after {args.num_runs} runs:')
    result_fields = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    final_results = {
        field: {
            'mean': np.mean([x[field] for x in test_metric_all_runs]),
            'std': np.std([x[field] for x in test_metric_all_runs])
        }
        for field in result_fields
    }

    for field in result_fields:
        logger.info(f'Average Test {field.capitalize()}: {final_results[field]["mean"]:.4f} ± {final_results[field]["std"]:.4f}')

    # 保存汇总结果
    result_json = {
        "summary": final_results,
        "all_runs": [
            {"run": i + 1, **test_metric}
            for i, test_metric in enumerate(test_metric_all_runs)
        ]
    }
    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    with open(os.path.join(save_result_folder, "status_prediction_summary.json"), 'w') as f:
        json.dump(result_json, f, indent=4)

    sys.exit()
