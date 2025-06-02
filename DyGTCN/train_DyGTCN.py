import logging
import time
import sys
import os
import re
from pickle import FALSE

from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import shutil
import json
import torch
import torch.nn as nn

# 导入DyGTCN模型和其他必要的模块
from models.DyGTCN import DyGTCN
from models.modules import TCNClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_idx_data_loader, get_DyGTCN_data
from utils.load_configs import get_DyGTCN_args
import matplotlib.pyplot as plt


# 加载maintenance.csv
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

# 定义预测结果修正函数
def apply_maintenance_mask(predictions, timestamps):
    days = timestamps.astype(int)
    # 创建预测结果的副本
    masked_predictions = predictions.clone()
    for i in range(masked_predictions.size(0)):
        current_day = days[i]
        if current_day in maintenance_dict:
            for unit in maintenance_dict[current_day]:
                if 1 <= unit <= 54:
                    # 在副本上修改，保持原始计算图完整
                    masked_predictions[i, unit-1, :] = 0
    return masked_predictions

def calculate_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """计算分类任务评估指标"""
    predicted_labels = (predictions > 0.5).float()

    # 计算基础指标
    correct = (predicted_labels == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total

    # 计算混淆矩阵
    true_pos = ((predicted_labels == 1) & (labels == 1)).sum().item()
    false_pos = ((predicted_labels == 1) & (labels == 0)).sum().item()
    false_neg = ((predicted_labels == 0) & (labels == 1)).sum().item()

    # 处理除零情况
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
    args = get_DyGTCN_args(is_evaluation=False)

    # 获取训练、验证和测试数据
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _ = \
        get_DyGTCN_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # 在train_node_test.py的数据加载后添加严格时序检查（第45行附近）
    print(f"训练集时间范围: {train_data.node_interact_times.min():.1f}~{train_data.node_interact_times.max():.1f}")
    print(f"验证集时间范围: {val_data.node_interact_times.min():.1f}~{val_data.node_interact_times.max():.1f}")
    assert train_data.node_interact_times.max() < val_data.node_interact_times.min(), "数据泄露！"

    # 初始化邻居采样器
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                 sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # 数据加载器
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=True)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)

    test_metric_all_runs = []
    best_accuracy = -1.0
    best_predictions = None

    for run in range(args.num_runs):
        val_loss_history = []  # 每个epoch的验证损失记录
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # 日志配置
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        # 创建模型
        dynamic_backbone = DyGTCN(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            channel_embedding_dim=args.channel_embedding_dim,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_input_sequence_length=args.max_input_sequence_length,
            device=args.device
        )

        TCN_Classifier = TCNClassifier(
            src_emb_dim=dynamic_backbone.node_feat_dim,
            dst_emb_dim=dynamic_backbone.node_feat_dim,
            load_feat_dim=node_raw_features.shape[1],
            num_units=54,
            time_steps=24
        )
        model = nn.Sequential(dynamic_backbone, TCN_Classifier)

        # 日志输出模型信息
        logger.info(f'model -> {model}')
        logger.info(f'#parameters: {get_parameter_sizes(model) * 4 / 1024 / 1024:.2f} MB')

        optimizer = create_optimizer(model=model,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',  # 监控验证损失
        #     factor=0.5,  # 学习率衰减因子
        #     patience=5,  # 5个epoch无改善后衰减
        #     verbose=True,
        #     min_lr=1e-6  # 最小学习率
        # )
        # 改用余弦退火调度
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=10,  # 每10个epoch重启
        #     T_mult=2,
        #     eta_min=1e-6
        # )

        model = convert_to_gpu(model, device=args.device)

        # 模型保存配置
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # 使用二分类交叉熵损失
        bce_loss_func = nn.BCELoss()

        for epoch in range(args.num_epochs):
            model.train()
            dynamic_backbone.set_neighbor_sampler(train_neighbor_sampler)

            train_losses = []
            train_metrics = []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids = train_data.src_node_ids[train_data_indices]
                batch_dst_node_ids = train_data.dst_node_ids[train_data_indices]
                batch_node_interact_times = train_data.node_interact_times[train_data_indices]
                batch_labels = torch.FloatTensor(train_data.labels[train_data_indices]).view(-1, 54, 24).to(args.device)
                batch_src_load = torch.from_numpy(node_raw_features[batch_src_node_ids]).float().to(args.device)

                # 获取时空嵌入
                src_emb, dst_emb = dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times
                )

                # 启停预测
                predictions = TCN_Classifier(src_emb, dst_emb, batch_src_load)

                #检修机组修正
                predictions = apply_maintenance_mask(predictions, batch_node_interact_times)

                # 计算损失
                loss = bce_loss_func(predictions, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # 计算指标
                metrics = calculate_classification_metrics(predictions, batch_labels)

                # 记录数据
                train_losses.append(loss.item())
                train_metrics.append(metrics)

                # 进度条更新
                avg_loss = np.mean(train_losses)
                avg_acc = np.mean([m['accuracy'] for m in train_metrics])
                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')

            # 验证流程
            model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for val_indices in val_idx_data_loader:
                    val_indices = val_indices.numpy()
                    v_src = val_data.src_node_ids[val_indices]
                    v_dst = val_data.dst_node_ids[val_indices]
                    v_time = val_data.node_interact_times[val_indices]
                    v_labels = torch.FloatTensor(val_data.labels[val_indices]).view(-1, 54, 24).to(args.device)
                    v_src_load = torch.from_numpy(node_raw_features[v_src]).float().to(args.device)

                    v_src_emb, v_dst_emb = dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=v_src,
                        dst_node_ids=v_dst,
                        node_interact_times=v_time
                    )

                    val_pred = TCN_Classifier(v_src_emb, v_dst_emb, v_src_load)
                    val_pred = apply_maintenance_mask(val_pred, v_time)
                    val_loss = bce_loss_func(val_pred, v_labels)

                    val_losses.append(val_loss.item())
                    val_metrics.append(calculate_classification_metrics(val_pred, v_labels))

            avg_val_f1 = np.mean([m['f1'] for m in val_metrics])
            avg_val_loss = np.mean(val_losses)  # 确保已计算平均验证损失
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean([m['accuracy'] for m in train_metrics])
            avg_val_acc = np.mean([m['accuracy'] for m in val_metrics])
            #scheduler.step(avg_val_loss)
            #scheduler.step()


            # 记录日志
            val_loss_history.append(avg_val_loss)

            logger.info(f'Epoch {epoch + 1}:')
            logger.info(f'Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}')
            logger.info(f'Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}')
            logger.info(f'Val Precision: {np.mean([m["precision"] for m in val_metrics]):.4f}')
            logger.info(f'Val Recall: {np.mean([m["recall"] for m in val_metrics]):.4f}')
            logger.info(f'Val F1: {avg_val_f1:.4f}')

        # 最终测试
        test_losses = []
        test_metrics = []
        total_inference_time = 0.0
        max_inference_time = 0.0

        all_predictions = []
        all_timestamps = []

        test_preds = []  # 原有变量可能需要调整
        test_timestamps = []

        model.eval()
        with torch.no_grad():
            for test_indices in test_idx_data_loader:
                test_indices = test_indices.numpy()
                t_src = test_data.src_node_ids[test_indices]
                t_dst = test_data.dst_node_ids[test_indices]
                t_time = test_data.node_interact_times[test_indices]
                batch_times = test_data.node_interact_times[test_indices]
                t_labels = torch.FloatTensor(test_data.labels[test_indices]).view(-1, 54, 24).to(args.device)
                t_src_load = torch.from_numpy(node_raw_features[t_src]).float().to(args.device)

                # 时间测量开始（新增）
                start_time = time.time()
                if args.device == 'cuda':
                    torch.cuda.synchronize()  # 确保CUDA操作同步

                t_src_emb, t_dst_emb = dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=t_src,
                    dst_node_ids=t_dst,
                    node_interact_times=t_time
                )

                test_pred = TCN_Classifier(t_src_emb, t_dst_emb, t_src_load)
                if args.device == 'cuda':
                    torch.cuda.synchronize()  # 确保CUDA操作同步
                end_time = time.time()
                # 时间测量结束
                batch_time = end_time - start_time
                total_inference_time += batch_time  # 累计时间
                max_inference_time = max(max_inference_time, batch_time)

                test_pred = apply_maintenance_mask(test_pred, t_time)
                test_loss = bce_loss_func(test_pred, t_labels)

                # 收集原始预测结果和时间戳
                all_predictions.append(test_pred.cpu().detach())
                all_timestamps.append(torch.from_numpy(batch_times))

                test_preds.append(test_pred.cpu())
                test_timestamps.append(torch.from_numpy(batch_times))

                test_losses.append(test_loss.item())
                test_metrics.append(calculate_classification_metrics(test_pred, t_labels))

        # 合并所有预测结果
        all_preds = torch.cat(all_predictions, dim=0)  # shape: (N, 54, 24)
        all_times = torch.cat(all_timestamps, dim=0)

        # 按时间排序
        sorted_indices = torch.argsort(all_times)
        sorted_preds = all_preds[sorted_indices]
        sorted_times = all_times[sorted_indices]  # 时间戳按时间排序

        # 将时间戳转换为整数天（310.0 -> 310）
        days = sorted_times.int()  # 转换为整数天
        unique_days = torch.unique(days)  # 唯一的天数（310到365之间的不同天数）

        # 初始化存储数组 (56天 × 24小时 × 54机组)
        daily_predictions = np.zeros((56 * 24, 54))

        # 遍历每一天
        for day_idx, day in enumerate(unique_days):
            # 获取当天的所有样本（可能存在多个样本）
            day_mask = (days == day)
            day_samples = sorted_preds[day_mask]  # shape: (N_samples, 54, 24)

            # 取最后一个样本的预测结果（最新预测最准确）
            last_sample_pred = day_samples[-1]  # shape: (54, 24)

            # 转置为 (24, 54)
            hourly_pred = last_sample_pred.permute(1, 0)

            # 填充到结果数组
            start_idx = day_idx * 24
            end_idx = (day_idx + 1) * 24
            daily_predictions[start_idx:end_idx] = hourly_pred.cpu().numpy()

        # 在保存CSV前进行二值化
        daily_predictions_binary = (daily_predictions >= 0.5).astype(int)

        # 保存结果
        test_metric = {
            'loss': np.mean(test_losses),
            'accuracy': np.mean([m['accuracy'] for m in test_metrics]),
            'precision': np.mean([m['precision'] for m in test_metrics]),
            'recall': np.mean([m['recall'] for m in test_metrics]),
            'f1': np.mean([m['f1'] for m in test_metrics])
        }
        test_metric_all_runs.append(test_metric)
        # 当前运行的准确率
        current_accuracy = test_metric['accuracy']
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_predictions = daily_predictions_binary.copy()

            # 保存最佳预测结果（立即保存）
            df_best = pd.DataFrame(best_predictions,
                                   columns=[f"Unit_{i + 1}" for i in range(54)])
            df_best.to_csv(f"./saved_results/{args.model_name}/{args.dataset_name}/all_predict_value.csv",
                           float_format="%d")
            logger.info(f'New best accuracy {current_accuracy:.4f}, predictions saved.')

        # 计算时间指标（在测试循环结束后添加）
        # 假设每个样本代表1小时，24小时为一天
        num_days = len(test_data.node_interact_times) / 24  # 根据实际数据结构调整
        avg_time_per_day = total_inference_time / num_days

        # 日志输出
        logger.info(f'Total Inference Time: {total_inference_time:.4f} seconds')
        logger.info(f'Average Time Per Day: {avg_time_per_day:.4f} seconds')
        logger.info(f'Max Inference Time (Worst Case): {max_inference_time:.4f} seconds')
        logger.info(f'Test Results:')
        logger.info(f'Loss: {test_metric["loss"]:.4f} | Acc: {test_metric["accuracy"]:.4f}')
        logger.info(f'Precision: {test_metric["precision"]:.4f} | Recall: {test_metric["recall"]:.4f}')
        logger.info(f'F1-score: {test_metric["f1"]:.4f}')
        logger.info(f'Run {run + 1} cost {time.time() - run_start_time:.1f}s\n')

        # 清理日志句柄
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

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
        logger.info(
            f'Average Test {field.capitalize()}: {final_results[field]["mean"]:.4f} ± {final_results[field]["std"]:.4f}')

    # 保存汇总结果
    result_json = {
        "summary": final_results,
        "all_runs": [
            {
                "run": i + 1,
                **test_metric
            } for i, test_metric in enumerate(test_metric_all_runs)
        ]
    }
    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    with open(os.path.join(save_result_folder, "status_prediction_summary.json"), 'w') as f:
        json.dump(result_json, f, indent=4)

    sys.exit()
