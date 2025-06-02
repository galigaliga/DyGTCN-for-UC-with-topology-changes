from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        super().__init__()
        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)

def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    dataset = CustomizedDataset(indices_list=indices_list)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

class NodeClassificationDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, days: np.ndarray = None):
        """
        Args:
            features: shape (num_days, num_features, num_nodes)
            labels: shape (num_days, num_classes, num_nodes)
            days: optional, array of day indices
        """
        self.features = features
        self.labels = labels
        if days is None:
            self.days = np.arange(features.shape[0])
        else:
            self.days = days

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        day = self.days[idx]
        x = self.features[day]  # shape: [F, N]
        y = self.labels[day]    # shape: [C, N]
        return x, y, day


def get_node_classification_data_loader(features: np.ndarray,
                                        labels: np.ndarray,
                                        days: np.ndarray = None,
                                        batch_size: int = 8,
                                        shuffle: bool = False):
    dataset = NodeClassificationDataset(features=features, labels=labels, days=days)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

class Data:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                 node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

def get_DyGTCN_data(dataset_name: str, val_ratio: float, test_ratio: float):
    # 加载基础数据
    graph_df = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv')
    edge_raw_features = np.load(f'./processed_data/{dataset_name}/ml_{dataset_name}.npy')

    # 验证时间戳范围
    assert graph_df.ts.min() >= 1.0 and graph_df.ts.max() <= 365.0, "时间戳范围应为1.0-365.0"
    assert np.all(np.diff(graph_df.ts.values) >= 0), "时间戳未按升序排列"

    # 处理节点特征
    node_features_df = pd.read_csv(f'./DG_data/{dataset_name}/all_busloads.csv')
    node_raw_features = node_features_df.values.astype(np.float32)
    hourly_features = node_raw_features.reshape(118, 24, 365)
    node_hourly_features = hourly_features.transpose(0, 2, 1)
    node_raw_features = node_hourly_features.reshape(118, -1)
    node_raw_features = np.vstack([np.zeros((1, node_raw_features.shape[1])), node_raw_features])

    # 处理边特征
    EDGE_FEAT_DIM = 172
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_raw_features = np.pad(edge_raw_features, [(0,0),(0,EDGE_FEAT_DIM-edge_raw_features.shape[1])])
    elif edge_raw_features.shape[1] > EDGE_FEAT_DIM:
        edge_raw_features = edge_raw_features[:, :EDGE_FEAT_DIM]

    # 加载标签数据
    unit_status_df = pd.read_csv(f'./DG_data/{dataset_name}/state.csv')
    labels = unit_status_df.values.astype(np.float32).reshape(24, 54, 365).transpose(2, 1, 0)

    # 严格按天数划分
    day_indices = (graph_df.ts.astype(int) - 1).values  # 将1.0-365.0转换为0-364的整数索引
    max_day = day_indices.max()
    total_days = max_day + 1  # 实际总天数

    # 计算各数据集天数
    train_days = int(total_days * (1 - val_ratio - test_ratio))
    val_days = int(total_days * val_ratio)
    test_days = total_days - train_days - val_days

    # 创建严格的天数掩码
    train_day_mask = day_indices < train_days
    val_day_mask = (day_indices >= train_days) & (day_indices < (train_days + val_days))
    test_day_mask = day_indices >= (train_days + val_days)

    # 基础数据
    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = np.arange(len(graph_df), dtype=np.longlong)

    # 构建数据集
    full_data = Data(src_node_ids, dst_node_ids, node_interact_times, edge_ids, labels[day_indices])
    train_data = Data(
        src_node_ids[train_day_mask],
        dst_node_ids[train_day_mask],
        node_interact_times[train_day_mask],
        edge_ids[train_day_mask],
        labels[day_indices[train_day_mask]]
    )
    val_data = Data(
        src_node_ids[val_day_mask],
        dst_node_ids[val_day_mask],
        node_interact_times[val_day_mask],
        edge_ids[val_day_mask],
        labels[day_indices[val_day_mask]]
    )
    test_data = Data(
        src_node_ids[test_day_mask],
        dst_node_ids[test_day_mask],
        node_interact_times[test_day_mask],
        edge_ids[test_day_mask],
        labels[day_indices[test_day_mask]]
    )

    # 严格时序验证
    time_checks = (
        train_data.node_interact_times.max() < val_data.node_interact_times.min(),
        val_data.node_interact_times.max() < test_data.node_interact_times.min()
    )
    assert all(time_checks), f"时序验证失败: 训练({time_checks[0]}), 验证({time_checks[1]})"

    print(f"\n严格天数划分结果（共{total_days}天）:")
    print(f"训练集: 第1-{train_days}天（{train_data.num_interactions}条样本）")
    print(f"验证集: 第{train_days+1}-{train_days+val_days}天（{val_data.num_interactions}条样本）")
    print(f"测试集: 第{train_days+val_days+1}-{total_days}天（{test_data.num_interactions}条样本）")

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, Data(*[np.array([])]*5), Data(*[np.array([])]*5)


def get_other_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    为TCN和GCN模型准备数据（含标签）

    参数:
        dataset_name: 数据集名称
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    返回:
        train_node: (train_days, 118, 24) 训练节点特征
        val_node: (val_days, 118, 24) 验证节点特征
        test_node: (test_days, 118, 24) 测试节点特征
        edge_index: (2, num_edges) 边连接关系
        train_labels: (train_days, 54, 24) 训练标签
        val_labels: (val_days, 54, 24) 验证标签
        test_labels: (test_days, 54, 24) 测试标签
    """
    # 读取节点特征（调整为days-first格式）
    node_features = pd.read_csv(f'./DG_data/{dataset_name}/all_busloads.csv').values.astype(np.float32)
    # 转换为(365, 118, 24)格式
    node_features = node_features.reshape(118, 24, 365).transpose(2, 1, 0)  # [365, 24, 118]

    # 读取边数据（源节点、目标节点）
    edge_df = pd.read_csv(f'./DG_data/{dataset_name}/node.csv', header=None)
    edge_index = np.array(edge_df)
    edge_index = edge_index.transpose() - 1  # 转为[2, num_edges]格式并调整索引从0开始

    # 读取标签数据（调整为days-first格式）
    label_df = pd.read_csv(f'./DG_data/{dataset_name}/state.csv')
    labels = label_df.values.astype(np.float32).reshape(24, 54, 365).transpose(2, 1, 0)  # [365, 54, 24]

    # 按天数划分数据集
    total_days = 365
    train_days = int(total_days * (1 - val_ratio - test_ratio))
    val_days = int(total_days * val_ratio)
    test_days = total_days - train_days - val_days

    # 划分数据集
    train_node = node_features[:train_days]  # [train_days, 118, 24]
    val_node = node_features[train_days:train_days + val_days]  # [val_days, 118, 24]
    test_node = node_features[train_days + val_days:]  # [test_days, 118, 24]

    # 划分标签数据
    train_labels = labels[:train_days]  # [train_days, 54, 24]
    val_labels = labels[train_days:train_days + val_days]  # [val_days, 54, 24]
    test_labels = labels[train_days + val_days:]  # [test_days, 54, 24]

    return train_node, val_node, test_node, edge_index, train_labels, val_labels, test_labels
