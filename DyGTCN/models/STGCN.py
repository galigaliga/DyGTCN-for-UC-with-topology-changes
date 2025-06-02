import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dims, output_steps, dropout, edge_index):
        """
        STGCN模型（从118节点映射到54台机组的预测）

        参数:
        num_nodes: 节点数量 (118)
        num_features: 输入特征维度 (24)
        hidden_dims: 隐藏层维度
        output_steps: 输出时间步长 (24)
        dropout: Dropout率
        edge_index: 图结构的边索引
        """
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        self.num_nodes = num_nodes
        self.output_steps = output_steps

        # 时间卷积层
        self.temporal_conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dims,
            kernel_size=3,
            padding=1
        )
        self.temporal_conv2 = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=3,
            padding=1
        )

        # 图卷积层
        self.gcn1 = GCNConv(hidden_dims, hidden_dims)
        self.gcn2 = GCNConv(hidden_dims, hidden_dims)

        # 输出层
        self.fc = nn.Linear(hidden_dims, output_steps)

        # 节点到机组映射层：118 → 54
        self.node_to_unit = nn.Linear(num_nodes, 54)

        # 正则化层
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dims)

    def forward(self, x):
        """
        前向传播

        输入:
        x: 输入特征张量 (batch_size, num_features, num_nodes)

        输出:
        output: 预测结果 (batch_size, 54, 24)
        """
        batch_size = x.size(0)

        # 转换输入格式以适应时间卷积
        x = x.permute(0, 2, 1)  # (batch, nodes, features)
        x = x.reshape(-1, 1, x.size(2))  # (batch*nodes, 1, features)

        # 时间卷积处理
        x = self.temporal_conv1(x)  # (batch*nodes, hidden, features)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.temporal_conv2(x)  # (batch*nodes, hidden, features)
        x = F.relu(x)
        x = x.mean(dim=2)  # (batch*nodes, hidden)

        # 图卷积处理
        x = self.gcn1(x, self.edge_index)
        x = F.relu(x)
        x = self.bn(x)
        x = self.gcn2(x, self.edge_index)
        x = F.relu(x)

        # 恢复形状 (batch, nodes, hidden)
        x = x.view(batch_size, self.num_nodes, -1)

        # 输出每个节点的时间步预测
        x = self.fc(x)  # (batch, nodes, output_steps=24)

        # 映射到机组维度
        x = x.permute(0, 2, 1)          # (batch, 24, 118)
        x = self.node_to_unit(x)        # (batch, 24, 54)
        output = x.transpose(1, 2)     # (batch, 54, 24)

        return output  # 不再加 sigmoid，适用于 BCEWithLogitsLoss
