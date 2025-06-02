import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """基于PyG的机组状态预测GCN模型（含全连接映射层）"""

    def __init__(self, num_nodes, num_features, hidden_dim, dropout):
        super(GCN, self).__init__()
        # 第一层图卷积
        self.conv1 = GCNConv(num_features, hidden_dim)
        # 第二层图卷积
        self.conv2 = GCNConv(hidden_dim, 24)  # 每个节点输出24小时预测
        # 全连接映射层（118节点 -> 54机组）
        self.fc = nn.Linear(num_nodes, 54)    # 输入118节点，输出54机组
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, GCNConv):
                nn.init.xavier_uniform_(module.lin.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        """
        Args:
            x: 输入特征张量，形状为 (batch_size, num_features, num_nodes)
            edge_index: 边索引张量，形状为 (2, num_edges)
        Returns:
            预测结果张量，形状为 (batch_size, 54, 24)
        """
        # 维度调整：(batch, features, nodes) -> (batch, nodes, features)
        x = x.permute(0, 2, 1)  # 新维度: (batch_size, 118, 24)
        batch_size = x.size(0)

        # 合并批次维度与节点维度
        x = x.reshape(-1, x.size(2))  # (batch_size*118, 24)

        # 扩展边索引以支持小批量处理
        edge_index = self._expand_edge_index(edge_index, batch_size)

        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层GCN
        x = self.conv2(x, edge_index)  # (batch_size*118, 24)

        # 恢复原始维度并应用全连接映射
        x = x.view(batch_size, 118, 24)  # (batch_size, 118, 24)
        x = x.permute(0, 2, 1)          # (batch_size, 24, 118)
        x = self.fc(x)                  # (batch_size, 24, 54)
        x = x.permute(0, 2, 1)          # (batch_size, 54, 24)

        return x

    def _expand_edge_index(self, edge_index, batch_size):
        """扩展边索引以支持小批量处理"""
        if batch_size == 1:
            return edge_index

        # 为每个样本复制边索引
        num_nodes = 118
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        new_edge_index = edge_index.repeat(1, batch_size) + (
            offsets.repeat_interleave(edge_index.size(1)).unsqueeze(0)
        )
        return new_edge_index