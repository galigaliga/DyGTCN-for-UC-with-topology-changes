import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            # 修正padding计算方式，确保输出长度与输入一致
            padding = (kernel_size - 1) * dilation_size // 2  # 关键修改点
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding,  # 使用修正后的padding
                dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """修正后的TCN机组状态预测模型"""

    def __init__(self, num_nodes, num_features, hidden_dim, dropout):
        super(TCN, self).__init__()

        # 时间卷积网络
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=[hidden_dim] * 3,  # 3层隐藏层
            kernel_size=3,
            dropout=dropout
        )

        # 全连接映射层
        self.node2unit = nn.Linear(num_nodes, 54)  # 118节点 -> 54机组

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.Sigmoid()
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, num_features, num_nodes)
        Returns:
            输出张量 (batch_size, 54, 24)
        """
        # 调整维度: [batch, features, nodes] -> [batch, nodes, features]
        x = x.permute(0, 2, 1)  # [batch, 118, 20]

        # 时间卷积处理: [batch, 118, 20] -> [batch, 118, hidden_dim]
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 节点到机组映射: [batch, 118, hidden] -> [batch, 54, hidden]
        x = self.node2unit(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 输出预测: [batch, 54, hidden] -> [batch, 54, 24]
        return self.output(x)