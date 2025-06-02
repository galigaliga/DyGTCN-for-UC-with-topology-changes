import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_nodes, seq_len, hidden_dim, num_units, output_steps):
        """
        LSTM模型用于时间序列预测
        :param num_nodes: 输入节点数 (118)
        :param seq_len: 输入序列长度 (时间步数)
        :param hidden_dim: LSTM隐藏层维度
        :param num_units: 输出机组数 (54)
        :param output_steps: 输出时间步数 (24)
        """
        super(LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_units = num_units
        self.output_steps = output_steps

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=num_nodes,  # 每个时间步的特征维度为节点数
            hidden_size=hidden_dim,
            batch_first=True  # 输入输出维度为 (batch, seq_len, features)
        )

        # 定义全连接层，将LSTM输出映射到机组维度
        self.fc = nn.Linear(hidden_dim, num_units * output_steps)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, num_nodes)
        :return: 预测结果 (batch_size, output_steps, num_units)
        """
        batch_size = x.size(0)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # 输出形状: (batch, seq_len, hidden_dim)

        # 取最后时间步的输出
        last_output = lstm_out[:, -1, :]  # 形状: (batch, hidden_dim)

        # 全连接层生成预测
        predictions = self.fc(last_output)  # 形状: (batch, num_units*output_steps)

        # 调整输出形状
        output = predictions.view(batch_size, self.output_steps, self.num_units)

        output = output.permute(0, 2, 1)

        return output  # 使用sigmoid激活输出0-1概率

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_nodes={self.num_nodes}, '
                f'seq_len={self.seq_len}, '
                f'hidden_dim={self.hidden_dim}, '
                f'num_units={self.num_units}, '
                f'output_steps={self.output_steps})')