import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    The Model of CNN1
    """

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        self.d_ff = configs.d_ff
        self.n_classes = configs.n_classes

        self.dropout = configs.dropout

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=self.d_ff,  # 50
            kernel_size=configs.kernel_size,  # 8
            padding=3,
            bias=True,
        )

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        self.dropout1 = nn.Dropout(p=self.dropout)

        # 2. 卷积层2: (2,8)卷积核，padding=valid，输出通道50
        self.conv2 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(2, 8),  # 卷积核大小
            padding=0,  # valid padding
            bias=True,
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.dropout2 = nn.Dropout(p=self.dr)

        # 计算卷积层输出维度，用于全连接层输入
        # 卷积2输出维度计算：(2,128) → (1, 121) （(128-8+1)=121）
        conv2_output_dim = 50 * 1 * 121  # 50通道 × 1高 × 121宽

        # 3. 全连接层1
        self.dense1 = nn.Linear(
            in_features=conv2_output_dim, out_features=256, bias=True
        )
        # 使用he_normal初始化（对应Keras的he_normal）
        nn.init.kaiming_normal_(self.dense1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.dense1.bias)

        self.dropout3 = nn.Dropout(p=self.dr)

        # 4. 全连接层2（分类层）
        self.dense2 = nn.Linear(in_features=256, out_features=classes, bias=True)
        nn.init.kaiming_normal_(self.dense2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.dense2.bias)

    def forward(self, x):
        """
        前向传播
        x: 输入张量，形状为 [batch_size, 2, 128]
        """
        # Reshape: [batch_size, 2, 128] → [batch_size, 1, 2, 128]
        # 对应Keras的Reshape(input_shape + [1])
        x = x.unsqueeze(1)  # 添加通道维度

        # 卷积层1 + ReLU + Dropout
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        # 卷积层2 + ReLU + Dropout
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        # Flatten: [batch_size, 50, 1, 121] → [batch_size, 50*1*121]
        x = x.flatten(start_dim=1)

        # 全连接层1 + ReLU + Dropout
        x = F.relu(self.dense1(x))
        x = self.dropout3(x)

        # 全连接层2 + Softmax
        x = self.dense2(x)
        x = F.softmax(x, dim=1)

        return x

    def load_weights(self, weights_path):
        """
        加载权重文件
        注意：这里默认加载PyTorch格式的权重文件(.pth)
        如果需要加载Keras的.h5权重，需要额外的转换逻辑
        """
        checkpoint = torch.load(weights_path, map_location=device)
        self.load_state_dict(checkpoint)
        print(f"成功加载权重文件: {weights_path}")


def count_parameters(model):
    """辅助函数：计算并打印模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    return total_params


if __name__ == "__main__":
    # 创建模型实例
    model = VT_CNN2(weights=None, input_shape=[2, 128], classes=11).to(device)

    # 打印模型结构
    print("VT-CNN2 模型结构:")
    print(model)

    # 打印参数数量
    count_parameters(model)

    # 测试前向传播
    batch_size = 16
    test_input = torch.randn(batch_size, 2, 128).to(device)  # 模拟输入
    test_output = model(test_input)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")  # 应输出 [16, 11]
    print(f"输出概率和: {torch.sum(test_output[0]).item():.4f}")  # Softmax输出和应为1
