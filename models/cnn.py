import numpy as np

class CNNClassifier:
    def __init__(self, learning_rate=0.01, epochs=10, filter_size=3, num_filters=4):
        self.lr = learning_rate
        self.epochs = epochs
        self.filter_size = filter_size     # 卷积核宽度 (高度固定为1)
        self.num_filters = num_filters     # 卷积核个数（输出通道数）
        # 权重将在fit中根据数据维度初始化

    def fit(self, X, y):
        # 将数据转换为numpy数组
        X = np.array(X)              # X.shape = (样本数, 特征数)
        y = np.array(y)
        n_samples, n_features = X.shape
        # 确定类别数并进行One-Hot编码
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        # 建立标签到索引的映射，以便One-Hot编码
        class_to_index = {c: i for i, c in enumerate(self.classes)}
        y_onehot = np.zeros((n_samples, num_classes))
        for i, label in enumerate(y):
            y_onehot[i, class_to_index[label]] = 1

        # 决定卷积核尺寸和输出维度
        filter_w = min(self.filter_size, n_features)   # 卷积核宽度，不超过特征数
        F = self.num_filters
        out_width = n_features - filter_w + 1          # 卷积输出长度（无填充）
        if out_width < 1:
            # 若特征维度比滤波器小，调整滤波器宽度为特征数
            filter_w = n_features
            out_width = 1

        # 初始化权重：卷积滤波器权重和偏置，全连接层权重和偏置
        # 卷积权重维度: (F, 1, filter_w) -> F个滤波器，单通道，高度1，宽度filter_w
        self.conv_weights = np.random.randn(F, 1, filter_w) * 0.01
        self.conv_bias = np.zeros(F)
        # 全连接权重维度: (F*out_width, num_classes)
        self.fc_weights = np.random.randn(F * out_width, num_classes) * 0.01
        self.fc_bias = np.zeros(num_classes)

        # 训练循环
        for epoch in range(self.epochs):
            # 遍历每个训练样本进行更新（SGD）
            for i in range(n_samples):
                x = X[i]                  # 单个样本输入，shape=(n_features,)
                target = y_onehot[i]      # 对应的One-Hot标签

                # **前向传播**：
                # 卷积层计算
                # 卷积前激活输出 conv_pre.shape = (F, out_width)
                conv_pre = np.zeros((F, out_width))
                for f in range(F):  # 对每个滤波器
                    for j in range(out_width):  # 沿宽度滑动窗口
                        # 滤波器f在位置j的卷积加权和
                        segment = x[j : j+filter_w]  # 输入窗口片段
                        conv_pre[f, j] = np.sum(self.conv_weights[f, 0, :] * segment) + self.conv_bias[f]
                # ReLU激活
                conv_post = np.maximum(conv_pre, 0)   # 将负值截断为0

                # 展开卷积输出并通过全连接层
                conv_flat = conv_post.reshape(F * out_width)            # 展平为向量
                output = conv_flat.dot(self.fc_weights) + self.fc_bias  # 全连接线性输出 (长度=num_classes)

                # 计算损失的输出误差
                error = output - target  # MSE损失对输出的梯度：预测值-真实值

                # **反向传播**：
                # 全连接层梯度
                grad_output = error  # 输出层梯度
                grad_fc_weights = np.outer(conv_flat, grad_output)      # 全连接权重梯度 (形状: [F*out_width, num_classes])
                grad_fc_bias = grad_output                              # 全连接偏置梯度 (形状: [num_classes])
                # 传播到卷积层输出 (展开后) 的梯度
                grad_conv_flat = self.fc_weights.dot(grad_output)       # 形状: [F*out_width]
                grad_conv = grad_conv_flat.reshape(F, out_width)        # 重塑为卷积输出形状
                # ReLU激活梯度传递：只有正值部分允许梯度通过
                grad_conv_pre = grad_conv * (conv_pre > 0)              # 对应位置conv_pre<=0的梯度置0

                # 卷积层权重和偏置梯度计算
                grad_conv_weights = np.zeros_like(self.conv_weights)    # 形状: [F, 1, filter_w]
                grad_conv_bias = np.zeros_like(self.conv_bias)          # 形状: [F]
                for f in range(F):
                    for j in range(out_width):
                        # 卷积滤波器f在位置j的输出对应于输入segment x[j:j+filter_w]
                        # 每个权重的梯度累加：grad_conv_pre[f,j] * 对应的输入值
                        for p in range(filter_w):
                            grad_conv_weights[f, 0, p] += grad_conv_pre[f, j] * x[j+p]
                        grad_conv_bias[f] += grad_conv_pre[f, j]

                # 参数更新（梯度下降法）
                self.fc_weights -= self.lr * grad_fc_weights
                self.fc_bias   -= self.lr * grad_fc_bias
                self.conv_weights -= self.lr * grad_conv_weights
                self.conv_bias   -= self.lr * grad_conv_bias

        # 返回训练后的模型自身
        return self

    def predict(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        # 使用训练时的权重进行前向计算，输出预测类别
        F = self.conv_weights.shape[0]
        filter_w = self.conv_weights.shape[2]
        out_width = n_features - filter_w + 1
        if out_width < 1:
            out_width = 1
        predictions = []
        for i in range(n_samples):
            x = X[i]
            # 前向传播（与训练时相同，但不需计算梯度）
            conv_pre = np.zeros((F, out_width))
            for f in range(F):
                for j in range(out_width):
                    conv_pre[f, j] = np.sum(self.conv_weights[f, 0, :] * x[j:j+filter_w]) + self.conv_bias[f]
            conv_post = np.maximum(conv_pre, 0)
            conv_flat = conv_post.reshape(F * out_width)
            output = conv_flat.dot(self.fc_weights) + self.fc_bias
            # 取最大输出对应的类别
            class_index = np.argmax(output)
            class_label = self.classes[class_index]
            predictions.append(class_label)
        return np.array(predictions)
