import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成数据
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 转化为二维数组
y = np.sin(x).ravel() + np.random.normal(0, 0.1, size=x.shape).ravel()  # 添加噪声

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 创建和训练神经网络模型
mlp = MLPRegressor(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', warm_start=True, random_state=42)

# 训练过程，记录每次训练的损失
num_epochs = 500
loss_values = []

for epoch in range(num_epochs):
    mlp.fit(x_train, y_train)  # 每次迭代更新模型
    y_train_pred = mlp.predict(x_train)
    
    # 计算均方误差
    loss = mean_squared_error(y_train, y_train_pred)
    loss_values.append(loss)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

# 预测
y_pred_train = mlp.predict(x_train)
y_pred_test = mlp.predict(x_test)

# 在整个 x 上进行预测，用于绘制拟合的曲线
y_pred_all = mlp.predict(x)

# 可视化结果
plt.figure(figsize=(10, 6))

# 绘制原始数据
plt.scatter(x_train, y_train, label='Train data', color='blue', s=10)
plt.scatter(x_test, y_test, label='Test data', color='green', s=10)

# 绘制真实的正弦曲线
plt.plot(x, np.sin(x), label='True sine wave', color='orange', linestyle='dashed')

# 绘制拟合的曲线
plt.plot(x, y_pred_all, label='Fitted curve by MLP', color='red', linewidth=2)

plt.title('Fitting a Sine Curve Using MLP')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss', color='purple')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.legend()
plt.show()