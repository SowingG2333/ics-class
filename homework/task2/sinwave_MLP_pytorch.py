import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)

net = net.to(torch.device('cuda'))

# 初始化权重
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

# 生成数据
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 转化为二维数组
y = np.sin(x).ravel()  # 添加噪声
x_tensor = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda'))
y_tensor = torch.tensor(y, dtype=torch.float32, device=torch.device('cuda')).view(-1, 1)  # 将y也转为二维以匹配输出
print(x_tensor.device)

# 定义损失函数
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 进行训练
num_epochs = 20000

for epoch in range(num_epochs):
    y_pred = net(x_tensor)
    
    loss = loss_func(y_pred, y_tensor)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

y_pred = y_pred.cpu().detach().numpy()

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original data (sin(x))', color='green')
plt.plot(x, y_pred, label='Fitted data by NN', color='red', linestyle='dashed')
plt.legend()
plt.title('Fitting a Neural Network to a Sin Wave')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()