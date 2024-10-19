import torch
import torch.nn as nn
import cv2
import numpy as np

# 定义 LeNet 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = LeNet()
net.load_state_dict(torch.load('LeNet.params', map_location=device))
net.to(device)
net.eval()

# 类别标签
class_names = [str(i) for i in range(10)]

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，结束程序")
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用高斯滤波去噪
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化处理
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 过滤小区域
        if w >= 10 and h >= 10:
            roi = thresh[y:y+h, x:x+w]
            # 调整大小为28x28
            resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            # 归一化并转换为张量
            img_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # 模型预测
            with torch.no_grad():
                outputs = net(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = predicted.item()

            # 在帧上绘制矩形和预测结果
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{predicted_label}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示帧
    cv2.imshow('Handwritten Digit Recognition', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
