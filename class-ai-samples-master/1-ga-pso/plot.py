# -*- encoding:utf8 -*-
# 画出图像如下
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10,6))
ax = Axes3D(fig)
x = np.arange(-1, 2, 0.01)
y = np.arange(-1, 2, 0.01)
X, Y = np.meshgrid(x, y)       
Z = X * np.sin(4*np.pi*X) - Y * np.sin(4*np.pi*Y+np.pi) + 1
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim([-4, 6])
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
