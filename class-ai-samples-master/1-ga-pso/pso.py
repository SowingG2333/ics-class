# -*- coding: utf-8 -*-
'''
@File    :   pso.py,
@Time    :   Tue Dec 15 10:45:42 CST 2020
@Version :   1.0,
@Contact :   https://blog.csdn.net/cyril_ki/article/details/108589078
@License :   GPL
@Desc    :   粒子群算法（PSO）的应用示例

Modified by: HAO Jiasheng, for teaching @ UESTC

'''

import random
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl

class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        # 初始化
        self.dimension = dimension  # 变量个数
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            # 做出修改
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, xx):
        """
        个体适应值计算
        """
        x = xx[0]
        y = xx[1]
    
        fit = x * np.sin(4*np.pi*x) - y * np.sin(4*np.pi*y + np.pi) + 1
        return fit

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + \
                        c1 * random.uniform(0, self.p_best[i] - self.x[i]) + \
                        c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def run(self, plot_func):
        best = []
        self.final_best = np.array([1, 2])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            temp = self.fitness(self.final_best)
            best.append(temp)
            result = ('【结果】当前最优值: {:.4f}, 位置：({:.4f},{:.4f}), 轮数: {}'\
                      .format(temp, self.final_best[0], self.final_best[1], gen))
            print(result)

        t = [i for i in range(self.time)]
        plot_func(t, best)

        return result
        
def plot_func(t, best):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.plot(t, best, color='red', marker='.', ms=15)
    plt.rcParams['axes.unicode_minus'] = False
    plt.margins(0)
    plt.xlabel(u"迭代次数")  # X轴标签
    plt.ylabel(u"适应度")  # Y轴标签
    plt.title(u"迭代过程")  # 标题
    plt.show()


if __name__ == '__main__':
    
    dimension = 2        # 问题维数
    time      = 50       # 迭代次数
    size      = 100      # 群体规模
    low       = [-1, -1] # 约束范围
    up        = [2, 2]   # 约束范围
    v_low     = -1       # 速度下界
    v_high    = 1        # 速度上界
    
    pso = PSO(dimension, time, size, low, up, v_low, v_high)

    result = pso.run(plot_func)
    print(result)


