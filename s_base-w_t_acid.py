import math
from pandas import read_excel
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
raw_data = read_excel("/Users/sch/Documents/仪分数据.xlsx")
x_data = np.array(raw_data["V(mL)"], dtype="float64")
y_data = np.array(raw_data["pH"], dtype="float64")
Kw = 10 ** (- 14)  # mol^2/L^2 水自耦电离常数


# 强碱滴定三元弱酸函数，参数为：
# v 滴入的滴定剂体积
# ka1/2/3 各级电离常数
# v0 待测溶液的初始体积
# ca 待测三元弱酸的浓度
# cb 碱滴定剂的浓度
# 该函数自变量为 v，输出值为滴入 v 体积滴定剂后溶液的平衡 pH，易知该函数为理想滴定曲线
def sb_wta(v, ka1, ka2, ka3, v0, ca, cb):
    global Kw
    b = cb * v / (v + v0)  # 滴入 v 体积滴定剂后碱阳离子浓度
    c = ca * v0 / (v + v0)  # 滴入 v 体积滴定剂后待测三元弱酸的所有组分总浓度

    # 酸碱平衡函数，参数为：
    # h 氢离子浓度
    # eb 滴入 v 体积滴定剂后碱阳离子浓度，与 b 相同
    # 该函数实际上输出当前溶液状态下氢离子浓度为 h 时溶液中所有正电荷加负电荷的值，由电荷守恒可得，该函数零点即为当前溶液状态下实际氢离子浓度
    def eq(h, eb):
        # 利用各物种分布系数进行计算
        denominator = (h ** 3 + ka1 * h ** 2 + ka1 * ka2 * h + ka1 * ka2 * ka3)
        h2a = c * ka1 * h ** 2 / denominator
        ha = c * ka1 * ka2 * h / denominator
        a = c * ka1 * ka2 * ka3 / denominator
        oh = Kw / h
        return h2a + 2 * ha + 3 * a + oh - (h + eb)
    # 二分法求函数 eq 的零点以得实际氢离子浓度。由于输入输出均为 pH 因此中间进行了换算
    left = 0
    right = 14
    eps = 0.0001
    while right - left > eps:
        middle = (left + right) / 2
        if eq(math.pow(10, -middle), b) == 0:
            break
        if eq(math.pow(10, -left), b) * eq(math.pow(10, -middle), b) < 0:
            right = middle
        else:
            left = middle
    return (left + right) / 2


# 由于函数 eq 中二分法的代码块不支持数组类型的输入，新建了函数 f 将输入的数组各值逐个代入 eq 并合并输出
def f(n, ka1, ka2, ka3, v0, ca, cb):
    out = []
    for i in n:
        out.append(sb_wta(i, ka1, ka2, ka3, v0, ca, cb))
    return np.array(out, dtype="float64")


# 进行非线性拟合。值得注意的是 curve_fit 为局部优化算法，使用时须要小心调试初值 p0，否则无法得到适宜的拟合
# 由于该函数参数较多，也须要加入合适的边界条件 bounds，否则无法得到适宜的拟合与协方差矩阵
line = curve_fit(f, x_data, y_data, p0=(0.008, 10**(-7), 10**(-13), 27, 0.036, 0.1006),
                 bounds=((0.001, 10**(-8), 10**(-14), 25, 0, 0.09),
                         (0.01, 10**(-6), 10**(-11), 30, 0.05, 0.11)))
# 输出结果
ka1_fit, ka2_fit, ka3_fit, v0_fit, ca_fit, cb_fit = line[0]
pcov = line[1]
print("Ka1: " + str(ka1_fit))
print("Ka2: " + str(ka2_fit))
print("Ka3: " + str(ka3_fit))
print("pKa1: " + str(-math.log(ka1_fit, 10)))
print("pKa2: " + str(-math.log(ka2_fit, 10)))
print("pKa3: " + str(-math.log(ka3_fit, 10)))
print("V0: " + str(v0_fit))
print("ca: " + str(ca_fit))
print("cb: " + str(cb_fit))
print(np.sqrt(np.diag(pcov)))
# 画图
x_fit = np.linspace(int(x_data[0]), int(x_data[len(x_data) - 1]), int((x_data[len(x_data) - 1] - x_data[0]) * 10))
y_fit = f(x_fit, ka1_fit, ka2_fit, ka3_fit, v0_fit, ca_fit, cb_fit)
plt.plot(x_fit, y_fit)
plt.scatter(x_data, y_data, s=10, marker="s")
plt.xticks(range(0, 26, 2))
plt.yticks(range(0, 14, 1))
plt.grid(linewidth=0.5)
plt.xlabel("V/mL")
plt.ylabel("pH")
plt.show()
