import numpy as np
import matplotlib.pyplot as plt

from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft

filename = r'F:\wind power pre\data2\handle_Turbine_Data .csv'
f = pd.read_csv(filename, usecols=[0])
# filename = r'.\data2\WindForecast_20210101-20211231.xlsx'
# f = pd.read_excel(filename, usecols=[0], sheet_name='Sheet1')
plt.plot(f.values)
alpha = 7000  # moderate bandwidth constraint
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
K_ = [1, 2, 3, 4, 5, 6, 7, 8]  # K modes
center_fre = []
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

"""
alpha、tau、K、DC、init、tol 六个输入参数的无严格要求；
alpha 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍；
tau 噪声容限 ；
K 分解模态（IMF）个数；
DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1；
init 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数；
tol 控制误差大小常量，决定精度与迭代次数
"""

for K in K_:
    u, u_hat, omega = VMD(f.values, alpha, tau, K, DC, init, tol)
    center_fre.append(omega[-1].tolist())
    # print(center_fre)
center_ = np.array(center_fre, dtype=object)
# plt.figure()
for i in range(len(K_)):
    a = center_[i]
    print('执行第%d组中心频率' % i)
    dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
    dataframe.to_csv(r".\frequency_data\%d个IMF中心频率-%d.csv" % ((i + 1), (i + 1)), index=False, sep=',')
u, u_hat, omega = VMD(f.values, alpha, tau, K_[4], DC, init, tol)
K = K_[4]
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(u.T)
plt.title('Decomposed modes')
plt.yticks(fontproperties='Times New Roman', size=16)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16)
plt.savefig('./img_data/Decomposed modes.png')
plt.show()
fig1 = plt.figure()
# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Simsun']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.yticks(fontproperties='Times New Roman', size=16)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16)
plt.ylabel('风电功率/MW')
plt.xlabel('样本数/个')
plt.plot(f.values)

fig1.suptitle('原始风电功率趋势图')
plt.savefig('./img_data/Original_components.png')
plt.show()

plt.figure(figsize=(7, 7), dpi=200)

for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(u[i, :], linewidth=0.2, c='r')
    # plt.yticks(fontproperties='Times New Roman', size=5)  # 设置大小及加粗
    # plt.xticks(fontproperties='Times New Roman', size=2)
    # plt.tick_params(labelsize=3, pad=0.02)
    # plt.tick_params(pad=0.03)
    plt.ylabel('IMF{}'.format(i + 1), fontsize=16)
    plt.tight_layout()
plt.tight_layout()
plt.savefig('./img_data/IMF.png')
plt.show()
# # 中心模态
plt.figure(figsize=(7, 7), dpi=200)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(abs(fft(u[i, :])))
    plt.yticks(fontproperties='Times New Roman', size=16)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.tick_params(labelsize=3, pad=0.02)
    # plt.tick_params(pad=0.03)
    plt.ylabel('IMF{}'.format(i + 1), fontsize=16)
plt.tight_layout()
plt.savefig('./img_data/IMF中心模态.png')
plt.show()
# # 保存子序列数据到文件中
# for i in range(K):
#     a = u[i, :]
#     print('执行第%d个子序列' % i)
#     dataframe = pd.DataFrame({'v{}'.format(i + 1): a})
#     dataframe.to_csv(r".\data\VMDban-%d.csv" % (i + 1), index=False, sep=',')
