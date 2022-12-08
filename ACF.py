import sys
import os
import pandas as pd
import matplotlib.pylab as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot

filename = r'.\data2\handle_Turbine_Data .csv'
f = pd.read_csv(filename, usecols=[0])
data = f.values
def draw_acf_pacf(data):
    """
    输入需要求解ACF\PACF的数据,
    data["xt"]
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 模型的平稳性检验
    """时序图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(data)
    plt.title("时序图")
    fig = plt.figure(figsize=(12, 8))
    """单位根检验"""
    print("单位根检验:\n")
    # print(adfuller(data))

    """ACF"""
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=20, ax=ax1)
    plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.ylabel('ACF', fontproperties='Times New Roman', fontsize=15)
    plt.title(' ')
    # plt.xlabel('a）ACF图', fontproperties='Simsun', fontsize=15)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()

    """PACF"""
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.ylabel('PACF', fontproperties='Times New Roman', fontsize=15)
    plt.title(' ')
    # plt.xlabel('b）PACF图', fontproperties='Simsun', fontsize=15)
    plt.savefig('./img_data/ACF_PACF.png', dpi=700, bbox_inches='tight')
    plt.show()

draw_acf_pacf(data)
