from matplotlib import pyplot as plt
num = [1000,900,800,700,600,500,400,300,200,100]
nun=[100,1000]
mape = [0.269830,0.261569,0.262915,0.256251,0.232411,0.228707,0.257001,0.243700,0.314135,0.242087]
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
# plt.plot(num,mape, color='purple', label='mape')
plt.hist(x=mape,bins=10,label='直方图',facecolor = 'g',edgecolor = 'b', alpha=1)
plt.tick_params(labelsize=10)
plt.axis([100, 1000, 0, 0.35])
plt.title('同一模型同源不同规模的MAPE值',fontsize=20)
plt.xlabel('采样点规模',fontsize=20)
plt.ylabel('MAPE',fontsize=20)
plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=20)
plt.tight_layout()

plt.legend(loc=2, prop={'size': 20})
plt.show()
