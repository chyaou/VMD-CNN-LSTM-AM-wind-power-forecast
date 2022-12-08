import math
from itertools import combinations

# Pearson algorithm
import pandas as pd


def pearson(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))


# Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))


# Kendall algorithm
def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0  # concordant count
    d = 0  # discordant count
    t = 0  # tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)


filename = r'.\data2\handle_Turbine_Data .csv'
f = pd.read_csv(filename, usecols=[0])
f2 = pd.read_csv(filename, usecols=[1])

GSM = f.values.tolist()
# LGC = f2.values.tolist()
GSM = [i[0] for i in GSM]


# LGC = [i[0] for i in LGC]
# read in file
# print(GSM)
def select_tezheng(i):
    fi = pd.read_csv(filename, usecols=[i])
    LGC = fi.values.tolist()
    LGC = [i[0] for i in LGC]
    return LGC


# kendall_test = kendall(data, LGC)
# pearson_test = pearson(GSM, LGC)
for LGC in range(1, 14):
    LGC_ = select_tezheng(LGC)
    spearman_test = spearman(GSM, LGC_)
    print("斯皮尔曼系数：", spearman_test)
# spearman_test = spearman(GSM, LGC)

# print("肯德尔系数：", kendall_test)
# print("皮尔逊系数：", pearson_test)
# print("斯皮尔曼系数：", spearman_test) # BearingShaftTemperature 斯皮尔曼系数： 0.7353663944717617
