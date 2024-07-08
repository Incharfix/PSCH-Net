import numpy as np


# 中心归一化. 减去均值，并除以点距原点的最大距离。


# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / m
#
#     return pc

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  #求中心，对pc数组的每行求平均值，通过这条函数最后得到一个1×3的数组[x_mean,y_mean,z_mean];
    pc = pc - centroid  #点云平移  或  # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 将同一列的元素取平方相加得(x^2+y^2+z^2)，再开方，取最大，得最大标准差
    #pc ** 2 平移后的点云求平方   #np.sum(pc ** 2, axis=1)：每列求和
    pc = pc / m   # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc

import numpy as np
import matplotlib.pyplot as plt


def noramlization01(data):
  minVals = data.min(0)
  maxVals = data.max(0)
  ranges = maxVals - minVals
  normData = np.zeros(np.shape(data))
  m = data.shape[0]
  normData = data - np.tile(minVals, (m, 1))
  normData = normData/np.tile(ranges, (m, 1))
  return normData, ranges, minVals


x = np.array([[78434.0829, 26829.86612], [78960.4042, 26855.13451], [72997.8308, 26543.79201],
       [74160.2849, 26499.56629], [75908.5746, 26220.11996], [74880.6989, 26196.03995],
       [74604.7169, 27096.87862], [79547.6796, 25986.68579], [74997.7791, 24021.50132],
       [74487.4915, 26040.18441], [77134.2636, 24647.274],  [74975.2792, 24067.31441],
       [76013.5305, 24566.02273], [79191.518, 26840.29867], [80653.4589, 25937.22248],
       [79185.9935, 26996.18228], [74426.881, 24227.71439], [73246.4295, 26561.59268],
       [77963.1478, 25580.05298], [74469.8778, 26082.15448], [81372.3787, 26649.69232],
       [76826.8262, 24549.77367], [77774.2608, 25999.96037], [79673.1361, 25229.04353],
       [75251.7951, 24902.72185], [78458.073, 23924.15117], [82247.5439, 29671.33493],
       [82041.2247, 27903.34268], [80083.2029, 28692.35517], [80962.0043, 28519.81002],
       [79799.8328, 28740.27736], [80743.9947, 28862.75402], [80888.449, 29724.53706],
       [81768.4638, 30180.20618], [80283.8783, 30417.55057], [79460.7078, 29092.52867],
       [75514.1202, 28071.73721], [80595.5945, 30292.25917], [80750.4876, 29651.32254],
       [80020.662, 30023.70025], [82992.3395, 29466.83067], [80185.5946, 29943.15481],
       [81854.6163, 29846.18257], [81526.4017, 30218.27078], [79174.5312, 29960.69999],
       [78112.3051, 26467.57545], [80262.4121, 29340.23218], [81284.9734, 28257.71529],
       [81928.9905, 28752.84811], [80739.2727, 29288.85126], [83135.3435, 30223.4974],
       [83131.8223, 29049.10112], [82549.9076, 28910.15209], [81574.0822, 28326.55367],
       [80507.399, 28553.56851], [82956.2103, 29157.62372], [81909.7132, 29359.24497],
       [80893.5603, 29326.64155], [82520.1272, 30424.96703], [82829.8548, 31062.24418],
       [80532.1495, 29198.10407], [80112.7963, 29143.47905], [81175.0882, 28443.10574]])


# print(x.shape)
# newgroup, _, _ = noramlization(x)
# newdata = newgroup
#
# print(len(x[:, 0]))
# print(len(x[:, 1]))
# print(newdata)