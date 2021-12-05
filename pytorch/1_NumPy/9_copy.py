import numpy as np

a = np.arange(6)
print(a.shape)
# 输出：(6,)
print(a)
# 输出：[0 1 2 3 4 5]

b = a.view()
print(b.shape)
# 输出：(6,)
b.shape = 2, 3
print(b)
# 输出：[[0 1 2]
#  [3 4 5]]
b[0, 0] = 111
print(a)
# 输出：[111   1   2   3   4   5]
print(b)
# 输出：[[111   1   2]
#  [  3   4   5]]
