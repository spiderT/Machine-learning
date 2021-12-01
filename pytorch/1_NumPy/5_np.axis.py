import numpy as np

a = np.arange(18).reshape(3, 2, 3)
# print(a)

# [[[ 0  1  2]
#   [ 3  4  5]]

#  [[ 6  7  8]
#   [ 9 10 11]]

#  [[12 13 14]
#   [15 16 17]]]

b = a.max(axis=0)
# print(b)

# [[12 13 14]
#  [15 16 17]]

c = a.max(axis=1)
# print(c)

# [[ 3  4  5]
#  [ 9 10 11]
#  [15 16 17]]

d = a.max(axis=2)
print(d)

# [[ 2  5]
#  [ 8 11]
#  [14 17]]
