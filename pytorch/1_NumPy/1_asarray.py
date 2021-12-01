import numpy as np

# arr_1_d = np.asarray([1])
# print(arr_1_d)  # [1]

arr_2_d = np.asarray([[1, 2], [3, 4]])
# print(arr_2_d)  # [[1 2] [3 4]]

# print(arr_1_d.ndim)  # 1
# print(arr_2_d.ndim)  # 2

# print(arr_1_d.shape) # (1,)
# print(arr_2_d.shape) # (2, 2)


# arr_2_d.reshape((4, 1))
# print(arr_2_d)

# print(arr_2_d.size) # 4
# print(arr_2_d.dtype) # int64

arr_3_d = np.asarray([[1, 2], [3, 4]], dtype="float64")
# print(arr_3_d.dtype) # float64

arr_3_d_int = arr_3_d.astype('int32')
# print(arr_3_d.dtype) # float64
# print(arr_3_d_int.dtype) # int32

print(arr_3_d) # [[ 1.  2.] [ 3.  4.]]






