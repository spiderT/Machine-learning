import torch as torch


# A = torch.rand(4, 4)
# print(A)
# # tensor([[0.6418, 0.4171, 0.7372, 0.0733],
# #         [0.0935, 0.2372, 0.6912, 0.8677],
# #         [0.5263, 0.4145, 0.9292, 0.5671],
# #         [0.2284, 0.6938, 0.0956, 0.3823]])

# B = torch.split(A, 2, 0)
# print(B)
# # 原来 4x4 大小的 Tensor A，沿着第 0 维度，也就是沿“行”的方向，按照每组 2“行”的大小进行切分，得到了两个 2x4 大小的 Tensor。
# # (tensor([[0.6418, 0.4171, 0.7372, 0.0733],
# #         [0.0935, 0.2372, 0.6912, 0.8677]]),
# # tensor([[0.5263, 0.4145, 0.9292, 0.5671],
# #         [0.2284, 0.6938, 0.0956, 0.3823]]))


# C=torch.split(A, 3, 0)
# print(C)
# # (tensor([[0.6418, 0.4171, 0.7372, 0.0733],
# #         [0.0935, 0.2372, 0.6912, 0.8677],
# #         [0.5263, 0.4145, 0.9292, 0.5671]]),
# # tensor([[0.2284, 0.6938, 0.0956, 0.3823]]))

A = torch.rand(5, 4)
print(A)
# tensor([[0.1005, 0.9666, 0.5322, 0.6775],
#         [0.4990, 0.8725, 0.5627, 0.8360],
#         [0.3427, 0.9351, 0.7291, 0.7306],
#         [0.7939, 0.3007, 0.7258, 0.9482],
#         [0.7249, 0.7534, 0.0027, 0.7793]])

B = torch.split(A, (2, 3), 0)
print(B)
# 将 Tensor A，沿着第 0 维进行切分，每一个结果对应维度上的尺寸或者说大小，分别是 2（行），3（行）。
# (tensor([[0.1005, 0.9666, 0.5322, 0.6775],
#         [0.4990, 0.8725, 0.5627, 0.8360]]),
# tensor([[0.3427, 0.9351, 0.7291, 0.7306],
#         [0.7939, 0.3007, 0.7258, 0.9482],
#         [0.7249, 0.7534, 0.0027, 0.7793]]))
