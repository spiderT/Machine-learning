import torch as torch

# a = torch.zeros(2, 3, 5)
# print(a.shape) # torch.Size([2, 3, 5])
# print(a.size()) # torch.Size([2, 3, 5])
# print(a.numel()) # 30 = 2*3*5


# x = torch.rand(2, 3, 5)
# print(x.shape)  # torch.Size([2, 3, 5])
# # x = x.permute(2, 1, 0)
# # print(x.shape)  # torch.Size([5, 3, 2])
#
# x = torch.rand(2, 3, 5)
# x = x.transpose(1, 0)
# print(x.shape)  # torch.Size([3, 2, 5])



x = torch.randn(4, 4)
# print(x.shape)  # torch.Size([4, 4])

x = x.view(2, 8)
# print(x.shape)  # torch.Size([2, 8])

x = x.permute(1, 0)
# print(x.shape)  # torch.Size([8, 2])

x = x.reshape(4, 4)
print(x.shape)
x.view(4, 4)  # torch.Size([4, 4])

# Traceback (most recent call last):
#   File "/Users/eleme/tt/summary/AI-learn/pytorch/2_Tensor/2_shape.py", line 29, in <module>
#     x.view(4, 4)
# RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

