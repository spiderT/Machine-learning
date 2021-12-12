import torch as torch

# x = torch.rand(2, 1, 3)
# print(x.shape)  # torch.Size([2, 1, 3])
#
# y = x.squeeze(1)
# print(y.shape)  # torch.Size([2, 3])
#
# z = y.squeeze(1)
# print(z.shape)  # torch.Size([2, 3])

x = torch.rand(2, 1, 3)
y = x.unsqueeze(2)
print(y.shape)  # torch.Size([2, 1, 1, 3])
