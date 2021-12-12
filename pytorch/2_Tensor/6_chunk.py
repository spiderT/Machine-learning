import torch as torch


A = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
B = torch.chunk(A, 2, 0)
print(B)
# (tensor([1, 2, 3, 4, 5]), tensor([ 6,  7,  8,  9, 10]))


C = torch.chunk(A, 3, 0)
print(C)
# (tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8]), tensor([ 9, 10]))


D = torch.ones(4, 4)
print(D)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
E = torch.chunk(D, 2, 0)
print(E)
# (tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]]),
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]]))
