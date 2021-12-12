import torch as torch


A = torch.arange(0, 4)
print(A)
# tensor([0, 1, 2, 3])

B = torch.arange(5, 9)
print(B)
# tensor([5, 6, 7, 8])

C = torch.stack((A, B), 0)
print(C)
# tensor([[0, 1, 2, 3],
#         [5, 6, 7, 8]])

D = torch.stack((A, B), 1)
print(D)
# tensor([[0, 5],
#         [1, 6],
#         [2, 7],
#         [3, 8]])
