import torch as torch

# a = torch.tensor(1)
# b = a.item()
# print(a) # tensor(1)
# print(b) # 1

a = [1, 2, 3]
b = torch.tensor(a)
c = b.numpy().tolist()

print(b) # tensor([1, 2, 3])
print(c) # [1, 2, 3]
