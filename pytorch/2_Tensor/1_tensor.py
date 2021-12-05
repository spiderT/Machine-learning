import torch as torch

a = torch.tensor(1)
b = a.item()

print(a) # tensor(1)
print(b) # 1