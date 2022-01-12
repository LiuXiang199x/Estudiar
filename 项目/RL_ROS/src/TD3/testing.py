import torch
for i in range(5):
  for j in range(5):
    print(i, j)
    if i == 3 and j == 4:
      break
  else:
    continue
  break
a = torch.ones(2,3).float()
b = a[0,0] == 1
print(b)
if b:
  print("1")