from numpy import DataSource
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        x = list(range(0,50,1))
        print("x:", x)
        self.data = x

    def __getitem__(self, index):
        # print(type(self.data[index]))
        # print(len(self.data[index]))
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
train_loader = DataLoader(dataset=dataset, batch_size=10, 
shuffle=True, num_workers=2)
print(type(dataset))
print(len(dataset))
for i in dataset:
    print(i)
    break

print(type(train_loader))
print(len(train_loader))
for i in train_loader:
    print(i)
    break
# for i, batch in enumerate(train_loader, 0):
#     print(i, batch)
    
# print("\n","* "*30, "\n")

# class MyDataloader(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle, num_workers):
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             collate_fn=self.collate,
#             shuffle=shuffle,
#             num_workers=num_workers,
#         )
    
#     def collate(self, data):
#         return sum(data)

# my_train_loader = MyDataloader(dataset=dataset, batch_size=10, 
# shuffle=True, num_workers=2)

# for i, batch in enumerate(my_train_loader, 0):
#     print(i, batch)
