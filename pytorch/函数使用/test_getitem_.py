# Python的魔法方法__getitem__ 可以让对象实现迭代功能，这样就可以使用for...in... 来迭代该对象了
from torch.utils.data import Dataset

a = [['a', '1'],['b','2']]

class AA(Dataset):
    def __init__(self, stre):
        self.a = stre
        
    def __getitem__(self, index):
        print("index:", index)
        print(self.a[index])
        return self.a[index]
    
a=AA(a)
print(type(a))
for item in a:
    print("item:", item)
# a会自己变成一个list



# class Animal:
#     def __init__(self, animal_list):
#         self.animals_name = animal_list

# animals = Animal(["dog","cat","fish"])
# for animal in animals:
#     print(animal) # TypeError: 'Animal' object is not iterable
    
# # 在用 for..in.. 迭代对象时，如果对象没有实现 __iter__ __next__ 迭代器协议，
# # Python的解释器就会去寻找__getitem__ 来迭代对象，如果连__getitem__ 都没有定义，
# # 这解释器就会报对象不是迭代器的错误

class Animal:
    def __init__(self, animal_list):
        self.animals_name = animal_list

    def __getitem__(self, index):
        return self.animals_name[index]

animals = Animal(["dog","cat","fish"])
for animal in animals:
    print(animal)
    

print("++++++++++++++++++++++++++++++++++++++++++++++++")
k = [1,2,3]
print("len(k):", len(k))
print("k.__len__():", k.__len__())

print("k[0]:", k[0])
print("k.__getitem__(0):", k.__getitem__(0))

class FrenchDeck:
    ranks = [1,2,3,4,5,6]
    suits = 'spades diamonds clubs hearts'.split()
    print("suits:", suits)
    
    def __init__(self):
        self._cards = [(rank, suit) for suit in self.suits
        for rank in self.ranks]
        print("self._cards:", self._cards)
        
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()
print('deck.__len__():',deck.__len__())
print('len(deck):',len(deck))
print("===================")
print('deck.__getitem__(0):',deck.__getitem__(0))
print('deck[0]:',deck[0])
# for item in deck:
#     print(item)