def func(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)
    
sum = 0
a = ['2','-23.3']
for item in a:
    sum += float(item)
    
print(sum)