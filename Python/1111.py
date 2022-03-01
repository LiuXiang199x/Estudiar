class a:
    def __init__(self) -> None:
        self.a=1
        self._a =2
        self.__a=3
        
c =a()
print(c.a)
print(c._a)
#print(c.__a)

class dd(a):
   def __init__(self) -> None:
       super(dd).__init__() 
       self.a=2
       self._a=3
       self.__a__=4

f = dd()
print(f.a)
print(f._a)
print(f.__a__)
f.__a__ = 5