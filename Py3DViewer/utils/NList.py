import numpy as np
from numba.typed import List as L
from numba import njit

@njit(cache=True)
def fast_indexing(l, item):
    indexed = L()
    for el in item:
        indexed.append(l[el])
    return indexed

class NList():
    def __init__(self, l):
        self.__list = l
            
    def __getitem__(self, item):
        if isinstance(item, (int, slice, np.int64, np.int32, np.int16, np.int8)): return self.__list[item]
        item = np.array(item, dtype=np.int64)
        return fast_indexing(self.__list, item)
    
    def __repr__(self):
        return repr(self.__list)
