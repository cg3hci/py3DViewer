import numpy as np
from numba.typed import List as L
from numba import njit

@njit(cache=True)
def fast_indexing(l, item):
    indexed = L()
    for el in item:
        indexed.append(l[el])
    return indexed

@njit(cache=True)
def _to_array(nb_l):
    
    dim1 = len(nb_l)
    dim2 = 0
    
    for el in nb_l:
        if len(el) > dim2:
            dim2=len(el)
    
    array = np.zeros((dim1, dim2), dtype=np.int64)-1
    
    for idx, row in enumerate(nb_l):
        for idy, column in enumerate(row):
            array[idx][idy] = column
            
    return array 

class NList():
    def __init__(self, l):
        self.__list = l
            
    def __getitem__(self, item):
        if isinstance(item, (int, slice, np.int64, np.int32, np.int16, np.int8)): return self.__list[item]
        item = np.array(item, dtype=np.int64)
        return fast_indexing(self.__list, item)
    
    def __repr__(self):
        return repr(self.__list)

    @property
    def content(self):
        return self.__list

    @property
    def array(self):
        return _to_array(self.__list)

    def __len__(self):
        return len(self.__list)
