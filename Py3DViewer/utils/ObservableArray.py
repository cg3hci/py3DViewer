import numpy as np
from .Observable import Subject

class ObservableArray(np.ndarray, Subject):
    
    def __init__(self, *args, **kwargs):
        Subject.__init__(self)
        np.ndarray.__init__(self)
       
    def __getitem__(self, index):
        to_return = super(ObservableArray, self).__getitem__(index)
        to_return._observers = self._observers
        return to_return
    
    def __repr__(self):
        to_return = repr(np.asarray(self))
        return to_return
        
    def __iadd__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__iadd__(*args,
                                                    **kwargs)
    def __isub__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__isub__(*args,
                                                    **kwargs)
    def __imul__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__imul__(*args,
                                                    **kwargs)
    def __idiv__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__idiv__(*args,
                                                    **kwargs)
    def __itruediv__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__itruediv__(*args,
                                                        **kwargs)
    def __imatmul__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__imatmul__(*args,
                                                       **kwargs)
    def __ipow__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__ipow__(*args, **kwargs)
    def __imod__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__imod__(*args, **kwargs)
    def __ifloordiv__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__ifloordiv__(*args,
                                                         **kwargs)
    def __ilshift__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__ilshift__(*args,
                                                       **kwargs)
    def __irshift__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__irshift__(*args,
                                                       **kwargs)
    def __iand__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__iand__(*args,
                                                    **kwargs)
    def __ixor__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__ixor__(*args,
                                                    **kwargs)
    def __ior__(self, *args, **kwargs):
        self._notify()
        return super(self.__class__, self).__ior__(*args,
                                                   **kwargs)
    def __setitem__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__setitem__(*args,
                                                **kwargs)
        self._notify()
        return to_return
    
    def __setslice__(self, *args, **kwargs):
        self._notify()
        print("Setting slice")
        return super(self.__class__, self).__setslice__(*args,
                                                 **kwargs)

