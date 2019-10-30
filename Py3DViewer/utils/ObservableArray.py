import numpy as np
from .Observable import Subject

class ObservableArray(np.ndarray, Subject):
    
    def __init__(self, *args, **kwargs):
        Subject.__init__(self)
        np.ndarray.__init__(self)
       
    def _notify(self, to_return):
        """
        if hasattr(to_return, "_observers") and hasattr(self, "_observers"):
            Subject._notify()
        else:
            print("Error! No observers found")
        """
        Subject._notify(self)
        
    def __getitem__(self, index):
        to_return = super(ObservableArray, self).__getitem__(index)
        if hasattr(to_return, "_observers") and hasattr(self, "_observers"):
            to_return._observers = self._observers
        return to_return
    
    def __repr__(self):
        to_return = repr(np.asarray(self))
        return to_return
        
    def __iadd__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__iadd__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __isub__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__isub__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __imul__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__imul__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __idiv__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__idiv__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __itruediv__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__itruediv__(*args,
                                                        **kwargs)
        self._notify(to_return)
        return to_return
    
    def __rmatmul__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__rmatmul__(*args,
                                                       **kwargs)
        self._notify(to_return)
        return to_return
    
    def __matmul__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__matmul__(*args,
                                                       **kwargs)
        self._notify(to_return)
        return to_return
    
    def __imatmul__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__imatmul__(*args,
                                                       **kwargs)
        self._notify(to_return)
        return to_return
    
    def __ipow__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__ipow__(*args, **kwargs)
        self._notify(to_return)
        return to_return
    
    def __imod__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__imod__(*args, **kwargs)
        self._notify(to_return)
        return to_return
        
    def __ifloordiv__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__ifloordiv__(*args,
                                                         **kwargs)
        self._notify(to_return)
        return to_return
    
    def __ilshift__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__ilshift__(*args,
                                                       **kwargs)
        self._notify(to_return)
        return to_return
    
    def __irshift__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__irshift__(*args,
                                                       **kwargs)
        self._notify(to_return)
        return to_return
    
    def __iand__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__iand__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __ixor__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__ixor__(*args,
                                                    **kwargs)
        self._notify(to_return)
        return to_return
    
    def __ior__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__ior__(*args,
                                                   **kwargs)
        self._notify(to_return)
        return to_return
        
    def __setitem__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__setitem__(*args,
                                                **kwargs)
        self._notify(to_return)
        return to_return
    
    def __setslice__(self, *args, **kwargs):
        to_return = super(self.__class__, self).__setslice__(*args,
                                                 **kwargs)
        self._notify(to_return)
        return to_return

