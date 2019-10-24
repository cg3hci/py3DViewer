import abc

"""
This file is sourced from https://sourcemaking.com/design_patterns/observer/python/1 and modified 
"""

class Subject:
    """
    Knows its observers. Any number of Observer objects may observe a
    subject.
    Send a notification to its observers when its state changes.
    """

    def __init__(self):
        self._observers = set()

    def attach(self, observer):
        observer._subject = self
        self._observers.add(observer)

    def detach(self, observer):
        observer._subject = None
        self._observers.discard(observer)

    def _notify(self):
        for observer in self._observers:
            observer.update()


class Observer(metaclass=abc.ABCMeta):
    """
    Defines an updating interface for objects that should be notified of
    changes in a subject.
    """

    def __init__(self):
        self._subject = None
        self._observer_state = None

    @abc.abstractmethod
    def update(self):
        pass


