from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def commit_metrics(self):
        pass
