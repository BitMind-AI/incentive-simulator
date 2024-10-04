from abc import ABC, abstractmethod


class Reward:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass