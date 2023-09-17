import abc
from typing import List, Tuple
from numpy import float32
from typing import Callable, List
import random as rand
import numpy as np

"""
Interface for the optimizer classes, will allow us to standardize the call API and outputs.
"""

class OptimizerInterface(abc.ABC):

    @abc.abstractmethod
    def generate_solution() -> Tuple[List[int], List[float]]:
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def step(self, i: int) -> Tuple[int, float32]:
        raise NotImplementedError
    
