import abc
from typing import List, Tuple
from numpy import float32
"""
Interface for the optimizer classes, will allow us to standardize the call API and outputs.
"""

class OptimizerInterface(abc.ABC):

    @abc.abstractmethod
    def generate_solution(data: List[float32]) -> List(Tuple(int, float32)):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def _step(data: List[float32], i: int) -> Tuple(int, float32):
        raise NotImplementedError
