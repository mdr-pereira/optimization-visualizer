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
    def generate_solution(data: List[float32]) -> List[Tuple[int, float32]]:
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def _step(data: List[float32], i: int) -> Tuple[int, float32]:
        raise NotImplementedError
    
    




class MyOptimizer(OptimizerInterface):
    def generate_solution(data: List[float]) -> List[Tuple[int, float]]:
        # Implement how to generate a solution based on your problem
        pass

    def step(data: List[float], i: int) -> Tuple[int, float]:
        # Implement a single optimization step
        pass

    def calculate_energy(solution: List[Tuple[int, float]]) -> float:
        # Implement how to calculate the energy (objective value) of a solution
        pass

    def simple_walk(data: List[np.float32]):
        res_i = []
        res_data = []
        MAX_I = len(data)
        i = rand.randint(0, MAX_I)

        while True:
            cur_v = data[i]

            tmp = data[i + 1] if i == 0 else (data[i - 1] if i == (MAX_I - 1) else min(data[i - 1], data[i + 1]))

            tmp = min(cur_v, tmp)

            if tmp == cur_v:
                break

            if tmp == data[i - 1]:
                i -= 1
            else:
                i += 1

            res_i.append(i)
            res_data.append(tmp)

        return res_i, res_data

