from optimizers.optimizer_interface import OptimizerInterface
from typing import List, Tuple
import random as rand

class SimpleWalker(OptimizerInterface):

    def __init__(self, dataset) -> None:
        self.MAX_I = len(dataset)
        self.dataset = dataset


    def generate_solution(self) -> Tuple[List[int], List[float]]:
        res_i = []
        res_data = []

        res = self.step(rand.randint(0, self.MAX_I))

        while res != None:
            i, data = res

            res_i.append(i)
            res_data.append(data)

            res = self.step(i)

        return res_i, res_data
    

    def step(self, i: int) -> Tuple[int, float] or None:
        data = self.dataset

        cur_v = data[i]

        tmp = data[i + 1] if i == 0 else (data[i - 1] if i == (self.MAX_I - 1) else min(data[i - 1], data[i + 1]))

        tmp = min(cur_v, tmp)

        if tmp == cur_v:
            return None

        if tmp == data[i - 1]:
            i -= 1
        else:
            i += 1

        return (i, tmp)
