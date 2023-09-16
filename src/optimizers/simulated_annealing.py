from optimizers.optimizer_interface import OptimizerInterface
from typing import List, Tuple
import random as rand

class SimulatedAnnealing(OptimizerInterface):

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.MAX_I = len(dataset)
        self.MIN_I = 0

        self.temp = 100
        self.max_steps = 1000
        self.MAX_PATIENCE = 10
        self.patience = self.MAX_PATIENCE

    def generate_solution(self) -> Tuple[List[int], List[float]]:
        res_i = []
        res_data = []

        res = self.step(rand.randint(self.MIN_I, self.MAX_I))
        steps = self.max_steps

        while res != None and steps > 0:
            i, data = res

            res_i.append(i)
            res_data.append(data)

            res = self.step(i)

            steps -= 1
            self.temp *= 0.99

        return res_i, res_data
    
    def _get_P(self, cur_v: float, next_v: float) -> float:
        return next_v-cur_v/self.temp
    
    def step(self, i: int) -> Tuple[int, float] or None:
        data = self.dataset
    
        next_i = rand.choice([i-1, i+1])
        cur_v = data[i]
        next_v = data[next_i]

        if next_i < self.MIN_I or next_i > self.MAX_I:
            return (i, cur_v)
        
        p = self._get_P(cur_v, next_v)

        if (next_v < cur_v) or (rand.random() > p):
            self.patience = self.MAX_PATIENCE
            return (next_i, next_v)
        
        self.patience -= 1
        if(self.patience == 0):
            return None

        return (i, cur_v)