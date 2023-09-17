from optimizers.optimizer_interface import OptimizerInterface
from typing import List, Tuple
import random as rand
import constants as const

class SimulatedAnnealing(OptimizerInterface):

    def __init__(self, function) -> None:
        self.name = "Simulated Annealing"
        
        self.function = function

        self.temp = 100
        self.max_steps = 1000
        self.MAX_PATIENCE = 10
        self.patience = self.MAX_PATIENCE

    def generate_solution(self) -> Tuple[List[int], List[float]]:
        agg_x = []

        x = rand.randint(const.XBOUND_MIN, const.XBOUND_MAX)

        while x != None:
            agg_x.append(x)

            x = self.step(x)

            self.temp *= 0.99

        return agg_x
    

    def get_probability(self, cur_v: float, next_v: float) -> float:
        return abs(next_v-cur_v)/self.temp
    
    
    def step(self, x: int, lr=0.1) -> Tuple[int, float] or None:
        cur_v = self.function(x)

        nxt_x = rand.choice([x-lr, x+lr])

        #Checks if the proposed next value is out of bounds. If so, we penalize the patience.
        if(nxt_x < const.XBOUND_MIN or nxt_x > const.XBOUND_MAX):
            self.patience -= 1
            return x
        
        next_v = self.function(nxt_x)
        p = self.get_probability(cur_v, next_v)

        if (next_v < cur_v) or (rand.random() > p):
            self.patience = self.MAX_PATIENCE
            return nxt_x
        
        #If the patience is 0, we return None to stop the loop.
        self.patience -= 1
        if(self.patience == 0):
            return None

        return x