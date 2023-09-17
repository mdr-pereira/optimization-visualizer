from optimizers.optimizer_interface import OptimizerInterface
from typing import List, Tuple
import random as rand
import constants as const

class SimpleWalker(OptimizerInterface):

    def __init__(self, function) -> None:
        self.name = "Simple Walker"
        
        self.function = function


    def generate_solution(self) -> Tuple[List[int], List[float]]:
        agg_x = []

        x = rand.randint(const.XBOUND_MIN, const.XBOUND_MAX)
    
        while x != None:
            agg_x.append(x)

            x = self.step(x)

        return agg_x
    

    def step(self, x: int, lr=0.1) -> Tuple[int, float] or None:
        cur_v = self.function(x)

        #Check if the next value is less than the current, w/in bounds.
        if (self.function(x+lr) < cur_v) and (x+lr < const.XBOUND_MAX):
            return x+lr
        
        #Check if the previous value is less than the current, w/in bounds.
        if (self.function(x-lr) < cur_v) and (x-lr > const.XBOUND_MIN):
            return x-lr
        
        return None
        
