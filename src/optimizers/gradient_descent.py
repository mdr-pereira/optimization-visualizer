from optimizers.optimizer_interface import OptimizerInterface
import sympy as sp
import constants as const
import random as rand

class GradientDescent(OptimizerInterface):

    def __init__(self, function: sp.Function, derivative:sp.Function, lr=0.01):
        self.name = "Gradient Descent"

        self.function = function 

        self.derivative = derivative

        self.lr = lr
        
    def generate_solution(self, max_iter=1000, epsilon=0.0001):
        x_agg = []

        x = rand.randint(const.XBOUND_MIN, const.XBOUND_MAX)

        for _ in range(max_iter):
            cur_v = self.function(x)

            x = self.step(x)
            next_v = self.function(x)

            if (x < const.XBOUND_MIN) or (x > const.XBOUND_MAX):
                break
            
            if abs(cur_v - next_v) < epsilon:
                break

            x_agg.append(x)

        return x_agg


    def step(self, x):
        return x - (self.lr * self.derivative(x))