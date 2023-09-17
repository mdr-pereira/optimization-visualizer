from optimizers.optimizer_interface import OptimizerInterface
import sympy as sp 
import constants as const
import random as rand

class GradientDescentWSchedule(OptimizerInterface):


    def __init__(self, function :sp.Function, derivative :sp.Function,  lr_init=0.1, lr_final=0.01, lr_decay=0.0001):
        self.name = "Gradient Descent w/ Momentum and LR Scheduling"

        self.function = function
        self.derivative = derivative
        self.PATIENCE_MAX = 10
        self.patience = self.PATIENCE_MAX

        self.lr_final = lr_final
        self.lr_decay = lr_decay
        self.cur_lr = lr_init
        
    def generate_solution(self, max_iter=1000, epsilon=0.0001):
        x_agg = []

        x = rand.randint(const.XBOUND_MIN, const.XBOUND_MAX)
        v = 0

        for _ in range(max_iter):
            cur_y = self.function(x)

            v = self.get_velocity(x, v)
            x = self.step(x, v)

            next_y = self.function(x)

            if abs(cur_y - next_y) < epsilon:
                break
            
            if (x < const.XBOUND_MIN):
                x = const.XBOUND_MIN
                v = 0
                self.patience -= 1
            elif (x > const.XBOUND_MAX):
                x = const.XBOUND_MAX
                v = 0
                self.patience -= 1
            else:
                self.patience = self.PATIENCE_MAX

            if self.patience == 0:
                break

            x_agg.append(x)

            self.cur_lr = max(self.cur_lr - self.lr_decay, self.lr_final)

        return x_agg
    
    def get_velocity(self, x, v, gamma=0.9):
        return (gamma * v) + (self.cur_lr * self.derivative(x))
    
    def step(self, x, v):
        return x - v