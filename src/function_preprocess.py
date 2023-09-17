from typing import Callable
import sympy as sp
import numpy as np

from constants import XBOUND_MAX, XBOUND_MIN
"""
We want to both receive and parse the function string over here, treating all possible errors it throws, additionally with the whole 2D/3D logic set eventually.

Additionally, it should also provide functions to enable us to produce the dataset of the landscape function, with arbitrary interval values.
"""

def process_function_str(equation_str: str) -> Callable[[np.float32], np.float32]:
    smbl_x = sp.symbols('x', real=True)

    # Parse the input string to create a function
    try:
        equation = sp.sympify(equation_str, locals={'x': smbl_x})
        
        derivative = sp.lambdify(smbl_x, sp.diff(equation, smbl_x))

        function = sp.lambdify(smbl_x, equation, "numpy")
    except sp.SympifyError:
        print("Invalid input. Please enter a valid polynomial equation.")
        exit(1)

    return equation, function, derivative

def generate_datapoints(function: Callable[[float], float], step: float = 1.0):
    temp = np.arange(XBOUND_MIN, XBOUND_MAX, step, dtype=np.float32)

    return list(map(function, temp))