import function_preprocess
import optimizer_interface
import solution_preparation


"""
This component will serve as the main I/O point, we want to call all stdin/out/err from here, while also trying to setup the file reads/writes from here too.

It also serves as the main entry point into the program, which will contain the main loop.
"""

def main_loop():

    landscape_eq_str = input("Enter an equation with parameter x. (e.g., 2*x**2 + 3*x - 5): ")
    optimizer_choice = input("Select an Optimizer by the number: 1.Hill-Climber ")
    function = function_preprocess.process_function_str(landscape_eq_str)
    values = function_preprocess.generate_datapoints(function)

    if optimizer_choice == 1:
        res_i, res_data = optimizer_interface.MyOptimizer.simple_walk(values)
    else:
        pass






if __name__=="__main__":
    main_loop()