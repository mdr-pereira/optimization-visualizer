import function_preprocess
from optimizers import simple_walker, simulated_annealing
import solution_preparation
import visualization
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
This component will serve as the main I/O point, we want to call all stdin/out/err from here, while also trying to setup the file reads/writes from here too.

It also serves as the main entry point into the program, which will contain the main loop.
"""

def main_loop():

    landscape_eq_str = input("Enter an equation with parameter x. (e.g., 2*x**2 + 3*x - 5): ")
    optimizer_choice = input("Select an Optimizer by the number: 1.Hill-Climber 2.Simulated Annealing \n-> ")
    function = function_preprocess.process_function_str(landscape_eq_str)
    data = function_preprocess.generate_datapoints(function,0.1)
    res_i = []
    res_data = []

    try:
        optimizer_choice = int(optimizer_choice)
    except:
        print("Invalid choice (1), exiting...")
        exit(1)

    scaler = StandardScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1,1)).reshape(-1)
    
    optimizer = None
    # Create a switch statement to choose the optimizer
    match optimizer_choice:
        case 1:
            optimizer = simple_walker.SimpleWalker(data)
        case 2:
            optimizer = simulated_annealing.SimulatedAnnealing(data)
        case _:
            print("Invalid choice, exiting...")
            exit(1)

    

    res_i, res_data = optimizer.generate_solution()

    

    visualization.plot(res_i,res_data,data)



if __name__=="__main__":
    main_loop()