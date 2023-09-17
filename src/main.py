import function_preprocess
from optimizers import simple_walker, simulated_annealing, gradient_descent, gradient_descent_mom,gradient_descent_sched
import visualization

"""
This component will serve as the main I/O point, we want to call all stdin/out/err from here, while also trying to setup the file reads/writes from here too.

It also serves as the main entry point into the program, which will contain the main loop.
"""

def main_loop():

    landscape_eq_str = input("Enter an equation with parameter x. (e.g., 2*x**2 + 3*x - 5) \n-> ")
    optimizer_choice = input("\nSelect an Optimizer by the number [1.Hill-Climber, 2.Simulated Annealing, 3.Gradient Descent, 4.Gradient Descent w/ Momentum, 5.Gradient Descent w/ Momentum & LR Scheduling]\n-> ")
    
    print("\nProcessing equation...\n")

    equation, function, derivative = function_preprocess.process_function_str(landscape_eq_str)

    try:
        optimizer_choice = int(optimizer_choice)
    except:
        print("Invalid choice, exiting...")
        exit(1)
    
    optimizer = None
    # Create a switch statement to choose the optimizer
    match optimizer_choice:
        case 1:
            optimizer = simple_walker.SimpleWalker(function)
        case 2:
            optimizer = simulated_annealing.SimulatedAnnealing(function)
        case 3:
            optimizer = gradient_descent.GradientDescent(function, derivative)
        case 4:
            optimizer = gradient_descent_mom.GradientDescentWMomentum(function, derivative)
        case 5:
            optimizer = gradient_descent_sched.GradientDescentWSchedule(function, derivative)
        case _:
            print("Invalid choice, exiting...")
            exit(1)

    agg_x = []
    while True:
        try:
            agg_x = optimizer.generate_solution()

            if(agg_x == None or len(agg_x) == 0):
                continue
        except:
            continue
        
        break
    
    visualization.plot(optimizer.name, equation, function, derivative, agg_x)



if __name__=="__main__":
    main_loop()