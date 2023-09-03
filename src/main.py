"""
This component will serve as the main I/O point, we want to call all stdin/out/err from here, while also trying to setup the file reads/writes from here too.

It also serves as the main entry point into the program, which will contain the main loop.
"""

def main_loop():

    landscape_eq_str = input("Enter an equation with parameter x. (e.g., 2*x**2 + 3*x - 5): ")


    print("A")

if __name__=="__main__":
    main_loop()