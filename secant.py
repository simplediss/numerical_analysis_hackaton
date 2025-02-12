import numpy as np


def secant_method(f, x0, x1, TOL = 1e-6, max_iter = 50):
    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "xo", "x1", "p"))
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            print( " method cannot continue.")
            return

        p = x0 - f(x0) * ((x1 - x0) / (f(x1) - f(x0)))

        if abs(p - x1) < TOL:
            return p  # Procedure completed successfully
        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(i, x0, x1,p))
        x0 = x1
        x1 = p
    return p


if __name__ == '__main__':
    f = lambda x: x**2 - 5*x +2
    x0 = 80
    x1 = 100
    TOL = 1e-6
    max_iter = 20
    roots = secant_method(f, x0, x1, TOL, max_iter)
    print( f"\n The equation f(x) has an approximate root at x = {roots}")


    func = lambda x: np.sin(x*x+5*x+6)/2*np.exp(-x)
    func2 = lambda x: np.cos(x**3 + 5*x**2 - 6) / 2*np.exp(-2*x)
    print(secant_method(func,-3,1,10))
    print(secant_method(func2,1,1.5,10))