import numpy as np


def newton_raphson(f, df, p0, TOL, max_iter):
    print("{:<10} {:<15} {:<15} ".format("Iteration", "po", "p1"))
    for i in range(max_iter):
        if df(p0) == 0:
            print( "Derivative is zero at p0, method cannot continue.")
            return

        p = p0 - f(p0) / df(p0)

        if abs(p - p0) < TOL:
            return p  # Procedure completed successfully
        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, p0, p))
        p0 = p
    return p


if __name__ == '__main__':
    f = lambda x: np.sin(2*x**3 + 5*x**2 - 6) / (2*np.exp(-2*x))
    df = lambda x: (6*x**2*np.cos(2*x**3 + 5*x**2 - 6) + 10*x*np.cos(2*x**3 + 5*x**2 - 6)) / (2*np.exp(-2*x)) \
           + (2*np.sin(2*x**3 + 5*x**2 - 6) * np.exp(-2*x))
    p0 = -1
    TOL = 1e-6
    max_iter = 100
    roots = newton_raphson(f, df,p0, TOL ,max_iter)
    print("\nThe equation f(x) has an approximate root at x = {:<15.16f} ".format(roots))