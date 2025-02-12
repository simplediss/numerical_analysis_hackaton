import numpy as np


def max_steps(a, b, err):
    s = int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))
    return s

def bisection_method(f, a, b, tol=1e-6):
    c, k = 0, 0
    steps = max_steps(a, b, tol)  # calculate the max steps possible

    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    # while the diff af a&b is not smaller than tol, and k is not greater than the max possible steps
    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2  # Calculation of the middle value

        if f(c) == 0 :
            return c  # Procedure completed successfully

        if f(c) * f(a) < 0:  # if sign changed between steps
            b = c  # move forward
        else:
            a = c  # move backward

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
        k += 1

    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")
    
    return c  # return the current root



if __name__ == '__main__':
    func1 = lambda x: np.sin(2*x**3 + 5*x**2 - 6) / ((2*np.exp(-2*x)))
    root1 = bisection_method(func1, -1, 1.5)
    print(f"\nThe equation f(x) has an approximate root at x = {root1}")