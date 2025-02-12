import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

def simpsons_rule(f, a, b, n):
    """
    Simpson's Rule for Numerical Integration

    Parameters:
    f (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals (must be even).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's Rule.")

    h = (b - a) / n
    integral = f(a) + f(b)  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x_i)
        else:
            integral += 4 * f(x_i)

    integral *= h / 3
    return integral

if __name__ == '__main__':
    f = lambda x: (x * np.exp(-x**2 + 5*x)) * (2*x**2 - 3*x - 5)
    # Integration range
    a, b = 0.5, 1
    n = 10  # Must be even


    print('\n\nSimpson`s Rule\n')
    print(f"Division into n={n} sections")
    integral = simpsons_rule(f, a, b, n)
    print(f"Numerical Integration of definite integral in range [{a},{b}] is {integral}\n")

    # # Plot the function
    # x_vals = np.linspace(a, b, 100)
    # y_vals = f(x_vals)

    # plt.plot(x_vals, y_vals, label=r'$e^{x^2}$')
    # plt.fill_between(x_vals, y_vals, alpha=0.3, color='orange')
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.title("Function Visualization for Simpson's Rule Integration")
    # plt.legend()
    # plt.grid()
    # plt.show()
