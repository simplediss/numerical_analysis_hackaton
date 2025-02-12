import numpy as np

def trapezoidal_rule(f, a, b, n):

    h = (b - a) / n
    T = f(a) + f(b)
    integral = 0.5 * T  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)

    integral *= h

    return integral


if __name__ == '__main__':
    f = lambda x:(x * np.exp(-x**2 + 5*x)) * (2*x**2 - 3*x - 5)
    # Integration range
    a, b = 0.5, 1
    n = 10

    print('\n\nTrapezoidal Rule\n')
    print(f"Division into n={n} sections")
    integral = trapezoidal_rule(f, a, b, n)
    print(f"Numerical Integration of definite integral in range [{a},{b}] is {integral}\n")