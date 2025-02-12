import numpy as np
import matplotlib.pyplot as plt
#import scipy as sci

from secant import secant_method
from bisection_method import bisection_method
from newton_raphson import newton_raphson

from trapezoidal_method import trapezoidal_rule
from simpsons_method import simpsons_rule

from matrix_util import gaussian_elimination
from gauss_seidel import gauss_seidel

from interpolations import linear, polynomial

CONST_1 = 750
CONST_2 = 87
CONST_3 = 325
CONST_4 = 85


def Q1() -> float:   #1.4053
    # the bigger (of the n-roots) real root at domain [-1, 1.5]
    f1 = lambda x: np.sin(2*x**3 + 5*x**2 - 6) / (2*np.exp(-2*x))
    # df1 = lambda x: (6*x**2*np.cos(2*x**3 + 5*x**2 - 6) + 10*x*np.cos(2*x**3 + 5*x**2 - 6)) / (2*np.exp(-2*x)) \+ (2*np.sin(2*x**3 + 5*x**2 - 6) * np.exp(-2*x))
    df1 = lambda x: np.exp(2 * x) * (np.sin(2 * x**3 + 5*x**2 - 6) + (3 * x**2 + 5 * x) * np.cos(2*x**3 + 5*x**2 - 6))

    q1_newton_raphson = newton_raphson(f1, df1, p0 = 1.4, TOL = 1e-6, max_iter = 100)
    q1_secant = secant_method(f1, x0 = -1, x1 = 1.5, TOL = 0.000_000_1)
    print(f'\n\nx1 newton raphson = {q1_newton_raphson}\n')
    print(f'x1 secant    = {q1_secant}\n')

    q1_x = max(q1_newton_raphson, q1_secant)
    return q1_x * CONST_1



def Q2() -> float:  
    # the value of the integral at integration range [0.5, 1]
    f = lambda x: (x * np.exp(-x**2 + 5*x)) * (2*x**2 - 3*x - 5)
    a, b = 0.5, 1
    n = 10  # Must be even for simpson's rule

    q2_simpsons = simpsons_rule(f, a, b, n)
    q2_trapezoidal = trapezoidal_rule(f, a, b, n)
    print(f'\n\nx2 simpsons = {q2_simpsons}\n')
    print(f'x2 trapezoidal = {q2_trapezoidal}\n')

    q2_x = q2_simpsons
    return q2_x * CONST_2


def Q3() -> float:  # [1.8276, 0.6552, 1.9655]
    # the value of the integral at integration range [0.5, 1]
    A = np.array([
        [0.04, 0.01, -0.01],
        [0.2 , 0.5 , -0.2 ],
        [1   , 2   , 4]
    ])
    b = np.array([0.06, 0.3, 11])
    q3_gaussian_elimination = gaussian_elimination(A, b)
    q3_gauss_seidel = gauss_seidel(A, b, np.zeros_like(b))
    print(f'\n\nx3 gauss elim.  = {q3_gaussian_elimination}\n')
    print(f'x3 gauss seidel = {q3_gauss_seidel}\n')

    q3_x = q3_gaussian_elimination[0]
    return q3_x * CONST_3

def Q4() -> float:
    points = [(1.3, 3.6984), (1.4, 3.9043)]
    # x_points = [1.2, 1.3, 1.4, 1.5, 1.6, 3.5095, 3.6984, 3.9043, 4.1295, 4.3756]
    # y_points = [3.5095, 3.6984, 3.9043, 4.1295, 4.3756]
    x_target = 1.37

    q4_linear_y_target = linear(points, x_target)
    q4_polynomial_y_target = polynomial(points, x_target)
    print(f'\n\nx4 linear = {q4_linear_y_target}\n')
    print(f'x4 polynomial = {q4_polynomial_y_target}\n')

    q4_x = 1
    return q4_x * CONST_4






def rounding_function(x: float) -> int:
    return int(np.abs(np.round(x)))

def formula1(L: int) -> int:
    """Defects as a function of Lines of code"""
    L = rounding_function(L)
    D = 4.86 + 0.018*L
    return D

def formula2(V: int) -> int:
    """Defects as a function of Volume metric (language dependent)"""
    V = rounding_function(V)
    D = V / 3000
    return D

def formula3(L: int) -> int:
    """Defects as a function of Lines of code (language LOC dependent)"""
    L = rounding_function(L)
    A = np.array([0.0047, 0.0023, 0.000_043])  # Fortran constants
    D = L*A[0] + L*A[1]*np.log(L) + L*A[2]*np.log(L)**2
    return D

def formula4(L: int) -> int:
    """Defects as a function of Lines of code"""
    L = rounding_function(L)
    D = 4.2 + 0.0015*L**(4/3)
    return D

def formula5(L: int) -> int:
    """Defects as a function of Lines of code"""
    L = rounding_function(L)
    D = 0.069 + 0.00156*L + 0.000_000_47*L**2
    return D



if __name__ == '__main__':
    q1_x = Q1(),
    q2_x = Q2(),
    q3_x = Q3(),
    q4_x = Q4(),
    
    formula_x = [
        q1_x,
        q2_x,
        q3_x,
        q4_x
    ]

    formula_x.sort()
    # plt.xticks(formula_x, ['#1', '#2', '#3', '#4'])

    # find `y` values for each of the heuristics using `x` values
    formula_y1 = [formula1(x) for x in formula_x]
    formula_y2 = [formula2(x) for x in formula_x]
    formula_y3 = [formula3(x) for x in formula_x]
    formula_y4 = [formula4(x) for x in formula_x]
    formula_y5 = [formula5(x) for x in formula_x]

    # plot the results on a single graph
    plt.plot(formula_x, formula_y1, label='f1')
    plt.plot(formula_x, formula_y2, label='f2')
    plt.plot(formula_x, formula_y3, label='f3')
    plt.plot(formula_x, formula_y4, label='f4')
    plt.plot(formula_x, formula_y5, label='f5')

    plt.xlabel('results from questions')
    plt.ylabel('formulas values')
    plt.title('formulas as functions of results | D(L)')

    plt.legend()
    plt.savefig('./plot.png')
    plt.show()



    # formula_x.sort()  # sort `x` values in ascending order
    # plt.xticks(formula_x, ['#1', '#2', '#3'])

    

    # x_values = np.linspace(-2, 2, 500)
    # y_values =  Q1(x_values)

    # plt.plot(x_values, y_values, label='f(x)')
    # plt.axhline(0, color='black', linestyle='--')  # קו האפס
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.grid()
    # # plt.show()





