import numpy as np
from solve_linear_system import solve_linear_system


def linear(points_table: list[tuple[float, float]], x: float) -> float:
    """
    Perform linear interpolation to estimate the y-value corresponding to a given x-value.

    Args:
        points_table (list[tuple[float, float]]): A list of tuples representing known points (x, y) for interpolation.
        x (float): The x-value for which to estimate the corresponding y-value.

    Returns:
        float: The estimated y-value corresponding to the provided x-value using linear interpolation.
    """

    def get_adjacent_points(i: int) -> tuple[float, float, float, float]:
        """
        Retrieve the adjacent points (x1, x2, y1, y2) from the points table based on the given index.

        Args:
            i (int): The index of the first point in the points table.

        Returns:
            tuple[float, float, float, float]: The x and y coordinates of the adjacent points (x1, x2, y1, y2).
        """
        x1 = points_table[i + 0][0]
        x2 = points_table[i + 1][0]
        y1 = points_table[i + 0][1]
        y2 = points_table[i + 1][1]
        return x1, x2, y1, y2
    
    def get_slope(x1: float, x2: float, y1: float, y2: float) -> float:
        """
        Calculate the slope of the line connecting two points.

        Args:
            x1 (float): The x-coordinate of the first point.
            x2 (float): The x-coordinate of the second point.
            y1 (float): The y-coordinate of the first point.
            y2 (float): The y-coordinate of the second point.

        Returns:
            float: The slope of the line connecting the two points.
        """
        return (y1 - y2) / (x1 - x2)

    for i in range(len(points_table) - 1):
        x1, x2, y1, y2 = get_adjacent_points(i)
        if x1 <= x <= x2:
            m = get_slope(x1, x2, y1, y2)
            return (m * x) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
    
    x1, x2, y1, y2 = get_adjacent_points(0)
    m = get_slope(x1, x2, y1, y2)
    return y1 + m * (x - x1)


def polynomial(table_points: list[tuple[float, float]], x: float) -> float:
    """
    Perform polynomial interpolation to estimate the y-value corresponding to a given x-value.

    Args:
        table_points (list[tuple[float, float]]): A list of tuples representing known points (x, y) for interpolation.
        x (float): The x-value for which to estimate the corresponding y-value.

    Returns:
        float: The estimated y-value corresponding to the provided x-value using polynomial interpolation.
    """
    A = [[point[0] ** i for i in range(len(table_points))] for point in table_points]
    b = [[point[1]] for point in table_points]
    S = solve_linear_system(np.array(A), np.array(b))
    return sum([S[i] * (x ** i) for i in range(len(S))])


def lagrange(table_points: list[tuple[float, float]], x: float) -> float:
    """
    Perform Lagrange interpolation to estimate the y-value corresponding to a given x-value.

    Args:
        table_points (list[tuple[float, float]]): A list of tuples representing known points (x, y) for interpolation.
        x (float): The x-value for which to estimate the corresponding y-value.

    Returns:
        float: The estimated y-value corresponding to the provided x-value using Lagrange interpolation.
    """
    x_data = [x for x, y in table_points]
    y_data = [y for x, y in table_points]

    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result



if __name__ == '__main__':
    """
    Compare the accuracy of different interpolation methods (linear, Lagrange, and polynomial) for estimating the y-value
    at a given x-value and print the results.

    A demonstration of linear interpolation at play in Desmos: https://www.desmos.com/calculator/rrepmm4c3m
    """
    
    def f(x: float) -> float:
        """
        Define the function for which to generate test points.

        Args:
            x (float): The x-value.

        Returns:
            float: The corresponding y-value for the function.
        """
        return x**3 - 4 * x**2 + 7 * x - 4
    
    def compare(points: list[tuple[float, float]], x: float, method) -> None:
        """
        Compare the actual function value to the interpolated value and print the difference.

        Args:
            points (list[tuple[float, float]]): A list of tuples representing known points (x, y).
            x (float): The x-value for which to estimate the corresponding y-value.
            method (function): The interpolation method to use (linear, Lagrange, or polynomial).
        """
        act = f(x)
        aprx = method(points, x)
        diff = np.abs(act - aprx)
        print(f'actual: {act}  |  interpolated: {aprx}  | difference: {diff}')

    x_vals = [3, 5, 7, 9, 11, 13, 15]
    points = [(x, f(x)) for x in x_vals]

    compare(points, 4, linear)
    compare(points, 4, lagrange)
    compare(points, 4, polynomial)