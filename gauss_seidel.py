import numpy as np
from matrix_util import is_diagonally_dominant, print_matrix, get_dim_of_square_matrix


def jacobi_iterative(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, tolerance: float = 1e-16, max_iterations: int = 200) -> np.ndarray:
    """
    Solves the linear system Ax = b using the Jacobi iterative method.

    Parameters:
    A (np.ndarray): Coefficient matrix (must be square and diagonally dominant).
    b (np.ndarray): Constant terms vector.
    x_0 (np.ndarray): Initial guess for the solution vector.
    tolerance (float): Convergence tolerance for the stopping criterion (default: 1e-16).
    max_iterations (int): Maximum number of iterations allowed (default: 200).

    Returns:
    np.ndarray: Solution vector x if the method converges within the maximum number of iterations, otherwise the best reached guess.
    """

    # Get the dimension of the square matrix A
    dim = get_dim_of_square_matrix(A)
    iteration = 1

    # Check if the matrix A is diagonally dominant
    if not is_diagonally_dominant(A):
        print('Matrix is not diagonally dominant')
        print_matrix(A, 5)
        print()
        return x_0

    # Print header for iterations
    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    # Start the iteration process
    while iteration <= max_iterations:
        # Initialize the new solution vector with zeros
        x = np.zeros(dim, dtype=np.double)

        # Update each element of the solution vector
        for i in range(dim):
            sigma = 0
            for j in range(dim):
                if j != i:
                    sigma += A[i][j] * x_0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        # Print the current iteration and solution vector
        print("{:<15} ".format(iteration) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        # Check for convergence
        if np.linalg.norm(x - x_0, np.inf) < tolerance:
            return x

        # Update for the next iteration
        iteration += 1
        x_0 = x.copy()

    # If the maximum number of iterations is exceeded
    print("Maximum number of iterations exceeded")
    return x


def gauss_seidel(A: np.ndarray, b: np.ndarray, x_0: np.ndarray, tolerance: float = 1e-16, max_iterations: int = 200) -> np.ndarray:
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    A (np.ndarray): Coefficient matrix (must be square and diagonally dominant).
    b (np.ndarray): Constant terms vector.
    x_0 (np.ndarray): Initial guess for the solution vector.
    tolerance (float): Convergence tolerance for the stopping criterion (default: 1e-16).
    max_iterations (int): Maximum number of iterations allowed (default: 200).

    Returns:
    np.ndarray: Solution vector x if the method converges within the maximum number of iterations, otherwise the best reached guess.
    """

    # Get the dimension of the square matrix A
    dim = get_dim_of_square_matrix(A)
    iteration = 1

    # Check if the matrix A is diagonally dominant
    if not is_diagonally_dominant(A):
        print('Matrix is not diagonally dominant')
        print_matrix(A, 5)
        print()
        return x_0

    # Print header for iterations
    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    
    # Initialize the solution vector with zeros
    x = np.zeros(dim, dtype=np.double)

    # Start the iteration process
    while iteration <= max_iterations:
        # Update each element of the solution vector
        for i in range(dim):
            sigma = 0
            for j in range(dim):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        # Print the current iteration and solution vector
        print("{:<15} ".format(iteration) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        # Check for convergence
        if np.linalg.norm(x - x_0, np.inf) < tolerance:
            return x

        # Update for the next iteration
        iteration += 1
        x_0 = x.copy()

    # If the maximum number of iterations is exceeded
    print("Maximum number of iterations exceeded")
    return x


if __name__ == "__main__":
    # A = np.array([
    #     [3,  1,  1],
    #     [1,  4, -2],
    #     [-1, 1,  5],
    # ])
    # b = np.array([ 5, 5, 5])

    A = np.array([
        [0.04, 0.01, -0.01],
        [0.2 , 0.5 , -0.2 ],
        [1   , 2   , 4]
    ])
    b = np.array([0.06, 0.3, 11])

    x = np.zeros_like(b, dtype=np.double)

    # print("Jacobi method:")
    # solution_jacobi = jacobi_iterative(A, b, x.copy())
    # print("\nApproximate solution:", solution_jacobi.tolist())
    
    print("\n\nGauss-Seidel method:")
    solution_gauss_siedel = gauss_seidel(A, b, x.copy())
    print("\nApproximate solution:", solution_gauss_siedel.tolist())

