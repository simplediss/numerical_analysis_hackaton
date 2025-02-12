import numpy as np


def print_matrix(matrix: np.ndarray, width: int = None) -> None:
    """
    Prints the given matrix with elements aligned for better readability.
    
    Parameters:
    matrix (np.ndarray): The matrix to be printed.
    width (int, optional): The reserved width for values (excludes the sign).
    
    Returns:
    None
    """
    max_width = max(len(str(element)) for row in matrix for element in row)
    width = max_width if width is None else width
    width = max(1, width)
    for row in matrix:
        print("  ".join(f"{('-' if np.sign(element) == -1 else ' ') + str(np.abs(element))[:width]:>{width}}" for element in row))


def elementary_matrix_for_row_addition(dim: int, target_row: int, addend_row: int, scalar: float = 1.0) -> np.ndarray:
    """
    Create an elementary matrix that adds a multiple of one row to another.

    Parameters:
    dim (int): Dimension of the square matrix.
    target_row (int): Index of the row to be modified.
    addend_row (int): Index of the row to be added.
    scalar (float, optional): Scalar multiple of the addend row. Defaults to 1.0.

    Returns:
    np.ndarray: The elementary matrix for the row addition operation.

    Raises:
    ValueError: If row indices are invalid or if target_row and addend_row are the same.
    """
    if target_row < 0 or addend_row < 0 or target_row >= dim or addend_row >= dim:
        raise ValueError("Invalid row indices.")

    if target_row == addend_row:
        raise ValueError("Source and target rows cannot be the same.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[target_row, addend_row] = scalar

    return np.array(elementary_matrix)


def get_dim_of_square_matrix(matrix: np.ndarray) -> int:
    """
    Get the dimension of a square matrix.

    Parameters:
    matrix (np.ndarray): The matrix whose dimension is to be retrieved.

    Returns:
    int: The dimension of the square matrix.
    """
    return matrix.shape[0]
def is_square(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is square.

    Parameters:
    matrix (np.ndarray): The matrix to check.

    Returns:
    bool: True if the matrix is square, False otherwise.
    """
    return matrix.shape[0] == matrix.shape[1]


def lu_decomposion(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs LU decomposition on a given square matrix.

    Parameters:
    matrix (np.ndarray): The matrix to decompose.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the lower triangular matrix (L) and the upper triangular matrix (U).

    Raises:
    ValueError: If the matrix is singular and LU decomposition cannot be performed.
    """
    # https://www.youtube.com/watch?v=BFYFkn-eOQk
    dim = get_dim_of_square_matrix(matrix)
    lower_matrix = np.identity(dim)
    upper_matrix = matrix.copy()

    # iterate over all rows
    for pivot_index in range(dim):
        # check whether the given matrix is singular
        if upper_matrix[pivot_index][pivot_index] == 0:
            raise ValueError("can't perform LU Decomposition")

        # reduce rows under the current row
        for i in range(pivot_index + 1, dim):
            ratio = -( upper_matrix[i][pivot_index] / upper_matrix[pivot_index][pivot_index] )
            elementary_matrix = elementary_matrix_for_row_addition(dim, i, pivot_index, ratio)
            elementary_inverse = np.linalg.inv(elementary_matrix)
            lower_matrix = np.dot(lower_matrix, elementary_inverse)
            upper_matrix = np.dot(elementary_matrix, upper_matrix)

    return lower_matrix, upper_matrix


def backward_substitution(augmented_matrix: np.ndarray) -> np.ndarray:
    """
    Solves an upper triangular system of linear equations using backward substitution.

    Parameters:
    augmented_matrix (np.ndarray): The augmented matrix representing the upper triangular system.

    Returns:
    np.ndarray: The solution vector.
    """
    rows = augmented_matrix.shape[0]
    solution_vector = np.zeros(rows)

    # iterate over each row (equation) in reverse order
    for i in range(rows - 1, -1, -1):
        # start with the last element of the current row
        solution_vector[i] = augmented_matrix[i][rows]

        # adjust the solution vector with accordance to solved variables
        for j in range(i + 1, rows):
            solution_vector[i] -= augmented_matrix[i][j] * solution_vector[j]

        # divide by the coefficient of the variable to solve for it
        solution_vector[i] = solution_vector[i] / augmented_matrix[i][i]

    return solution_vector


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a system of linear equations Ax = b using LU decomposition.

    Parameters:
    A (np.ndarray): The coefficient matrix.
    b (np.ndarray): The right-hand side vector.

    Returns:
    np.ndarray: The solution vector x.
    
    Raises:
    Exception: If matrix A is not square or if dimensions of A and b do not match.
    """
    # Check if the matrix A is square
    if not is_square(A):
        raise Exception('Matrix A must be square.')
    
    # Check if the dimensions of A and b match
    if get_dim_of_square_matrix(A) != b.shape[0]:
        raise Exception('Dimensions of matrix A and vector b must match.')
    
    # create the augmented matrix via concatenating A & b
    augmented_matrix = np.c_[A, b]

    # perform LU decomposition to get the row-echelon form of the augmented matrix
    lower, upper = lu_decomposion(augmented_matrix)

    # perform backward substitution and return the solution vector
    return backward_substitution(upper)



