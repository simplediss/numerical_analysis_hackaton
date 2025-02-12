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


def random_matrix(N: int, M: int, low: int = 0, high: int = 10) -> np.ndarray:
    """
    Generates a random matrix of size N x M with integer values between low and high (inclusive).
    
    Parameters:
    N (int): Number of rows.
    M (int): Number of columns.
    low (int): Minimum value (inclusive) for random integers. Default is 0.
    high (int): Maximum value (inclusive) for random integers. Default is 10.
    
    Returns:
    np.ndarray: N x M matrix with random integers.
    """
    return np.random.randint(low, high + 1, size=(N, M))


def is_square(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is square.

    Parameters:
    matrix (np.ndarray): The matrix to check.

    Returns:
    bool: True if the matrix is square, False otherwise.
    """
    return matrix.shape[0] == matrix.shape[1]


def get_dim_of_square_matrix(matrix: np.ndarray) -> int:
    """
    Get the dimension of a square matrix.

    Parameters:
    matrix (np.ndarray): The matrix whose dimension is to be retrieved.

    Returns:
    int: The dimension of the square matrix.
    """
    return matrix.shape[0]


def determinant(matrix: np.ndarray) -> float:
    """
    Calculates the determinant of a given square matrix.
    
    Parameters:
    matrix (np.ndarray): The input square matrix.
    
    Returns:
    float: The determinant of the matrix.
    
    Raises:
    ValueError: If the matrix is not square.
    """
    if not is_square(matrix):
        raise ValueError("Input matrix must be square.")

    # Base case for 1x1 matrix
    if matrix.shape == (1, 1):
        return matrix[0, 0]

    # Base case for 2x2 matrix
    if matrix.shape == (2, 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    # Recursive case for larger matrices
    det = 0
    for col in range(matrix.shape[1]):
        sub_matrix = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        cofactor = ((-1) ** col) * matrix[0, col] * determinant(sub_matrix)
        det += cofactor

    return det


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


def elementary_matrix_for_row_swap(dim: int, row_a: int, row_b: int) -> np.ndarray:
    """
    Create an elementary matrix that swaps two rows.

    Parameters:
    dim (int): Dimension of the square matrix.
    row_a (int): Index of the row to be swapped.
    row_b (int): Index of the row to be swapped.

    Returns:
    np.ndarray: The elementary matrix for the row swapping operation.

    Raises:
    ValueError: If row indices are invalid or if row_a and row_b are the same.
    """
    if row_a < 0 or row_b < 0 or row_a >= dim or row_b >= dim:
        raise ValueError("Invalid row indices.")

    if row_a == row_b:
        raise ValueError("Swapped rows cannot be the same.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[[row_a, row_b]] = elementary_matrix[[row_b, row_a]]

    return np.array(elementary_matrix)


def elementary_matrix_for_scalar_multiplication(dim: int, row: int, scalar: float) -> np.ndarray:
    """
    Create an elementary matrix that multiplies a row by a scalar.

    Parameters:
    dim (int): Dimension of the square matrix.
    row (int): Index of the row to be scaled.
    scalar (float): Scalar to multiply the row by.

    Returns:
    np.ndarray: The elementary matrix for the scalar multiplication operation.

    Raises:
    ValueError: If row index is invalid or if scalar is zero.
    """
    if row < 0 or row >= dim:
        raise ValueError("Invalid row index.")

    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[row, row] = scalar

    return np.array(elementary_matrix)


def get_inverse_elementary_matrices(matrix: np.ndarray) -> list[np.ndarray]:
    """
    Compute the sequence of elementary matrices used to invert a given matrix.

    Parameters:
    matrix (np.ndarray): The square matrix to be inverted.

    Returns:
    list[np.ndarray]: A list of elementary matrices used to invert the input matrix.

    Raises:
    ValueError: If the matrix is singular or not square.
    """
    dim = get_dim_of_square_matrix(matrix)
    elementary_matrices = []

    mat = matrix.copy()

    # Iterate over each row
    for i in range(dim):
        if mat[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        # Scale the current row to normalize the diagonal
        if mat[i, i] != 1:
            scalar = 1.0 / mat[i, i]
            elementary_matrix = elementary_matrix_for_scalar_multiplication(dim, i, scalar)
            mat = np.dot(elementary_matrix, mat)
            elementary_matrices.append(elementary_matrix)

        # Zero out the elements above and below the diagonal
        rows_above_below = list(range(dim))
        rows_above_below.remove(i)
        for j in rows_above_below:
            scalar = -mat[j, i]
            elementary_matrix = elementary_matrix_for_row_addition(dim, j, i, scalar)
            mat = np.dot(elementary_matrix, mat)
            elementary_matrices.append(elementary_matrix)

    return elementary_matrices


def inverse(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square matrix using elementary row operations.

    Parameters:
    matrix (np.ndarray): The square matrix to be inverted.

    Returns:
    np.ndarray: The inverse of the input matrix.

    Raises:
    ValueError: If the matrix is not square.
    """
    if not is_square(matrix):
        raise ValueError("Input matrix must be square.")

    inv = np.identity(get_dim_of_square_matrix(matrix))

    # TODO: use pivoting?
    elementary_matrices = get_inverse_elementary_matrices(matrix)
    for elementary in elementary_matrices:
        inv = np.dot(elementary, inv)

    return inv


def max_norm(matrix: np.ndarray) -> float:
    """
    Calculates the max norm (infinity norm) of a given matrix.
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    float: The max norm of the matrix.
    """
    # Calculate the sum of absolute values of each row
    row_sums = np.sum(np.abs(matrix), axis=1)
    
    # Find and return the maximum row sum
    return np.max(row_sums)


def condition(matrix: np.ndarray) -> float:
    """
    Calculates the condition value of a given matrix using the max norm (infinity norm).
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    float: The condition value of the matrix.
    """
    return max_norm(matrix) * max_norm(inverse(matrix))


def is_diagonally_dominant(matrix: np.ndarray) -> bool:
    """
    Checks if the given matrix is diagonally dominant.
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    bool: True if the matrix is diagonally dominant, False otherwise.
    """
    if matrix is None:
        return False

    # Calculate the absolute values of the diagonal elements
    diagonal_elements = np.diag(np.abs(matrix))
    
    # Calculate the row sums without the diagonal elements
    row_sums = np.sum(np.abs(matrix), axis=1) - diagonal_elements
    
    # Check if each diagonal element is greater than the corresponding row sum
    return np.all(diagonal_elements > row_sums)


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


def fix_dominant_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Rearranges the matrix to ensure a dominant diagonal, where each diagonal element is greater than or 
    equal to the sum of the absolute values of the other elements in the row.

    Parameters:
    matrix (np.ndarray): The matrix to rearrange.

    Returns:
    np.ndarray: The reordered matrix with a dominant diagonal.
    """
    n = matrix.shape[0]
    reordered_matrix = matrix.copy()
    
    for i in range(n):
        # find the row with the largest absolute value in the current column
        pivot_row = np.argmax(np.abs(reordered_matrix[i:, i])) + i
        
        if pivot_row != i:
            # create the row swap matrix
            elementary_matrix = elementary_matrix_for_row_swap(n, i, pivot_row)
            # apply the row swap to the matrix
            reordered_matrix = np.dot(elementary_matrix, reordered_matrix)
    
    return reordered_matrix


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


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
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




if __name__ == '__main__':
    A = np.array([
        [1,  4, -3],
        [-2, 1,  5],
        [3,  2,  1],
    ])

    print('\nA:')
    print_matrix(A, 5)

    l, u = lu_decomposion(A)
    
    print('\nlower:')
    print_matrix(l, 5)

    print('\nupper:')
    print_matrix(u, 5)


    b = np.array([ 5, 5, 5 ])
    print(f'\n\nb = {b.tolist()}')
    print(f'solution = {gaussian_elimination(A, b).tolist()}')

