import matrix_utility
from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def matrix_inverse(matrix):
    print(bcolors.OKBLUE,
          f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n",
          bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix

    if np.linalg.det(matrix) == 0:
        # if matrix[i, i] == 0:
        raise ValueError("Matrix is singular, cannot find its inverse.")
    make_diagonal_nonzero(matrix, identity)

    for i in range(n):

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN,
                  "------------------------------------------------------------------------------------------------------------------",
                  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN,
                      "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    return identity

def make_diagonal_nonzero(matrix, identity):
    n = len(matrix)

    for k in range(n):
        if matrix[k, k] == 0:
            # Find a non-zero element in the same column below the current zero diagonal element
            for b in range(k + 1, n):
                if matrix[b, k] != 0:
                    # Swap rows to make the diagonal element nonzero
                    matrix[[k, b], :] = matrix[[b, k], :]
                    identity[[k, b], :] = identity[[b, k], :]

    return matrix, identity


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)
    A = np.array([[1, 3, 3],
                  [2, 0, 6],
                  [7, 0, 9]])
    A_before = A.copy()

    try:
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(
            "=====================================================================================================================",
            bcolors.ENDC)

    except ValueError as e:
        print(str(e))
    # returnAtoNormal(A)
    # the results vector
    B = np.array([7, 2, 5])

    # dot mul the inverse matrix A with the B vector of the results to calculate the X which is the final result vector
    X = np.dot(A_inverse, B)

    print(X)

    import numpy as np


    # the checking if the inverse is the real inverse of the matrix and return if the values
    #between these 2 arrays are equal in some lvl return true, else return false or if we have nans return false
    def checkInverse(inverseMatrix, matrix):
        #check the size of the matrix
        n = matrix.shape[0]
        #creating a dot multipication between the original matrix to the inverse one
        product = np.dot(matrix, inverseMatrix)
        #create a id matrix size n like the matrix
        identity = np.identity(n)
        return np.allclose(product, identity)


    invA = A_inverse
    print("the result of the check is: ", checkInverse(invA, A_before))
