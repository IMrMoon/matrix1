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
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    return identity


if __name__ == '__main__':

    A = np.array([[1, 1, 1],
                  [0, 2, 4],
                  [1, 0, 1]])

    try:
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))


    #the results vector
    B = np.array([7, 2, 5])

    #dot mul the inverse matrix A with the B vector of the results to calculate the X which is the final result vector
    X = np.dot(A_inverse, B)

    print(X)

    import numpy as np

    #the checking if the inverse is the real inverse of the matrix and return true or false
    def checkInverse(inverseMatrix, matrix):
        n = matrix.shape[0]
        id = np.dot(inverseMatrix, matrix)

        #working too, but not a good check
        # for i in range(n):
        #     if id[i, i] == 1:
        #         return False
        # return True

        for i in range(n):
            for j in range(n):
                if i != j and id[i, j] != 0:
                    return False
                elif i == j and id[i, j] != 1:
                    return False

        return True

    print("the result of the check is: ", checkInverse(A_inverse, A))



