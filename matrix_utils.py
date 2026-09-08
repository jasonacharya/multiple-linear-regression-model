def transpose(matrix):
    num_of_rows = len(matrix)
    num_of_columns = len(matrix[0])
    transposed = []
    for col_index in range(num_of_columns):
        new_row = []
        for row_index in range(num_of_rows):
            new_row.append(matrix[row_index][col_index])
        transposed.append(new_row)
    return transposed


def matrix_multiply(A, B):
    num_of_rows_A = len(A)
    num_of_cols_A = len(A[0])
    num_of_rows_B = len(B)
    num_of_cols_B = len(B[0])

    if num_of_cols_A != num_of_rows_B:
        raise ValueError("Invalid matrix dimensions")

    result = []
    for row_index in range(num_of_rows_A):
        new_row = []
        for col_index in range(num_of_cols_B):
            sum_product = 0
            for k in range(num_of_cols_A):
                sum_product += A[row_index][k] * B[k][col_index]
            new_row.append(sum_product)
        result.append(new_row)
    return result


def matrix_inverse(matrix):
    size = len(matrix)

    for row in matrix:  # check if matrix is square
        if len(row) != size:
            raise ValueError("Only square matrices can be inverted")

    augmented = []   # create [matrix | identity matrix]

    for row_index in range(size):
        new_row = []
                           
        for col_index in range(size):  # copy the original matrix values
            new_row.append(float(matrix[row_index][col_index]))

        for col_index in range(size):  # add identity matrix values
            if row_index == col_index:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        augmented.append(new_row)

    # Apply Gauss-Jordan elimination
    for pivot_index in range(size):
        largest_row = pivot_index

        for row_index in range(pivot_index + 1, size):
            if abs(augmented[row_index][pivot_index]) > abs(augmented[largest_row][pivot_index]):
                largest_row = row_index

        if abs(augmented[largest_row][pivot_index]) < 1e-12:  # zero pivot = matrix has no inverse
            raise ValueError("Matrix cannot be inverted")
        
        augmented[pivot_index], augmented[largest_row] = (   # move the best pivot row into position
            augmented[largest_row],
            augmented[pivot_index],
        )

        pivot_value = augmented[pivot_index][pivot_index]   # divide the pivot row so the pivot becomes 1

        for col_index in range(2 * size):
            augmented[pivot_index][col_index] /= pivot_value

        for row_index in range(size):   # make every other value in the pivot column equal to 0
            if row_index != pivot_index:
                factor = augmented[row_index][pivot_index]

                for col_index in range(2 * size):
                    augmented[row_index][col_index] -= (factor * augmented[pivot_index][col_index])

    inverse = []   # extract inverse from right

    for row_index in range(size):
        inverse.append(augmented[row_index][size:])

    return inverse