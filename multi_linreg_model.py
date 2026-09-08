import csv
import random


filepath = 'multi_linreg_students.csv'
def load_data(filepath):
    X = []
    y = []
    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            X.append([float(row[0]), float(row[1]), float(row[2])]) # first 3 columns as features
            y.append(float(row[3]))  # exam score as target 
    return X, y


def split(X, y):
    ran_generator = random.Random(42)
    indices = list(range(len(X)))
    ran_generator.shuffle(indices)

    # create new lists for shuffled data
    X_shuffled = [] 
    y_shuffled = []
    for i in indices:
        X_shuffled.append(X[i])
        y_shuffled.append(y[i])

    split_index = int(0.8 * len(X))
    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]
    return X_train, y_train, X_test, y_test


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

    # Create [matrix | identity matrix]
    augmented = []

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

        # Move the best pivot row into position
        augmented[pivot_index], augmented[largest_row] = (
            augmented[largest_row],
            augmented[pivot_index],
        )

        # Divide the pivot row so the pivot becomes 1
        pivot_value = augmented[pivot_index][pivot_index]

        for col_index in range(2 * size):
            augmented[pivot_index][col_index] /= pivot_value

        # Make every other value in the pivot column equal to 0
        for row_index in range(size):
            if row_index != pivot_index:
                factor = augmented[row_index][pivot_index]

                for col_index in range(2 * size):
                    augmented[row_index][col_index] -= (factor * augmented[pivot_index][col_index])

    # Extract the inverse from the right side
    inverse = []

    for row_index in range(size):
        inverse.append(augmented[row_index][size:])

    return inverse


def fit_model(X, y):
    X_with_intercept = []  # add a column of ones to X for the intercept term
    for row in X:
        X_with_intercept.append([1.0] + row)

    y_column_vector = []  # convert y to a column vector
    for val in y:
        y_column_vector.append([val])

    X_transposed = transpose(X_with_intercept)
    X_transposed_X = matrix_multiply(X_transposed, X_with_intercept)
    X_transposed_X_inv = matrix_inverse(X_transposed_X)
    X_transposed_y = matrix_multiply(X_transposed, y_column_vector)
    coefficient_matrix = matrix_multiply(X_transposed_X_inv, X_transposed_y)

    coefficients = [] # convert cofficient matrix to a list
    for i in coefficient_matrix:
        coefficients.append(i[0])

    return coefficients


def predict(X, coefficients):
    X_with_intercept = []
    for row in X:
        X_with_intercept.append([1.0] + row)

    coefficient_column = []
    for i in coefficients:
        coefficient_column.append([i])

    prediction_matrix = matrix_multiply(X_with_intercept, coefficient_column)

    predictions = []  # convert prediction matrix to a list
    for i in prediction_matrix:
        predictions.append(i[0])

    return predictions


def evaluate(y, predictions):
    total_y = 0
    for value in y:
        total_y += value
    y_mean = total_y/len(y)

    sse_fit = 0
    for i in range(len(y)):
        sq_errors = (y[i] - predictions[i]) ** 2
        sse_fit += sq_errors
    mse = sse_fit / len(y)

    sse_mean = 0
    for i in range(len(y)):
        sq_errors_mean = (y[i] - y_mean) ** 2
        sse_mean += sq_errors_mean

    r_squared = 1 - (sse_fit / sse_mean) 

    return mse, r_squared


def main():
    X, y = load_data(filepath)
    X_train, y_train, X_test, y_test = split(X, y)

    coefficients = fit_model(X_train, y_train)

    b0, b1, b2, b3 = coefficients

    print("\n========== Multiple Linear Regression Model Summary ==========\n")
    print("Regression equation:")
    print(f"y_predicted = {b0:.4f} + {b1:.4f}(hours studied) + {b2:.4f}(classes attended) + {b3:.4f}(past score)\n")

    y_predicted_train = predict(X_train, coefficients)
    mse_train, r_squared_train = evaluate(y_train, y_predicted_train)

    y_predicted_test = predict(X_test, coefficients)
    mse_test, r_squared_test = evaluate(y_test, y_predicted_test)


    print("Training Data Evaluation:")
    print("-------------------------")
    print(f"Mean Squared Error (MSE): {mse_train}")
    print(f"R-squared: {r_squared_train}\n")

    print("Testing Data Evaluation:")
    print("-------------------------")
    print(f"Mean Squared Error (MSE): {mse_test}")
    print(f"R-squared: {r_squared_test}\n")


    print("Enter the prediction values...")
    hours_studied = float(input("Number of hours studied: "))
    classes_attended = float(input("Number of classes attended: "))
    past_score = float(input("Past exam score: "))

    new_student = [
        [hours_studied, classes_attended, past_score]
    ]

    predicted_score = predict(new_student, coefficients)[0]

    print(f"Predicted exam score: {predicted_score}")

if __name__ == "__main__":
    main()
