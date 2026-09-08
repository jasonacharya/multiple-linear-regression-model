from matrix_utils import transpose, matrix_multiply, matrix_inverse


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