from data_utils import load_data, split
from regression import fit_model, predict, evaluate

def main():
    filepath = 'multi_linreg_students.csv'
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
    print("------------------------------")
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
