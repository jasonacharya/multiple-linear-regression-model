# Multiple Linear Regression Model

A Multiple Linear Regression model written from scratch using standard Python. It predicts a student’s exam score using:

* Hours studied
* Classes attended
* Past exam score

The model uses the normal equation, manually implemented matrix operations, and an 80/20 training and testing split. It does not use NumPy, pandas, Scikit-learn, or another machine-learning library.

## Project Structure

```text
multiple-linear-regression-model/
├── data_utils.py
├── main.py
├── matrix_utils.py
├── regression.py
├── multi_linreg_students.csv
└── README.md
```

* `main.py` — Runs the program and handles user input and output.
* `data_utils.py` — Loads, shuffles, and splits the dataset.
* `matrix_utils.py` — Contains manually implemented matrix operations.
* `regression.py` — Fits the model, makes predictions, and calculates evaluation metrics.
* `multi_linreg_students.csv` — Contains the student data used by the model.

## Model

The prediction equation is:

```text
predicted_score = b0
                + b1(hours_studied)
                + b2(classes_attended)
                + b3(past_score)
```

The coefficients are calculated using the normal equation:

```text
β = (XᵀX)⁻¹Xᵀy
```

## Running the Program

Python 3 is required. No third-party packages need to be installed.

Run the program from the project directory:

```bash
python main.py
```

The program displays:

* The learned regression coefficients
* Training and testing MSE
* Training and testing R²
* A predicted exam score based on user input
