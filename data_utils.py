import csv
import random


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