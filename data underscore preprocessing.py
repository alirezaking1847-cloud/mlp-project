
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    INPUT_SIZE,
    TEST_SIZE,
    VALIDATION_SIZE
)


def load_data():
    """Load dataset from CSV file."""

    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' not found in dataset."
        )

    return data


def preprocess_data(data):
    """Clean and scale the dataset."""

    values = data[TARGET_COLUMN].dropna().values
    values = values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    return scaled_data, scaler


def create_sequences(data, input_size):
    """Create input sequences for MLP model."""

    X = []
    y = []

    for i in range(len(data) - input_size):
        X.append(data[i:i + input_size].flatten())
        y.append(data[i + input_size, 0])

    return np.array(X), np.array(y)


def split_data(X, y):
    """Split data into training, validation and testing sets."""

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False
    )

    validation_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        shuffle=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():

    data = load_data()

    scaled_data, scaler = preprocess_data(data)

    X, y = create_sequences(
        scaled_data,
        INPUT_SIZE
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    print("Data preprocessing completed successfully!")

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Testing data:", X_test.shape)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )


if __name__ == "__main__":
    main()

