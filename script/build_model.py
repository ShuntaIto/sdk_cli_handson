import argparse
import os
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class DataPrepObject(TypedDict):
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.DataFrame
    original_train: pd.DataFrame
    original_valid: pd.DataFrame


def rmse(validation, target):
    return np.sqrt(mean_squared_error(validation, target))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train_data",
        type=str,
        help="input data",
        default="../data/nyc_taxi_dataset_train.csv",
    )
    parser.add_argument(
        "--input_valid_data",
        type=str,
        help="input data",
        default="../data/nyc_taxi_dataset_valid.csv",
    )

    args = parser.parse_args()

    return args


def load_data(train_data_path: str, valid_data_path: str) -> DataPrepObject:

    df_train = pd.read_csv(train_data_path)
    df_valid = pd.read_csv(valid_data_path)
    df_train.head()
    print("datasets loaded.")

    col_target = "totalAmount"

    X_train = df_train.drop(columns=col_target)
    y_train = df_train[col_target].to_numpy().ravel()

    X_valid = df_valid.drop(columns=col_target)
    y_valid = df_valid[col_target].to_numpy().ravel()

    print("datasets preprocessing complete.")

    data_prep_object: DataPrepObject = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "original_train": df_train,
        "original_valid": df_valid,
    }

    return data_prep_object


def train_model(
    data_prep_object: DataPrepObject
) -> LinearRegression:

    print("training start.")
    model = LinearRegression().fit(
        data_prep_object["X_train"], data_prep_object["y_train"]
    )
    print("training completed.")

    return model


def validate_model(
    model: LinearRegression,
    data_prep_object: DataPrepObject
) -> None:

    col_target = "totalAmount"
    print("model validating start.")
    train_preds = model.predict(data_prep_object["X_train"])
    train_rmse = rmse(
        data_prep_object["original_train"][col_target], train_preds
    )
    print(f"train data RMSE: {train_rmse}")

    valid_preds = model.predict(data_prep_object["X_valid"])
    valid_rmse = rmse(
        data_prep_object["original_valid"][col_target], valid_preds
    )
    print(f"valid data RMSE: {valid_rmse}")

    print("model validating completed.")


if __name__ == "__main__":

    args = parse_args()

    data_prep_object = load_data(
        train_data_path=args.input_train_data,
        valid_data_path=args.input_valid_data,
    )
    model = train_model(data_prep_object)
    validate_model(model, data_prep_object)
