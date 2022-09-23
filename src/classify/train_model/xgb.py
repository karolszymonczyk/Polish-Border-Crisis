"""
This module provides training and evaluation of xgboost classifier.
"""
import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def prepare_data(data_path):
    annotated_df = pd.read_json(data_path, lines=True)
    annotated_df["id_str"] = annotated_df["id_str"].astype("int").astype("str")
    annotated_df.loc[annotated_df["target"] == "NA", "target"] = "0"

    le = LabelEncoder()
    le.fit(annotated_df["target"])
    annotated_df["label"] = annotated_df["target"].apply(lambda x: le.transform([x])[0])

    train, test = train_test_split(
        annotated_df,
        test_size=args.test_size,
        stratify=annotated_df["target"],
        random_state=42,
    )

    X_train, y_train = train["embedding"], train["label"]
    X_test, y_test = test["embedding"], test["label"]

    return X_train, y_train, X_test, y_test, le


def test_model(model, X, y, le):
    y_hat = model.predict(X.to_list())
    return classification_report(
        y.to_list(), y_hat, zero_division=0, target_names=le.classes_
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates XGBoost classifier."
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--params_path", type=str, default="./xgb_params.json")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--data_path", type=str, default="../../../data/embedded_sample.jl"
    )
    parser.add_argument("--out_dir", type=str, default="../../../models/xgb/")

    args = parser.parse_args()
    params = json.load(open(args.params_path))

    print("Loading data...")
    X_train, y_train, X_test, y_test, le = prepare_data(args.data_path)
    le_path = os.path.join(args.out_dir, f"le_{args.model_name}.npy")
    np.save(le_path, le.classes_)

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model = model.fit(X_train.to_list(), y_train.to_list())

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    model_path = os.path.join(args.out_dir, args.model_name)
    pickle.dump(model, open(f"{model_path}.pkl", "wb"))
    report = test_model(model, X_test, y_test, le)

    print("XGBoost results:")
    print(report)
