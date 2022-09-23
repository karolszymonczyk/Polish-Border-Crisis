"""
This module provides tweets classification using trained model.
"""
import os
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm
from train_model.models.MLP import MLP
from sklearn.preprocessing import LabelEncoder


def mlp_predict(model, embedding, le):
    x = torch.tensor(embedding).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds = model(x)
    preds = torch.argmax(preds, 1)
    return le.inverse_transform(preds)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedd tweets text using LaBSE transformer."
    )
    parser.add_argument("--model_type", type=str, choices=["mlp", "xgb"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--params_path", type=str)
    parser.add_argument(
        "--data_path", type=str, default="../../data/embedded_tweets.jl"
    )
    parser.add_argument("--out_dir", type=str, default="../../data")

    args = parser.parse_args()
    params = json.load(open(args.params_path))

    print("Loading model...")
    if args.model_type == "mlp":
        model = MLP.load_from_checkpoint(checkpoint_path=args.model_path, **params)
    else:
        model = pickle.load(open(args.model_path, "rb"))

    print("Loading data...")
    data = pd.read_json(args.data_path, lines=True)

    le = LabelEncoder()
    model_name = os.path.basename(os.path.normpath(args.model_path))
    pure_model_name = model_name.split(".")[0]
    le_path = args.model_path.replace(model_name, f"le_{pure_model_name}.npy")
    le.classes_ = np.load(le_path, allow_pickle=True)

    print("Classifying tweets...")
    tqdm.pandas()
    if args.model_type == "mlp":
        data["label"] = data["embedding"].progress_apply(
            lambda x: mlp_predict(model, x, le)
        )
    else:
        data["label"] = data["embedding"].progress_apply(
            lambda x: model.predict([x])[0]
        )

    data.drop("embedding", inplace=True, axis=1)
    out_path = os.path.join(args.out_dir, "classified_tweets.jl")
    data.to_json(out_path, lines=True, orient="records")
