"""
This module provides Multilayer Perceptron classifier trainig.
"""
import os
import json
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from tensorboard import program
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

from models.MLP import MLP
from data_modules.tweets_data_module import TweetsDataModule


def train_model(
    dataloader,
    model,
    model_name,
    accelerator,
    devices,
    epochs,
    model_path,
):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_path,
        filename=model_name,
        save_top_k=1,
        mode="min",
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        "./logs/mlp/lightning_logs/", name=model_name
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    trainer.fit(model, dataloader)

    return trainer


def test_model(model, X, y, le):
    model.eval()
    with torch.no_grad():
        preds = model(X)
    preds = torch.argmax(preds, 1)

    return classification_report(y, preds, zero_division=0, target_names=le.classes_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates Multilayer Perceptron classifier."
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--params_path", type=str, default="./mlp_params.json")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accelerator", type=str, default="mps")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--data_path", type=str, default="../../../data/embedded_sample.jl"
    )
    parser.add_argument("--out_dir", type=str, default="../../../models/mlp/")

    args = parser.parse_args()
    params = json.load(open(args.params_path))

    data = TweetsDataModule(
        data_path=args.data_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = MLP(**params)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "./logs"])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    print("Training MLP model...")
    trainer = train_model(
        data,
        model,
        args.model_name,
        epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        model_path=args.out_dir,
    )

    X_test, y_test = data.get_test_sets()
    le = data.get_label_encoder()
    le_path = os.path.join(args.out_dir, f"le_{args.model_name}.npy")
    np.save(le_path, le.classes_)

    model_path = os.path.join(args.out_dir, args.model_name)
    mlp = MLP.load_from_checkpoint(checkpoint_path=f"{model_path}.ckpt", **params)
    report = test_model(mlp, X_test, y_test, le)

    print("MLP results:")
    print(report)
