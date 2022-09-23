"""
This module provides Multilayer Perceptron model.
"""
import os
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple
from sklearn.metrics import f1_score, accuracy_score, classification_report


class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, lr=1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr

        assert len(hidden_dims) > 0, "hidden_dims list can not be empty"

        a, b = itertools.tee(hidden_dims)
        next(b, None)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for d1, d2 in zip(a, b):
            layers.append(nn.Linear(d1, d2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], out_features=output_dim))
        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten().type(torch.LongTensor).to(self.device)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)

        return {
            "loss": loss,
            "logits": y_hat.detach(),
            "gold": y,
            "batch_idx": batch_idx,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten().type(torch.LongTensor).to(self.device)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = torch.argmax(self(x), dim=1)
        f1 = f1_score(y.cpu(), y_hat.cpu(), average="micro")
        acc = accuracy_score(y.cpu(), y_hat.cpu())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1_micro", f1.astype(np.float32), prog_bar=True)
        self.log("val_acc", acc.astype(np.float32), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten().type(torch.LongTensor)
        y_hat = self(x)
        y_hat = torch.argmax(self(x), dim=1)

        report = classification_report(y, y_hat, output_dict=True)
        self.log_dict(report)
        return report

    def training_epoch_end(self, outputs):
        batch_size = self.trainer.datamodule.get_batch_size()
        sample_map = self.trainer.datamodule.get_sample_map()

        data = {"guid": [], f"logits_epoch_{self.current_epoch}": [], "gold": []}
        for batch in outputs:
            batch_idx = batch["batch_idx"]
            curr_batch_size = len(batch["logits"])
            data["guid"] += [
                sample_map[batch_size * batch_idx + idx]
                for idx in range(curr_batch_size)
            ]
            data[f"logits_epoch_{self.current_epoch}"] += batch["logits"].tolist()
            data["gold"] += batch["gold"].tolist()

        df = pd.DataFrame(data)
        logs_path = "./logs/mlp/training_dynamics/"
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        df.to_json(
            os.path.join(logs_path, f"dynamics_epoch_{self.current_epoch}.jsonl"),
            lines=True,
            orient="records",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        z = self(x)
        return z, y


def train():
    pass
