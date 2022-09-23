"""
This module provides datamodule for tweets dataset.
"""
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class TweetsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 0,
        random_state: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.sample_map = None
        self.le = LabelEncoder()

    def process_data(self, data):
        data.loc[data["target"] == "NA", "target"] = "0"
        data["id_str"] = data["id_str"].astype("int").astype("str")

        self.le.fit(data["target"])
        y = data["target"].apply(lambda x: self.le.transform([x])[0])

        X = pd.DataFrame(data["embedding"].tolist())
        X = pd.concat([pd.DataFrame(X), data["id_str"]], axis=1)

        return X, y

    def calc_sample_map(self, data):
        self.sample_map = {
            idx: sample_id
            for idx, sample_id in enumerate(data["id_str"].values.tolist())
        }
        return data

    def format_data(self, data):
        return torch.from_numpy(np.vstack(data.to_numpy())).float()

    def setup(self, stage: Optional[str] = None):
        data = pd.read_json(self.data_path, lines=True)

        X, y = self.process_data(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        X_train = self.calc_sample_map(X_train)
        X_train = X_train.drop("id_str", axis=1)
        X_test = X_test.drop("id_str", axis=1)

        self.X_train = self.format_data(X_train)
        self.X_test = self.format_data(X_test)
        self.y_train = self.format_data(y_train)
        self.y_test = self.format_data(y_test)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def get_test_sets(self):
        return self.X_test, self.y_test

    def get_batch_size(self):
        return self.batch_size

    def get_label_encoder(self):
        return self.le

    def get_sample_map(self):
        return self.sample_map
