"""
This module provides utils raw tweets data preprocessing.
"""
import os
import re
import demoji
import jsonlines
import pandas as pd
from tqdm.auto import tqdm
from pymongo import MongoClient, database


def get_database():
    """Provides the mongodb atlas url to connect python to database using pymongo."""
    username = os.environ["MONGO_INITDB_ROOT_USERNAME"]
    password = os.environ["MONGO_INITDB_ROOT_PASSWORD"]
    ca_path = os.environ["CA_CERTIFICATE_PATH"]
    connString = f"mongodb+srv://{username}:{password}@db-mongodb-pbc-d07d8146.mongo.ondigitalocean.com/admin?authSource=admin&replicaSet=db-mongodb-pbc&tls=true&tlsCAFile={ca_path}"
    client = MongoClient(connString)

    # Create the database
    return client["pbc"]


def get_local_database():
    """Provides the local mongodb to connect python to database using pymongo."""
    connString = os.environ["MONGODB_LOCALHOST_CONNSTRING"]
    client = MongoClient(connString)

    # Create the database
    return client["pbc"]


def download_data(
    db: database.Database, col: str, filepath: str, fields: list[str]
) -> None:
    """Downloads data collection with specified fields from mongo database."""
    projection = {col: 1 for col in fields}
    projection["_id"] = 0
    res = db[col].find({}, projection)
    count = db[col].count_documents({})
    with jsonlines.open(filepath, mode="w") as writer:
        for post in tqdm(res, total=count):
            post["created_at"] = post["created_at"].isoformat()
            post["download_datetime"] = post["download_datetime"].isoformat()
            writer.write(post)


def demojify(df: pd.DataFrame, text_col: str, new_col: str) -> pd.DataFrame:
    """Parse emojis into text and export text to a new column."""
    tqdm.pandas()
    df[new_col] = df.progress_apply(
        lambda row: demoji.replace_with_desc(row[text_col]), axis=1
    )

    return df


def clean_row(text):
    """Removes usernames, links and additional whitespaces from text."""
    wo_usernames = re.sub("@[^\s]+", "", text)
    wo_links = re.sub("http[^\s]+", "", wo_usernames)
    wo_whitespaces = " ".join(wo_links.split())

    return wo_whitespaces


# def clean_tweet(tweet: str) -> str:
def clean_text(df: pd.DataFrame, text_col: str, new_col: str) -> str:
    """Removes usernames, links and additional whitespaces and export text to a new column."""
    tqdm.pandas()
    df[new_col] = df.progress_apply(lambda row: clean_row(row[text_col]), axis=1)

    return df
