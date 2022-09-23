"""
This module provides text from tweets embedding using LaBSE transformer.
"""
import os
import argparse
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedd tweets text using LaBSE transformer."
    )
    parser.add_argument("--labse_model_path", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=150)
    parser.add_argument("--in_dir", type=str, default="../../data/")
    parser.add_argument("--out_dir", type=str, default="../../data/")

    args = parser.parse_args()

    sample_df = pd.read_csv(
        os.path.join(args.in_dir, "sample_with_target.tsv"),
        sep="\t",
        converters={"target": str},
    )
    tweets_df = pd.read_json(os.path.join(args.in_dir, "tweets.jl"), lines=True)

    print("Loading LaBSE model...")
    if len(args.labse_model_path) == 0:
        model = SentenceTransformer("sentence-transformers/LaBSE")
    else:
        model = SentenceTransformer(args.labse_model_path)
    model.max_seq_length = args.max_seq_length

    tqdm.pandas()
    print("Embedding tweets...")
    tweets_df["embedding"] = tweets_df["clean_text"].progress_apply(model.encode)
    tweets_df.to_json(
        os.path.join(args.in_dir, "embedded_tweets.jl"), lines=True, orient="records"
    )

    tweets_df["id_str"] = tweets_df["id_str"].astype(str)
    sample_ids = list(sample_df["id"].astype(str).values)
    sample_embedded_df = pd.DataFrame(tweets_df[tweets_df["id_str"].isin(sample_ids)])
    sample_embedded_df["target"] = sample_embedded_df["id_str"].apply(
        lambda x: sample_df[sample_df["id"] == int(x)]["target"]
    )
    sample_embedded_df.to_json(
        os.path.join(args.in_dir, "embedded_sample.jl"), lines=True, orient="records"
    )
