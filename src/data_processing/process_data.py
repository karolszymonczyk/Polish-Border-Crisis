"""
This module process raw tweets data and exports it to a new data file.
"""
import os
import argparse
import pandas as pd

from utils import demojify, clean_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse emojis into text, save it in a new column and export."
    )
    parser.add_argument("--in_file", type=str, default="../../data/raw_tweets.jl")
    parser.add_argument("--out_dir", type=str, default="../../data/")
    parser.add_argument("--text_col", type=str, default="full_text")
    parser.add_argument("--new_col", type=str, default="clean_text")
    parser.add_argument("--filetype", type=str, choices=["jl", "tsv"], default="jl")

    args = parser.parse_args()

    df = pd.read_json(args.in_file, lines=True, dtype=False)

    print("Demojifying tweets...")
    demoji_df = demojify(df, args.text_col, args.new_col)
    print("Cleaning tweets...")
    clean_df = clean_text(df, args.text_col, args.new_col)

    out_file = os.path.join(args.out_dir, f"tweets.{args.filetype}")
    clean_df.to_csv(out_file, sep="\t", index=False)
    if args.filetype == "tsv":
        clean_df.to_csv(out_file, sep="\t", index=False, encoding="utf-8")
    else:
        clean_df.to_json(out_file, lines=True, orient="records")
