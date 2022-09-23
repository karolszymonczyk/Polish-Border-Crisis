"""
This module dowloads raw tweets and users data from the mongodb and exports it to JSONLines file.
"""
import os
import dotenv
import argparse

from utils import get_database, get_local_database, download_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download tweets and users from the database and save it to a JSONLines file."
    )
    parser.add_argument("--out_dir", type=str, default="../../data/")
    parser.add_argument("--env", type=str, default="../.env")

    args = parser.parse_args()
    dotenv.load_dotenv(dotenv_path=args.env)

    tweets_fields = [
        "id_str",
        "created_at",
        "download_datetime",
        "full_text",
        "in_reply_to_screen_name",
        "lang",
        "quote_count",
        "reply_count",
        "retweet_count",
        "favorite_count",
        "user_id_str",
    ]
    users_fields = [
        "id_str",
        "created_at",
        "download_datetime",
        "name",
        "screen_name",
        "lang",
        "description",
        "favourites_count",
        "followers_count",
        "friends_count",
        "location",
        "media_count",
    ]

    db = get_local_database()
    out_tweets_file = os.path.join(args.out_dir, "raw_tweets.jl")
    print(f"Downloading tweets from database to {out_tweets_file}...")
    download_data(
        db,
        "tweets",
        out_tweets_file,
        tweets_fields,
    )

    out_users_file = os.path.join(args.out_dir, "users.jl")
    print(f"Downloading users from database to {out_users_file}...")
    download_data(
        db,
        "users",
        out_users_file,
        users_fields,
    )
