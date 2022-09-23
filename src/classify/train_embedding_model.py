"""
This module provides training of LaBSE transformer for given embedding dimensionality.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
"""
import os
import csv
import gzip
import torch
import random
import logging
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    util,
    models,
    evaluation,
    InputExample,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains LaBSE for given embedding dimensionality."
    )
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="../../models/")

    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout

    model = SentenceTransformer("sentence-transformers/LaBSE")

    # We use AllNLI as a source of sentences to compute PCA
    nli_dataset_path = "../../datasets/AllNLI.tsv.gz"

    # We use the STS benchmark dataset to see how much performance we loose by using the dimensionality reduction
    sts_dataset_path = "../../datasets/stsbenchmark.tsv.gz"

    if not os.path.exists(nli_dataset_path):
        print("Downloading NLI dataset...")
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        print("Downloading STS benchmark dataset...")
        util.http_get(
            "https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path
        )

    # We measure the performance of the original model
    # and later we will measure the performance with the reduces dimension size
    logger.info("Read STSbenchmark test dataset")
    eval_examples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["split"] == "test":
                score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
                eval_examples.append(
                    InputExample(
                        texts=[row["sentence1"], row["sentence2"]], label=score
                    )
                )

    # Evaluate the original model on the STS benchmark dataset
    stsb_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        eval_examples, name="sts-benchmark-test"
    )

    logger.info("Original model performance:")
    stsb_evaluator(model)

    ######## Reduce the embedding dimensions ########

    # Read sentences from NLI dataset
    nli_sentences = set()
    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            nli_sentences.add(row["sentence1"])
            nli_sentences.add(row["sentence2"])

    nli_sentences = list(nli_sentences)
    random.shuffle(nli_sentences)

    # To determine the PCA matrix, we need some example sentence embeddings.
    # Here, we compute the embeddings for 20k random sentences from the AllNLI dataset
    pca_train_sentences = nli_sentences[0:20000]
    train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

    # Compute PCA on the train embeddings matrix
    pca = PCA(n_components=args.embedding_dim)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=args.embedding_dim,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module("dense", dense)

    # Evaluate the model with the reduce embedding size
    logger.info("Model with {} dimensions:".format(args.embedding_dim))
    stsb_evaluator(model)

    model.save(os.path.join(args.out_dir, f"labse_{args.embedding_dim}_model"))
