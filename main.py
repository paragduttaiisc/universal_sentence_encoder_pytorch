"""Quick-start demo: embed a sentence with the PyTorch port of USE-large (v5)."""

import argparse
from pathlib import Path

from use_embed import MODEL_PATH, VOCAB_PATH, USEEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed a sentence with the PyTorch port of "
        "Universal Sentence Encoder (large, v5)."
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH,
        help=f"path to the PyTorch embedding model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--vocab", type=Path, default=VOCAB_PATH,
        help=f"path to the vocabulary .npz (default: {VOCAB_PATH})",
    )
    args = parser.parse_args()

    embedder = USEEmbedder(vocab_path=args.vocab, model_path=args.model)
    print("USE v5 loaded")

    sentence = "The quick brown fox jumped over the lazy dog"
    embedding = embedder.embed(sentence)
    print(len(embedding), "dimensional embedding vector:")
    print("[", embedding[0], embedding[1], embedding[2], " ... ",
          embedding[-3], embedding[-2], embedding[-1], "]")


if __name__ == "__main__":
    main()
