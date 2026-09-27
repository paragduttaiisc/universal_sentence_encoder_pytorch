"""TF-free USE-large (v5) language embedding for RT-1.

Replaces the TF Hub module `tfhub.dev/google/universal-sentence-encoder-large/5`
with two pieces:

1. A Python tokenizer replicating the module's text preprocessor, verified
   bit-exact against the TF module's token ids:
     - case-fold ASCII A-Z -> a-z (26 individual replaces)
     - strip every char with Unicode category P* (TF regex \\pP)
     - if result is empty or all spaces -> single token "<EMPTY>"
     - else wrap as "<S> <text> </S>"
     - split on single spaces, dropping empty fields
     - ids: vocab table lookup; OOV -> fingerprint64(tok) % 200000 + 200004
2. ``use_v5_embed.pt``: onnx2pytorch conversion of the module's numeric
   (token-ids -> 512-d L2-normalized embedding) ONNX graph, verified against
   the TF module to <~1.3e-7 max abs diff on the 14-text battery.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from fingerprint64 import fingerprint64

_REPO = Path(__file__).resolve().parent
VOCAB_PATH = _REPO / "use_v5_vocab.npz"
MODEL_PATH = _REPO / "use_v5_embed.pt"

NUM_BUCKETS = 200000
OOV_BASE = 200004  # = table_size (N)
ALL_SPACES_RE = re.compile(r" *")


def casefold_ascii(s: str) -> str:
    """TF preprocessor applies 26 global regex replaces A->a .. Z->z only."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def strip_punctuation(s: str) -> str:
    """TF preprocessor applies regex replace of \\pP -> ''.

    RE2 \\pP == Unicode P* categories (Pc Pd Pe Pf Pi Po Ps); symbols
    (Sm/Sk/So/Sc), letters, digits, spaces and control chars survive.
    """
    return "".join(c for c in s if not unicodedata.category(c).startswith("P"))


def tokenize(text: str) -> list[str]:
    """TF-exact token list for one task description."""
    s = strip_punctuation(casefold_ascii(text))
    s = "" if ALL_SPACES_RE.fullmatch(s) else s
    if s == "":
        wrapped = "<EMPTY>"
    else:
        wrapped = "<S> " + s + " </S>"
        wrapped = "<EMPTY>" if ALL_SPACES_RE.fullmatch(wrapped) else wrapped
    return [t for t in wrapped.split(" ") if t]


class USEEmbedder:
    """text -> (512,) float32 embedding, matching the TF hub module."""

    def __init__(self, vocab_path: Path = VOCAB_PATH,
                 model_path: Path = MODEL_PATH):
        v = np.load(vocab_path, allow_pickle=True)
        self.vocab = {w: int(i) for w, i in
                      zip(v["words"].astype(str), v["ids"].astype(np.int64))}
        self.table_size = int(v["table_size"])
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        self._vocab_arr = None

    def token_ids(self, text: str) -> np.ndarray:
        """int64 [L] token ids (TF StringToHashBucketFast OOV rule)."""
        ids = []
        for tok in tokenize(text):
            i = self.vocab.get(tok)
            if i is None:
                i = fingerprint64(tok) % NUM_BUCKETS + OOV_BASE
            ids.append(i)
        return np.asarray(ids, dtype=np.int64)

    def embed(self, text: str) -> np.ndarray:
        """(512,) float32 L2-normalized embedding of `text`."""
        ids = self.token_ids(text)
        L = ids.shape[0]
        dense_shape = np.array([1, L], dtype=np.int64)
        positions = np.stack([np.zeros(L, dtype=np.int64),
                              np.arange(L, dtype=np.int64)], axis=1)
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(ids),
                torch.from_numpy(dense_shape),
                torch.from_numpy(positions),
            )
        t = out[0] if isinstance(out, (tuple, list)) else out
        t = t[0] if isinstance(t, (tuple, list)) else t
        return t.detach().numpy().reshape(512).astype(np.float32)
