# Universal Sentence Encoder (large, v5) — PyTorch

A TensorFlow-free PyTorch port of Google's
[Universal Sentence Encoder (large)](https://tfhub.dev/google/universal-sentence-encoder-large/5)
([`tfhub.dev/google/universal-sentence-encoder-large/5`](https://tfhub.dev/google/universal-sentence-encoder-large/5),
Hugging Face mirror:
[vprelovac/universal-sentence-encoder-large-5](https://huggingface.co/vprelovac/universal-sentence-encoder-large-5)).

The original TF Hub module is an inference-only model. This repository contains the
full inference stack converted to PyTorch through an ONNX intermediary, so it can be
dropped into any PyTorch pipeline (e.g. RT-1-style task-description embedding) with
no TensorFlow dependency at run time.

```
text ──► tokenizer (TF-exact) ──► token ids ──► use_v5_embed.pt ──► 512-d L2-normalized embedding
        (use_embed.py,           (fingerprint64.py                 (onnx2pytorch
         fingerprint64.py)         for OOV hashing)                 conversion of the
                                                                          module's ONNX graph)
```

## Requirements

- Python 3.10
- CPU or CUDA-capable machine; `use_v5_embed.pt` is ~1.4 GB

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the bare minimum that is actually imported (or unpickled) at
run time: `numpy`, `torch`, and `onnx2pytorch` (the model file is a pickled
`onnx2pytorch` graph, so the library is required to load it). `onnx2pytorch` pulls in
its own transitive dependencies (`onnx`, `torchvision`) via pip.

## Quick start

```bash
python main.py
```

Expected output:

```
USE v5 loaded
512 dimensional embedding vector:
[ 0.0022906582 -0.00490418 -0.053970866  ...  -0.015888546 -0.061159834 -0.1130215 ]
```

## Python API

```python
from use_embed import USEEmbedder

embedder = USEEmbedder()  # defaults: use_v5_vocab.npz / use_v5_embed.pt in this repo

ids = embedder.token_ids("pick up the red mug")   # int64 [L] TF-exact token ids
vec = embedder.embed("pick up the red mug")       # (512,) float32, L2-normalized
```

Typical use — similarity between task descriptions (embeddings are L2-normalized, so
cosine similarity is a dot product):

```python
import numpy as np

a = embedder.embed("pick up the red mug")
b = embedder.embed("take the red cup off the table")
print(float(np.dot(a, b)))  # cosine similarity
```

`USEEmbedder` also accepts explicit paths:
`USEEmbedder(vocab_path="...", model_path="...")`.
## How it works

The port consists of two pieces that together reproduce the TF Hub module exactly:

1. **Tokenizer** ([`use_embed.py`](use_embed.py)) — a pure-Python replication of the
   module's text preprocessor, verified bit-exact against the TF module's token ids:

   - case-fold ASCII `A-Z` → `a-z` (the TF preprocessor applies 26 individual regex
     replaces, so only ASCII is affected)
   - strip every character in Unicode category `P*` (the TF regex `\pP`; symbols,
     digits, letters and spaces survive)
   - if the result is empty or all spaces, use the single token `<EMPTY>`
   - otherwise wrap as `<S> <text> </S>` and split on single spaces, dropping empty
     fields
   - token ids come from the vocabulary table in `use_v5_vocab.npz`; out-of-vocabulary
     tokens follow TF's `StringToHashBucketFast` rule:
     `fingerprint64(token) % 200000 + 200004`

2. **Model** ([`use_v5_embed.pt`](use_v5_embed.pt)) — the module's numeric graph
   (token ids → 512-d embedding), exported from the TF module to ONNX and converted to
   PyTorch with `onnx2pytorch`. It takes the token ids, a dense shape, and segment/
   position tensors and returns the L2-normalized 512-d embedding.

`fingerprint64.py` is a pure-Python port of FarmHash's Fingerprint64 — the exact hash
TF uses for `StringToHashBucketFast` — which is what keeps OOV token ids bit-exact
without any external hash library.

## Files

| File | Description |
| --- | --- |
| `use_embed.py` | Tokenizer + `USEEmbedder` (text → 512-d embedding) |
| `fingerprint64.py` | Pure-Python FarmHash Fingerprint64 (TF OOV hashing) |
| `main.py` | Quick-start demo |
| `use_v5_vocab.npz` | Vocabulary table (`words`, `ids`, `table_size`), ~1.8 MB |
| `use_v5_embed.pt` | PyTorch embedding model (ONNX-converted), ~1.4 GB |
| `requirements.txt` | Minimal pinned dependencies |

## Verification against the TF module

- Tokenizer: **bit-exact** token ids against the TF Hub module's
  preprocessor (probe battery covering punctuation, mixed case,
  empty/whitespace-only input, and OOV words via FarmHash fingerprinting).
- Model: max absolute embedding difference **< ~1.3e-7** against the TF
  module on the 14-text battery.

## Source

- TF Hub: <https://tfhub.dev/google/universal-sentence-encoder-large/5>
- Hugging Face mirror of the inference model: <https://huggingface.co/vprelovac/universal-sentence-encoder-large-5>

## License

Code in this repository: CC0-1.0 (public domain), see [LICENSE](LICENSE).
