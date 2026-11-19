"""Build deterministic Aristocrat puzzles from local text and public-domain books."""

import hashlib
import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from aristocrat_chat import SPACE_ID, extract_features

# Whole books by the same author stay in the same split.
BOOKS = {
    "main": ("austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt",
             "carroll-alice.txt", "melville-moby_dick.txt", "whitman-leaves.txt"),
    "refiner": ("bryant-stories.txt", "milton-paradise.txt", "shakespeare-caesar.txt",
                "shakespeare-hamlet.txt", "shakespeare-macbeth.txt"),
    "validation": ("edgeworth-parents.txt", "burgess-busterbrown.txt"),
    "test": ("chesterton-ball.txt", "chesterton-brown.txt", "chesterton-thursday.txt"),
}
CORPUS_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
# Pin the archive so an upstream change or damaged cache cannot silently change training data.
CORPUS_SHA256 = "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611"


def read_book_archive(cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "gutenberg.zip"
    if path.is_file():
        raw = path.read_bytes()
    else:
        print(f"Downloading the NLTK Gutenberg corpus from {CORPUS_URL}", flush=True)
        try:
            with urlopen(CORPUS_URL, timeout=60) as response:
                raw = response.read()
        except OSError as error:
            raise RuntimeError(
                f"Could not download the Gutenberg corpus. Save {CORPUS_URL} to {path} "
                "or use --bundled-only with a new run directory."
            ) from error

    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != CORPUS_SHA256:
        raise ValueError(
            f"Gutenberg archive at {path} failed SHA-256 verification "
            f"(expected {CORPUS_SHA256}, got {actual_hash}). "
            "Replace it with the verified archive or use --bundled-only with a new run directory."
        )

    with ZipFile(BytesIO(raw)) as archive:
        documents = {
            name: [words(archive.read(f"gutenberg/{book}").decode("latin-1")) for book in names]
            for name, names in BOOKS.items()
        }
    if not path.exists():
        # A unique temporary file prevents concurrent preparations from sharing a partial cache.
        with NamedTemporaryFile(dir=cache, prefix="gutenberg-", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                temp.write(raw)
            except BaseException:
                temp.close()
                temp_path.unlink(missing_ok=True)
                raise
        try:
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return documents


def words(text):
    text = text.lower().replace("’", "").replace("'", "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[a-z]+", text)


def load_documents(base, cache, bundled_only, extra_corpus):
    bundled = words(json.loads((base / "data.json").read_text(encoding="utf-8"))["text"])
    if bundled_only:
        cuts = [0, int(len(bundled) * 0.68), int(len(bundled) * 0.82),
                int(len(bundled) * 0.91), len(bundled)]
        splits = {
            name: [bundled[start:end]]
            for name, start, end in zip(BOOKS, cuts, cuts[1:])
        }
    else:
        splits = read_book_archive(cache)
        splits["main"].append(bundled)
    for path in extra_corpus:
        raw = path.read_text(encoding="utf-8-sig")
        splits["main"].append(words(json.loads(raw)["text"] if path.suffix.lower() == ".json" else raw))
    return splits


def data_signature(documents):
    digest = hashlib.sha256()
    for name, docs in documents.items():
        digest.update(name.encode())
        for document in docs:
            digest.update(" ".join(document).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def make_puzzles(documents, count, seed):
    """Return (cipher tokens, plaintext tokens) using unique word windows."""
    rng = np.random.default_rng(seed)
    documents = [document for document in documents if len(document) >= 32]
    if not documents:
        raise ValueError("Each data split needs a text document with at least 32 words.")
    starts = np.cumsum([len(document) for document in documents])
    alphabet = np.arange(1, SPACE_ID)
    seen = set()
    puzzles = []
    attempts = 0
    while len(puzzles) < count:
        attempts += 1
        if attempts > max(count * 20, 1_000):
            raise ValueError(f"Only generated {len(puzzles):,} unique puzzles. Add text or reduce the count.")
        draw = int(rng.integers(starts[-1]))
        doc_index = int(np.searchsorted(starts, draw, side="right"))
        start = draw - (starts[doc_index - 1] if doc_index else 0)
        target = int(rng.integers(65, 201))
        chosen = []
        letters = 0
        for index in range(start, min(start + 100, len(documents[doc_index]))):
            word = documents[doc_index][index]
            chosen.append(word)
            letters += len(word)
            if letters >= target:
                break
        if letters < target:
            continue
        plain = " ".join(chosen)
        if plain in seen:
            continue
        seen.add(plain)
        tokens = np.frombuffer(plain.encode("ascii"), dtype=np.uint8)
        tokens = np.where(tokens == 32, SPACE_ID, tokens - 96).astype(np.uint8)
        while True:
            key = rng.permutation(alphabet)
            if np.all(key != alphabet):
                break
        cipher = np.append(key, SPACE_ID)[tokens - 1].astype(np.uint8)
        puzzles.append((cipher, tokens))
    return puzzles


def supervised_symbols(puzzles):
    """Generate one feature row per observed cipher symbol and its label."""
    capacity = len(puzzles) * 26
    features = np.empty((capacity, 197), dtype=np.float16)
    labels = np.empty(capacity, dtype=np.uint8)
    offset = 0
    for cipher, plain in puzzles:
        rows, counts = extract_features(cipher - 1)
        observed = counts > 0
        cipher_to_plain = np.zeros(26, dtype=np.uint8)
        letters = cipher != SPACE_ID
        cipher_to_plain[cipher[letters] - 1] = plain[letters] - 1
        end = offset + int(observed.sum())
        features[offset:end] = rows[observed]
        labels[offset:end] = cipher_to_plain[observed]
        offset = end
    return features[:offset], labels[:offset]
