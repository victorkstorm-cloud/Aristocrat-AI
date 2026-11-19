"""Build a reproducible inference lexicon from the Gutenberg training books.

Only the books named in the main and refiner splits are read. The bundled text,
extra corpora, validation books and test books never contribute to the counts.
"""

import argparse
from collections import Counter
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = BASE / "corpus_cache" / "gutenberg.zip"
DEFAULT_LEXICON = BASE / "models" / "english_words.tsv.gz"


def training_books(archive_path):
    """Yield normalized documents from the eleven language-model training books."""
    from training_data import BOOKS, CORPUS_SHA256, words

    raw = Path(archive_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != CORPUS_SHA256:
        raise ValueError(f"Gutenberg archive SHA-256 mismatch: expected {CORPUS_SHA256}, got {actual}")
    with ZipFile(BytesIO(raw)) as archive:
        for split in ("main", "refiner"):
            for book in BOOKS[split]:
                yield words(archive.read(f"gutenberg/{book}").decode("latin-1"))


def build_word_counts(archive_path=DEFAULT_ARCHIVE):
    """Count normalized words in the verified main and refiner books only."""
    counts = Counter()
    for document in training_books(archive_path):
        counts.update(document)
    return counts


def write_word_counts(counts, path=DEFAULT_LEXICON):
    """Write sorted UTF-8 TSV with a stable gzip header and no timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(f"{word}\t{counts[word]}\n" for word in sorted(counts)).encode("utf-8")
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            compressed.write(data)


def load_word_counts(path=DEFAULT_LEXICON):
    """Load the packaged lexicon as a mapping from normalized word to count."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return {word: int(count) for word, count in (line.rstrip("\n").split("\t") for line in source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_LEXICON)
    args = parser.parse_args()
    counts = build_word_counts(args.archive)
    write_word_counts(counts, args.output)
    print(f"Wrote {len(counts):,} words ({sum(counts.values()):,} tokens) to {args.output}")


if __name__ == "__main__":
    main()
