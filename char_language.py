"""Build and load the training-only interpolated five-gram language model.

Source: the verified NLTK Gutenberg archive, main and refiner books only.
Each book is padded with three leading spaces and one trailing space. Symbols
are base-27 digits: a..z are 0..25 and space is 26.
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_lexicon import DEFAULT_ARCHIVE, training_books

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "models" / "english_fivegrams.npz"
VOCABULARY = 27
SIZE = VOCABULARY ** 5


def build_char_language(archive_path=DEFAULT_ARCHIVE):
    """Return 27**5 log probabilities derived only from training books."""
    counts = {n: np.zeros(VOCABULARY ** n, dtype=np.uint32) for n in (3, 4, 5)}
    for document in training_books(archive_path):
        text = '   ' + ' '.join(document) + ' '
        data = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
        digits = np.where(data == 32, 26, data - 97).astype(np.int64)
        for n in counts:
            index = np.zeros(len(digits) - n + 1, dtype=np.int64)
            for pos in range(n):
                index = index * VOCABULARY + digits[pos:pos + len(index)]
            counts[n] += np.bincount(index, minlength=VOCABULARY ** n).astype(np.uint32)
    # P(next | four-character context), interpolated with the suffix
    # three-character context and add-one smoothing over the 27 symbols.
    out = np.empty(SIZE, dtype=np.float32)
    block = VOCABULARY ** 4
    suffix = np.arange(block, dtype=np.int64)
    lower = (counts[4].astype(np.float64) + 1) / (counts[3][suffix // VOCABULARY].astype(np.float64) + VOCABULARY)
    for first in range(VOCABULARY):
        begin = first * block
        indices = begin + suffix
        prob = (counts[5][indices].astype(np.float64) + 10 * lower) / (counts[4][indices // VOCABULARY].astype(np.float64) + 10)
        out[begin:begin + block] = np.log(prob).astype(np.float32)
    return out


def save_char_language(table, path=DEFAULT_MODEL):
    """Write the table in a compressed NumPy archive."""
    table = np.asarray(table)
    if table.shape != (SIZE,) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError("Expected a finite float32 array with 27**5 entries")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, log_probabilities=table)


@lru_cache(maxsize=1)
def load_char_language(path=DEFAULT_MODEL):
    """Return a cached, read-only 1D float32 table indexed by base-27 grams."""
    with np.load(path, allow_pickle=False) as data:
        table = data['log_probabilities']
    if table.shape != (SIZE,) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError(f"Invalid character language model: {path}")
    table.flags.writeable = False
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    table = build_char_language(args.archive)
    save_char_language(table, args.output)
    print(f'Wrote {len(table):,} log probabilities to {args.output}')


if __name__ == '__main__':
    main()
