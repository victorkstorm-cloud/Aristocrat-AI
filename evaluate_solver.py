"""Benchmark the deployed neural and search solver on reproducible puzzles."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

# Set before NumPy (also when spawned workers import this module on Windows).
if __name__ in ("__main__", "__mp_main__"):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import aristocrat_chat as chat
import build_lexicon
import training_data as data

BASE = Path(__file__).resolve().parent
FRESH_BOOKS = ("blake-poems.txt", "bible-kjv.txt")
FRESH_SEED = 20261119
_MODEL = None
_COUNTS = None


def sha256(path):
    # Git stores Python sources with LF even if a checkout uses CRLF.
    if path.suffix == ".py":
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def init_worker(model_dir):
    global _MODEL, _COUNTS
    _MODEL = chat.load_model(Path(model_dir))
    _COUNTS = chat.get_language_search()


def evaluate(item):
    book, cipher, plain = item
    weights, refiner, strength, language, evidence, passes = _MODEL
    _, neural, final = chat.solve_tokens(cipher, weights, refiner, strength, language,
                                         evidence, passes, _COUNTS)
    mask = cipher != chat.SPACE_ID
    baseline = chat.guess_tokens(cipher, neural)[mask]
    enhanced = chat.guess_tokens(cipher, final)[mask]
    target = plain[mask]
    return {"book": book, "letters": int(mask.sum()),
            "baseline_correct": int((baseline == target).sum()),
            "enhanced_correct": int((enhanced == target).sum()),
            "baseline_solved": bool(np.all(baseline == target)),
            "enhanced_solved": bool(np.all(enhanced == target)),
            "changed": bool(np.any(baseline != enhanced))}


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0., center-half), min(1., center+half)]


def summarize(rows):
    n = len(rows)
    letters = sum(r['letters'] for r in rows)
    b = sum(r['baseline_solved'] for r in rows)
    e = sum(r['enhanced_solved'] for r in rows)
    return {"puzzles": n, "letters": letters,
            "baseline": {"solved": b, "solve_rate": b/n if n else None,
                         "solve_rate_wilson_95": wilson(b,n),
                         "letter_accuracy": sum(r['baseline_correct'] for r in rows)/letters if letters else None},
            "enhanced": {"solved": e, "solve_rate": e/n if n else None,
                         "solve_rate_wilson_95": wilson(e,n),
                         "letter_accuracy": sum(r['enhanced_correct'] for r in rows)/letters if letters else None},
            "changed": sum(r['changed'] for r in rows),
            "fixed": sum(not r['baseline_solved'] and r['enhanced_solved'] for r in rows),
            "broken": sum(r['baseline_solved'] and not r['enhanced_solved'] for r in rows),
            "letter_improved": sum(r['enhanced_correct'] > r['baseline_correct'] for r in rows),
            "letter_worsened": sum(r['enhanced_correct'] < r['baseline_correct'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation','test','fresh'), default='validation')
    parser.add_argument('--count', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=None, help='default: 5092 for validation, 5093 for test, 20261119 for fresh')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--per-puzzle', action='store_true', help='include paired results for each puzzle')
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or (args.seed is not None and args.seed < 0):
        parser.error('count and workers must be positive and seed nonnegative')
    started = time.perf_counter()
    seed = args.seed if args.seed is not None else {'validation': 5092, 'test': 5093, 'fresh': FRESH_SEED}[args.split]
    model_dir = (args.model_dir or chat.default_model_dir()).resolve()
    # This verifies the archive's pinned hash before reading any fresh books.
    splits = data.read_book_archive(BASE / 'corpus_cache')
    if args.split == 'fresh':
        tasks = []
        datasets = []
        with ZipFile(BASE / 'corpus_cache' / 'gutenberg.zip') as archive:
            for i, book in enumerate(FRESH_BOOKS):
                count = args.count // 2 + (i < args.count % 2)
                book_seed = seed + i
                datasets.append({'book': book, 'count': count, 'seed': book_seed})
                if count:
                    doc = data.words(archive.read('gutenberg/' + book).decode('latin-1'))
                    tasks.extend((book, cipher, plain) for cipher, plain in data.make_puzzles([doc], count, book_seed))
    else:
        datasets = [{'books': list(data.BOOKS[args.split]), 'count': args.count, 'seed': seed}]
        tasks = [(None, cipher, plain) for cipher, plain in data.make_puzzles(splits[args.split], args.count, seed)]
    if args.workers == 1:
        init_worker(model_dir)
        rows = list(map(evaluate, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(str(model_dir),)) as pool:
            rows = list(pool.map(evaluate, tasks, chunksize=4))
    artifact_paths = {
        'model_weights': model_dir / 'aristocrat_weights.npz',
        'refiner_weights': model_dir / 'aristocrat_refiner.npz',
        'lexicon': build_lexicon.DEFAULT_LEXICON,
        'fivegram_model': BASE / 'models' / 'english_fivegrams.npz',
        'fivegram_loader_code': BASE / 'char_language.py',
        'solver_code': BASE / 'aristocrat_chat.py',
        'search_keys_code': BASE / 'search_keys.py',
        'word_search_code': BASE / 'word_search.py',
        'benchmark_code': BASE / 'evaluate_solver.py',
        'data_code': BASE / 'training_data.py',
        'lexicon_loader_code': BASE / 'build_lexicon.py',
    }
    artifacts = {label: sha256(path) for label, path in artifact_paths.items()}
    report = {'split': args.split, 'seed': seed, 'datasets': datasets,
              'archive_sha256': data.CORPUS_SHA256, 'artifacts_sha256': artifacts,
              'python_version': platform.python_version(), 'numpy_version': np.__version__,
              'workers': args.workers, 'elapsed_seconds': time.perf_counter() - started,
              'overall': summarize(rows),
              'per_book': {book: summarize([r for r in rows if r['book'] == book])
                           for book in FRESH_BOOKS} if args.split == 'fresh' else {}}
    if args.per_puzzle:
        report['per_puzzle'] = rows
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
