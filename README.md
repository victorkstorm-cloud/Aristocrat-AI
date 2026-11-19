# Aristocrat AI

Aristocrat AI solves English monoalphabetic substitution puzzles of the kind used in Science Olympiad Codebusters. It expects a substitution that maps every letter to a different letter. Decoding runs on the CPU with NumPy. The PyTorch trainer defaults to a CUDA GPU with BF16 support.

## Install and decode

Use Python 3.11 or later. To install and launch the decoder from the repository root on Windows, run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe aristocrat_chat.py
```

On macOS or Linux, use `python3 -m venv .venv` and `.venv/bin/python` in place of the Windows paths. Paste a cipher at the prompt. Use `/paste` for multiple lines and `/quit` to exit.

The default decoder produces a first guess, refines it with the neural models, then searches for a consistent substitution across the entire puzzle. It proposes words that match the repeated-letter pattern of each ciphertext word and tries key swaps using word frequencies and four- and five-character English probabilities. The output preserves the input's case and punctuation.

Use `--neural-only` to run the previous refinement method. `--passes N` sets the number of neural refinement passes. Setting it to zero returns the first guess and skips the language search. The CLI selects the trained pair in `models/rtx5090/` by default. Use `--model-dir .` to select the older pair or `--model-dir PATH` for another exported pair.

## Whole-puzzle accuracy

An exact solve gets every letter right. Each comparison below uses the same ciphertexts, same neural weights, and three neural passes on both sides.

| Evaluation | Neural refinement | With language search | Final letter accuracy |
| --- | ---: | ---: | ---: |
| Original Chesterton benchmark, 1,200 puzzles | 556 (46.3%) | **1,092 (91.0%)** | 99.80% |
| Fresh Blake and King James Bible puzzles, 600 total | 235 (39.2%) | **480 (80.0%)** | 99.48% |
| Northridge invitational, 14 English Aristocrats | 4 (28.6%) | **13 (92.9%)** | — |

On the original set, the search corrected 547 previously missed puzzles and changed 11 previously exact answers into misses. On the fresh set, it solved 251/300 Blake puzzles and 229/300 Bible puzzles. The contest check covers all 14 eligible English plaintext-recovery questions across the 2025–26 Northridge Division B and C sample exams. They contain 75–88 letters each. The solver received only the ciphertext, reconstructed using the organizer's recorded substitution for each question. [Organizer's sample tests](https://toebes.com/codebusters/samples.html)

The word counts and five-character model were built from the 11 named Gutenberg decoder/refiner training books. The builders exclude the bundled text, validation authors, test authors, and both fresh books. Search settings were selected on validation puzzles before these three evaluations. The previous neural weights have incomplete corpus provenance. Different authors, short quotations, and unusual names can change the exact-solve rate.

To reproduce the evaluations, run these commands from the repository root. The first two verify and cache the Gutenberg archive. The third downloads and verifies the organizer's exam files. A single worker uses less memory.

```sh
python evaluate_solver.py --split test --count 1200 --seed 5093 --workers 4
python evaluate_solver.py --split fresh --count 600 --seed 20261119 --workers 4
python benchmarks/evaluate_northridge.py
```

The generated-puzzle reports include seeds, data and model hashes, confidence intervals, and counts of fixed and lost solves. [Benchmark reports](benchmarks/results/)

## Neural training benchmark

The included `models/rtx5090/benchmark.json` records one run on 1,200 generated puzzles from three held-out G. K. Chesterton books, using three refinement passes. Exact solves require every letter to be correct.

| Model | Initial letter accuracy | Refined letter accuracy | Initial exact solves | Refined exact solves |
| --- | ---: | ---: | ---: | ---: |
| Previous | 89.22% | 94.48% | 84 | 535 |
| RTX 5090 run | 90.40% | 94.80% | 109 | 556 |

Letter accuracy counts individual letters across the generated test set. The bundled data.json file has no source manifest, and the previous model's complete training corpus is unavailable. Prior exposure to the test books remains unknown. The recorded best epochs were 14 for the decoder and 3 for the refiner.

## Training

Install training dependencies and a PyTorch build appropriate for your hardware using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/). The local GPU smoke test used Python 3.13, PyTorch 2.8.0 with CUDA 12.8, and an RTX 5090. After installing PyTorch in the same environment, run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe train_5090.py
```

By default, training makes 120,000 decoder and 35,000 refiner puzzles, evaluates on separate validation and test splits, and writes checkpoints, a standalone `chat` export with both language resources, and a neural-only `benchmark.json` under `run_5090/`. The default maximums are 60 decoder and 150 refiner epochs; validation loss selects the exported epochs. Run the same command again to resume from the last completed checkpoint. Training saves progress every five epochs and at the end of each stage. Use a new `--run-dir` when changing data or training settings. Checkpoint files can require several gigabytes. For a smaller pipeline check use `--smoke`; `--prepare-only` checks data preparation without PyTorch. `--device cpu` is available, although full training is computationally demanding.

The default data preparation downloads the [NLTK Gutenberg corpus](https://github.com/nltk/nltk_data) on first use, verifies its pinned SHA-256, and caches it locally in `corpus_cache/`. `--bundled-only` avoids the download and divides the included `data.json` text into contiguous splits, so results are not directly comparable to the held-out-books benchmark. Add text to the decoder training split with `--corpus path/to/text.txt` or a UTF-8 JSON file with a top-level `text` string; the flag can be repeated. Check the source and usage terms of any corpus before contributing or using it.

Run the regression and distributed-model checks with

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tests/check_release.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance.
