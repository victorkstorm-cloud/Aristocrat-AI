# Aristocrat AI

Aristocrat AI solves English monoalphabetic substitution puzzles of the kind used in Science Olympiad Codebusters. It expects a substitution that maps every letter to a different letter. Decoding runs on the CPU with NumPy. The PyTorch trainer defaults to a CUDA GPU with BF16 support.

## Install and decode

Use Python 3.11 or later. To install and launch the decoder from the repository root on Windows, run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe aristocrat_chat.py
```

On macOS or Linux, use `python3 -m venv .venv` and `.venv/bin/python` in place of the Windows paths. Paste a cipher at the prompt. Use `/paste` for multiple lines and `/quit` to exit. The program prints the initial decode and its best candidate from refinement. `--passes 0` disables refinement; a nonnegative `--passes N` sets the number of passes.

The larger trained pair and its historical benchmark are in `models/rtx5090/`. The chat selects that pair by default when present, otherwise it uses the previous pair alongside the script. To select the previous pair explicitly, run `python aristocrat_chat.py --model-dir .` from the repository root. Use `--model-dir PATH` to select another directory containing both weight files.

## Historical benchmark

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

By default, training makes 120,000 decoder and 35,000 refiner puzzles, evaluates on separate validation and test splits, and writes checkpoints, a `chat` export, and `benchmark.json` under `run_5090/`. The default maximums are 60 decoder and 150 refiner epochs; validation loss selects the exported epochs. Run the same command again to resume from the last completed checkpoint. Training saves progress every five epochs and at the end of each stage. Use a new `--run-dir` when changing data or training settings. Checkpoint files can require several gigabytes. For a smaller pipeline check use `--smoke`; `--prepare-only` checks data preparation without PyTorch. `--device cpu` is available, although full training is computationally demanding.

The default data preparation downloads the [NLTK Gutenberg corpus](https://github.com/nltk/nltk_data) on first use, verifies its pinned SHA-256, and caches it locally in `corpus_cache/`. `--bundled-only` avoids the download and divides the included `data.json` text into contiguous splits, so results are not directly comparable to the held-out-books benchmark. Add text to the decoder training split with `--corpus path/to/text.txt` or a UTF-8 JSON file with a top-level `text` string; the flag can be repeated. Check the source and usage terms of any corpus before contributing or using it.

Run the regression and distributed-model checks with

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tests/check_release.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance.
