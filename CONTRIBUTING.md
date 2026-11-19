# Contributing

Please keep changes focused and explain their purpose and any behavior changes in the pull request. For bugs, include a reproducible example and the expected and observed behavior. Avoid committing private text, downloaded corpora, local environments, caches, or training checkpoints.

## Development setup

Create a virtual environment and install `requirements.txt` for inference, or `requirements-train.txt` for data preparation and training. Training also needs a separately installed PyTorch build suitable for your hardware; see the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). The default full training configuration expects CUDA and BF16 support. CPU mode and a small smoke run are available for pipeline checks.

From the repository root, run:

```sh
python -m unittest discover -s tests
python train_5090.py --prepare-only --bundled-only
```

For training-related changes, also consider `python train_5090.py --smoke --device cpu` or a suitable GPU smoke run. Full training is expensive and is not required for every change. The default corpus is downloaded on demand and verified; use `--bundled-only` for an offline check. Use a fresh `--run-dir` when changing training settings or data, and describe the data, settings, hardware, and evaluation procedure when reporting results.

The root model files are the previous pair; the larger released pair and its historical report live under `models/rtx5090/`. Preserve compatibility with both when changing inference. Do not replace model artifacts or benchmark results without explaining how they were produced and validating them together. Do not assume the historical held-out-books results apply to a bundled-only run.

For decoder changes, tune on the validation split. Evaluate the selected configuration once on the original test split and fresh books. Record paired exact solves, letter accuracy, and lost solves with the seeds and model hashes. Build the word and character resources from the verified training books with `python build_lexicon.py` and `python char_language.py`. The packaged resources allow offline decoding.

Please submit only text and other material you have permission to share. No project license is specified here; do not assume one without checking with the project owner.
