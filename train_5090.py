"""Train a wider Aristocrat decoder and refiner, then export chat weights."""

import argparse
import gc
import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np

import aristocrat_chat as chat
from training_data import data_signature, load_documents, make_puzzles, supervised_symbols

BASE = Path(__file__).resolve().parent
MAIN_WIDTHS = (10240, 10240)
REFINER_WIDTHS = (2560, 2560)


def widened_layers(weights, widths):
    """Clone hidden neurons and divide each outgoing weight among its copies."""
    first, first_bias, middle, middle_bias, last, last_bias = weights
    units1, units2 = len(first_bias), len(middle_bias)
    index1 = np.arange(widths[0]) % units1
    index2 = np.arange(widths[1]) % units2
    copies1 = np.bincount(index1)[index1].astype(np.float32)
    copies2 = np.bincount(index2)[index2].astype(np.float32)
    widened_middle = middle[np.ix_(index1, index2)]
    widened_middle /= copies1[:, None]
    return (
        first[:, index1],
        first_bias[index1],
        widened_middle,
        middle_bias[index2],
        last[index2] / copies2[:, None],
        last_bias,
    )


def run_training(args, documents, fingerprint, legacy_fingerprint=None):
    import torch
    from scipy.optimize import linear_sum_assignment
    from torch import nn
    from torch.nn import functional as F

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install the CUDA 12.8 PyTorch wheel and check nvidia-smi.")
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_bf16_supported():
        raise SystemExit("This run needs BF16 support. Check the NVIDIA driver and PyTorch build.")
    if use_cuda:
        print(f"Training on {torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probe = torch.ones((32, 32), device=device)
            (probe @ probe).sum().item()

    class Network(nn.Module):
        def __init__(self, widths, refiner=False, dropout=0.12):
            super().__init__()
            self.embedding = nn.Embedding(28, 24) if refiner else None
            self.layers = nn.ModuleList(
                (nn.Linear(312 if refiner else 197, widths[0]),
                 nn.Linear(widths[0], widths[1]), nn.Linear(widths[1], 26))
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, features):
            if self.embedding is not None:
                features = self.embedding(features).flatten(1)
            for layer in self.layers[:-1]:
                features = self.dropout(F.silu(layer(features)))
            return self.layers[-1](features)

        @torch.no_grad()
        def set_weights(self, weights, widen=False):
            if self.embedding is not None:
                embedding, *weights = weights
                self.embedding.weight.copy_(torch.from_numpy(embedding))
            if widen:
                weights = widened_layers(weights, tuple(layer.out_features for layer in self.layers[:2]))
            for layer, kernel, bias in zip(self.layers, weights[::2], weights[1::2]):
                layer.weight.copy_(torch.from_numpy(kernel.T))
                layer.bias.copy_(torch.from_numpy(bias))

        def numpy_weights(self):
            weights = []
            if self.embedding is not None:
                weights.append(self.embedding.weight.detach().cpu().numpy())
            for layer in self.layers:
                weights.append(layer.weight.detach().cpu().numpy().T)
                weights.append(layer.bias.detach().cpu().numpy())
            return weights

    def feature_tensor(values, model):
        dtype = torch.int32 if model.embedding is not None else torch.float32
        return torch.as_tensor(values, device=device).to(dtype)

    @torch.no_grad()
    def infer(model, values, mixed=True):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_cuda and mixed):
            return model(feature_tensor(values, model)).float().cpu().numpy()

    def atomic_save(value, destination):
        temp = destination.with_suffix(".tmp")
        torch.save(value, temp)
        temp.replace(destination)

    def load(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint and state["fingerprint"] != legacy_fingerprint:
            raise SystemExit(f"Data or training settings changed. Use a new --run-dir for {path.name}.")
        return state

    def fit(name, model, build_data, epochs, patience, batch, rate):
        best_path = args.run_dir / f"{name}_best.pt"
        last_path = args.run_dir / f"{name}_last.pt"
        model.to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: {parameters:,} parameters", flush=True)
        last = load(last_path) if last_path.is_file() else None
        # New checkpoints refer to an immutable best snapshot. A snapshot may be
        # written before the last checkpoint commits, so it must not be selected
        # merely because it has the highest epoch number.
        if last and "best_epoch" in last:
            best_epoch = last["best_epoch"]
            committed_best = args.run_dir / f"{name}_best_{best_epoch}.pt"
        else:
            committed_best = best_path
        best = load(committed_best) if committed_best.is_file() else None
        start = last["epoch"] if last else 0
        if last and (best is None or best["epoch"] > start):
            raise SystemExit(f"{name} checkpoints are inconsistent; use a new --run-dir or restore a matching best checkpoint.")
        # A best-only checkpoint is an uncommitted partial run. Start afresh.
        if not last:
            best = None
        best_loss = best["loss"] if best else float("inf")
        best_epoch = best["epoch"] if best else 0
        if last and last["done"]:
            model.load_state_dict(best["model"])
            print(f"{name}: restored best epoch {best_epoch} (validation loss {best_loss:.4f})", flush=True)
            return model.eval(), best_epoch
        if last and "best_epoch" not in last and not last["done"]:
            # Upgrade legacy checkpoints before writing a versioned reference.
            committed_best = args.run_dir / f"{name}_best_{best_epoch}.pt"
            atomic_save(best, committed_best)
        del best
        candidate_best = committed_best if last else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, fused=use_cuda)
        if last:
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            del last

        train_arrays, valid_arrays = build_data()
        x, y = (torch.as_tensor(array, device=device) for array in train_arrays)
        val_x, val_y = (torch.as_tensor(array, device=device) for array in valid_arrays)
        del train_arrays, valid_arrays
        minimum_epochs = (3 * epochs + 4) // 5
        for epoch in range(start + 1, epochs + 1):
            torch.manual_seed(args.seed + epoch)
            if epoch > 1:
                progress = (epoch - 1) / max(epochs - 1, 1)
                optimizer.param_groups[0]["lr"] = rate * (1 - 0.9 * progress)
            model.train()
            order = torch.randperm(len(x), device=device)
            for indices in order.split(batch):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_cuda):
                    output = model(feature_tensor(x[indices], model))
                    loss = F.cross_entropy(output, y[indices].long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()

            model.eval()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_cuda):
                val_loss = torch.zeros((), device=device)
                for val_features, val_labels in zip(val_x.split(batch), val_y.split(batch)):
                    output = model(feature_tensor(val_features, model))
                    val_loss += F.cross_entropy(output, val_labels.long(), reduction="sum")
                current_loss = val_loss.item() / len(val_y)
            if not np.isfinite(current_loss):
                raise RuntimeError(f"{name} loss became non-finite at epoch {epoch}.")
            if current_loss < best_loss:
                best_loss, best_epoch = current_loss, epoch
                candidate_best = args.run_dir / f"{name}_best_{best_epoch}.pt"
                atomic_save(dict(model=model.state_dict(), loss=best_loss,
                                 epoch=best_epoch, fingerprint=fingerprint), candidate_best)
            print(f"{name} epoch {epoch:3d}/{epochs}  val loss {current_loss:.4f}  "
                  f"best epoch {best_epoch}", flush=True)
            done = epoch == epochs or (epoch >= minimum_epochs and epoch - best_epoch >= patience)
            if epoch % 5 == 0 or done:
                atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 epoch=epoch, done=done, fingerprint=fingerprint,
                                 best_epoch=best_epoch), last_path)
                # Only remove older snapshots after the referencing checkpoint is durable.
                for snapshot in args.run_dir.glob(f"{name}_best_*.pt"):
                    if snapshot != candidate_best:
                        snapshot.unlink()
            if done:
                break
        model.load_state_dict(load(args.run_dir / f"{name}_best_{best_epoch}.pt")["model"])
        return model.eval(), best_epoch

    def key_from_scores(scores):
        cost = -scores.astype(np.float64)
        np.fill_diagonal(cost, np.inf)
        _, cipher_to_plain = linear_sum_assignment(cost)
        return np.argsort(cipher_to_plain)

    def main_predictions(puzzles, main, mixed=True):
        assign = key_from_scores if mixed else chat.predict_key
        for start in range(0, len(puzzles), 128):
            group = puzzles[start:start + 128]
            rows, counts = zip(*(chat.extract_features(cipher - 1) for cipher, _ in group))
            flat = np.concatenate(rows, axis=0)
            logits = infer(main, flat, mixed).reshape(-1, 26, 26)
            logits[np.asarray(counts) == 0] = 0
            for (cipher, plain), scores in zip(group, logits):
                yield cipher, plain, scores, assign(scores)

    def correction_batch(tokens_list, keys, model, mixed=True):
        positions = [tokens != chat.SPACE_ID for tokens in tokens_list]
        windows = [
            chat.context_windows(chat.guess_tokens(tokens, key))[mask]
            for tokens, key, mask in zip(tokens_list, keys, positions)
        ]
        boundaries = np.cumsum([len(window) for window in windows])
        all_logits = infer(model, np.concatenate(windows, axis=0), mixed)
        logits_by_puzzle = np.split(all_logits, boundaries[:-1])
        results = []
        for tokens, mask, logits in zip(tokens_list, positions, logits_by_puzzle):
            symbols = tokens[mask] - 1
            totals = np.zeros((26, 26), dtype=np.float32)
            np.add.at(totals, symbols, logits)
            results.append(totals / np.maximum(np.bincount(symbols, minlength=26), 1)[:, None])
        return windows, results

    def refiner_examples(puzzles, main, old_refiner, old_strength):
        # Half of the puzzles also contribute a partially corrected guess.
        counts = [int(np.count_nonzero(cipher != chat.SPACE_ID)) for cipher, _ in puzzles]
        rows = sum(counts) + sum(counts[::2])
        contexts = np.empty((rows, 13), dtype=np.uint8)
        labels = np.empty(rows, dtype=np.uint8)
        offset = 0
        stream = main_predictions(puzzles, main)
        while group := list(islice(stream, 128)):
            ciphers, plains, scores, keys = zip(*group)
            second_indices = range(0, len(group), 2)
            second_ciphers = [ciphers[index] for index in second_indices]
            first_windows = [
                chat.context_windows(chat.guess_tokens(cipher, key))[cipher != chat.SPACE_ID]
                for cipher, key in zip(ciphers, keys)
            ]
            _, corrections = correction_batch(second_ciphers, [keys[index] for index in second_indices], old_refiner)
            second_keys = [
                key_from_scores(scores[index] + old_strength * correction)
                for index, correction in zip(second_indices, corrections)
            ]
            second_windows = [
                chat.context_windows(chat.guess_tokens(cipher, key))[cipher != chat.SPACE_ID]
                for cipher, key in zip(second_ciphers, second_keys)
            ]
            targets = [plain[plain != chat.SPACE_ID] - 1 for plain in plains]
            all_targets = targets + [targets[index] for index in second_indices]
            for context, target in zip(first_windows + second_windows, all_targets):
                end = offset + len(target)
                contexts[offset:end] = context
                labels[offset:end] = target
                offset = end
        return contexts[:offset], labels[:offset]

    def benchmark(predictions, refiner, strength, language, evidence, passes):
        solved_first = solved_final = correct_first = correct_final = total = puzzles = 0
        stream = iter(predictions)
        while group := list(islice(stream, 128)):
            ciphers, plains, initial_scores, first_keys = zip(*group)
            letter_masks = [cipher != chat.SPACE_ID for cipher in ciphers]
            observed = [np.unique(cipher[mask] - 1) for cipher, mask in zip(ciphers, letter_masks)]
            visited = [{chat.decoded_key_identity(key, symbols)}
                       for key, symbols in zip(first_keys, observed)]
            current_keys = list(first_keys)
            best_keys = list(first_keys)
            qualities = [
                score_candidate(cipher, key, scores, language, evidence)
                for cipher, key, scores in zip(ciphers, first_keys, initial_scores)
            ]
            for _ in range(passes):
                _, corrections = correction_batch(ciphers, current_keys, refiner, mixed=False)
                for index, correction in enumerate(corrections):
                    for multiplier in (1.0, 2.0):
                        candidate = chat.predict_key(initial_scores[index] + multiplier * strength * correction)
                        identity = chat.decoded_key_identity(candidate, observed[index])
                        if identity not in visited[index]:
                            visited[index].add(identity)
                            current_keys[index] = candidate
                            break
                    quality = score_candidate(ciphers[index], current_keys[index], initial_scores[index],
                                              language, evidence)
                    if quality > qualities[index]:
                        qualities[index] = quality
                        best_keys[index] = current_keys[index]
            for cipher, plain, initial, final, mask in zip(ciphers, plains, first_keys, best_keys, letter_masks):
                first_correct = chat.guess_tokens(cipher, initial)[mask] == plain[mask]
                final_correct = chat.guess_tokens(cipher, final)[mask] == plain[mask]
                solved_first += bool(first_correct.all())
                solved_final += bool(final_correct.all())
                correct_first += int(first_correct.sum())
                correct_final += int(final_correct.sum())
                total += len(first_correct)
                puzzles += 1
        return {
            "puzzles": puzzles,
            "first_solved": solved_first,
            "refined_solved": solved_final,
            "first_letter_accuracy": correct_first / total,
            "refined_letter_accuracy": correct_final / total,
        }

    torch.manual_seed(args.seed)
    old_main_weights = chat.load_main_weights(BASE / "aristocrat_weights.npz")
    with np.load(BASE / "aristocrat_refiner.npz", allow_pickle=False) as old:
        old_refiner_weights = tuple(old[f"weight_{index}"].astype(np.float32) for index in range(7))
        old_strength = float(old["strength"])
        old_passes = int(old["max_passes"])
        evidence = float(old["evidence_weight"])
        language = old["language"]
    old_refiner = Network((716, 703), refiner=True)
    old_refiner.set_weights(old_refiner_weights)
    old_refiner.to(device).eval()

    main_widths = (64, 64) if args.smoke else MAIN_WIDTHS
    refiner_widths = (64, 64) if args.smoke else REFINER_WIDTHS
    main = Network(main_widths)
    if not args.smoke and not (args.run_dir / "main_last.pt").exists():
        main.set_weights(old_main_weights, widen=True)
    val_count, test_count = (12, 12) if args.smoke else (600, 1200)
    validation = make_puzzles(documents["validation"], val_count, args.seed + 2)
    test = make_puzzles(documents["test"], test_count, args.seed + 3)

    def main_data():
        print("Building decoder training features", flush=True)
        puzzles = make_puzzles(documents["main"], args.main_examples, args.seed)
        return supervised_symbols(puzzles), supervised_symbols(validation)

    main, main_epoch = fit("main", main, main_data, args.main_epochs, args.main_patience,
                           args.main_batch, 0.00008 if not args.smoke else 0.001)
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    main_payload, rounded_main = pack_main(main.numpy_weights())
    main.set_weights(rounded_main)
    del rounded_main
    refiner = Network(refiner_widths, refiner=True)
    if not args.smoke and not (args.run_dir / "refiner_last.pt").exists():
        refiner.set_weights(old_refiner_weights, widen=True)

    def refiner_data():
        print("Decoding refiner training puzzles with the trained decoder", flush=True)
        puzzles = make_puzzles(documents["refiner"], args.refiner_examples, args.seed + 1)
        return (
            refiner_examples(puzzles, main, old_refiner, old_strength),
            refiner_examples(validation, main, old_refiner, old_strength),
        )

    refiner, refiner_epoch = fit("refiner", refiner, refiner_data, args.refiner_epochs,
                                 args.refiner_patience, args.refiner_batch,
                                 0.00015 if not args.smoke else 0.001)
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # Round to the shipping precision for the final benchmark.
    refiner_payload = [weight.astype(np.float16) for weight in refiner.numpy_weights()]
    refiner.set_weights([weight.astype(np.float32) for weight in refiner_payload])
    val_predictions = list(main_predictions(validation, main, mixed=False))
    choices = (1.0,) if args.smoke else (0.5, 1.0, 2.0, 4.0, 8.0)
    validation_metrics = {
        strength: benchmark(val_predictions, refiner, strength, language, evidence, old_passes)
        for strength in choices
    }
    strength = max(
        choices, key=lambda value: (validation_metrics[value]["refined_solved"],
                                    validation_metrics[value]["refined_letter_accuracy"])
    )

    old_main = Network((5710, 6261))
    old_main.set_weights(old_main_weights)
    old_main.to(device).eval()
    results = {
        "test_split": "bundled contiguous holdout" if args.bundled_only else "held-out books",
        "main_best_epoch": main_epoch,
        "refiner_best_epoch": refiner_epoch,
        "refinement_passes": old_passes,
        "refiner_strength": strength,
        "previous_checkpoint": benchmark(main_predictions(test, old_main, mixed=False), old_refiner,
                                         old_strength, language, evidence, old_passes),
        "new_checkpoint": benchmark(main_predictions(test, main, mixed=False), refiner, strength,
                                    language, evidence, old_passes),
    }
    output = args.run_dir / "chat"
    output.mkdir(exist_ok=True)
    save_archive(output / "aristocrat_weights.npz", main_payload)
    refiner_arrays = {f"weight_{index}": weight for index, weight in enumerate(refiner_payload)}
    save_archive(output / "aristocrat_refiner.npz",
                 dict(refiner_arrays, strength=strength, language=language,
                      evidence_weight=evidence, max_passes=old_passes))
    (output / "aristocrat_chat.py").write_bytes((BASE / "aristocrat_chat.py").read_bytes())
    report = json.dumps(results, indent=2)
    (args.run_dir / "benchmark.json").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    print(f"Trained chat: {output / 'aristocrat_chat.py'}", flush=True)


def score_candidate(cipher, key, scores, language, evidence):
    fluency, confidence = chat.candidate_scores(cipher, key, scores, language)
    return fluency + evidence * confidence


def pack_main(weights):
    archive = {}
    rounded = []
    for index, weight in enumerate(weights):
        if index % 2:
            archive[f"arr_{index}"] = weight
            rounded.append(weight)
            continue
        # Bound temporary arrays even for the largest hidden-layer matrix.
        scale = np.empty(weight.shape[1], dtype=np.float32)
        quantized = np.empty(weight.shape, dtype=np.int8)
        restored = np.empty(weight.shape, dtype=np.float32)
        for start in range(0, weight.shape[1], 128):
            end = min(start + 128, weight.shape[1])
            block = weight[:, start:end]
            block_scale = np.maximum(np.max(np.abs(block), axis=0) / 127, np.finfo(np.float32).tiny)
            scale[start:end] = block_scale
            scratch = block / block_scale
            np.rint(scratch, out=scratch)
            np.clip(scratch, -127, 127, out=scratch)
            quantized[:, start:end] = scratch
            np.multiply(quantized[:, start:end], block_scale, out=restored[:, start:end], dtype=np.float32)
        archive[f"arr_{index}"] = quantized
        archive[f"scale_{index}"] = scale
        rounded.append(restored)
    return archive, rounded


def save_archive(path, arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=BASE / "run_5090")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bundled-only", action="store_true", help="Use only included text and local --corpus files")
    parser.add_argument("--corpus", type=Path, action="append", default=[], help="Extra UTF-8 text or JSON with a text key")
    parser.add_argument("--prepare-only", action="store_true", help="Check data and feature preparation without PyTorch")
    parser.add_argument("--smoke", action="store_true", help="Use tiny scratch models and a one-epoch run")
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--main-examples", type=int, default=120_000)
    parser.add_argument("--refiner-examples", type=int, default=35_000)
    parser.add_argument("--main-epochs", type=int, default=60)
    parser.add_argument("--refiner-epochs", type=int, default=150)
    parser.add_argument("--main-patience", type=int, default=15)
    parser.add_argument("--refiner-patience", type=int, default=25)
    parser.add_argument("--main-batch", type=int, default=2048)
    parser.add_argument("--refiner-batch", type=int, default=8192)
    args = parser.parse_args()
    if args.smoke:
        if args.run_dir == BASE / "run_5090":
            args.run_dir = BASE / "run_smoke"
        args.bundled_only = True
        args.main_epochs = args.refiner_epochs = 1
        args.main_examples = args.refiner_examples = 32
        args.main_batch = args.refiner_batch = 128
    for key, value in vars(args).items():
        if type(value) is int and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    documents = load_documents(BASE, BASE / "corpus_cache", args.bundled_only, args.corpus)
    print("Corpus words by split", {name: sum(map(len, docs)) for name, docs in documents.items()}, flush=True)
    if args.prepare_only:
        for index, (name, docs) in enumerate(documents.items()):
            puzzles = make_puzzles(docs, 100, args.seed + index)
            features, labels = supervised_symbols(puzzles)
            print(f"{name}: {len(puzzles)} puzzles, feature shape {features.shape}, {len(labels)} labels")
        return
    settings = {key: value for key, value in vars(args).items() if type(value) in (int, bool, str)}
    signature = data_signature(documents)
    legacy = json.dumps((1, signature, settings), sort_keys=True)
    model_hashes = []
    for filename in ("aristocrat_weights.npz", "aristocrat_refiner.npz"):
        with (BASE / filename).open("rb") as source:
            model_hashes.append(hashlib.file_digest(source, "sha256").hexdigest())
    fingerprint = json.dumps((2, signature, settings, model_hashes), sort_keys=True)
    baseline = ["be798fedb2eff294f35830afffaf6aae27e6f35a40646db668b24d193e05b710",
                "0fda17aac9038daa690d547c4c936bb8a5f8314008d73f1ff9241cd40a7f3e22"]
    from filelock import FileLock, Timeout

    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(args.run_dir / ".training.lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        parser.exit(1, f"Another trainer is using {args.run_dir}. Choose a different --run-dir.\n")
    try:
        run_training(args, documents, fingerprint, legacy if model_hashes == baseline else None)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
