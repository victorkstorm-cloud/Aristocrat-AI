"""Decode Science Olympiad Aristocrat puzzles in a terminal."""

import argparse
import re
from functools import lru_cache
from pathlib import Path

from string import ascii_lowercase as ALPHABET

import numpy as np


LETTER_IDS = {letter: index for index, letter in enumerate(ALPHABET, start=1)}
ALPHABET_SIZE = 26
SPACE_ID = 27
SPACE_INDEX = SPACE_ID - 1
OFFSETS = (-3, -2, -1, 1, 2, 3)
CONTEXT_RADIUS = 6
MODEL_FILES = ("aristocrat_weights.npz", "aristocrat_refiner.npz")


def extract_features(tokens):
    """Describe each ciphertext letter using frequency and nearby letters."""
    letters = tokens[tokens != SPACE_INDEX]
    counts = np.bincount(letters, minlength=ALPHABET_SIZE)
    # Tied counts share a rank, so changing the spelling of the key
    # cannot change the feature values of a particular letter.
    ranks = (counts[None, :] > counts[:, None]).sum(axis=1)
    denominators = np.maximum(counts, 1)
    frequency = counts / len(letters)
    # Spaces have their own bin and never affect the letter ranks.
    neighbor_ranks = np.append(ranks, ALPHABET_SIZE)
    features = [
        frequency[:, None],
        (ranks / (ALPHABET_SIZE - 1))[:, None],
        (np.log1p(counts) / 8)[:, None],
        np.full((ALPHABET_SIZE, 1), np.log1p(len(letters)) / 8),
    ]

    for offset in OFFSETS:
        distance = abs(offset)
        before, after = tokens[:-distance], tokens[distance:]
        center, neighbor = (before, after) if offset > 0 else (after, before)
        is_letter = center != SPACE_INDEX
        center, neighbor = center[is_letter], neighbor[is_letter]
        transitions = np.zeros((ALPHABET_SIZE, ALPHABET_SIZE + 1))
        np.add.at(transitions, (center, neighbor_ranks[neighbor]), 1)
        repeats = np.bincount(center[center == neighbor], minlength=ALPHABET_SIZE)
        features.append(transitions / denominators[:, None])
        features.append((repeats / denominators)[:, None])

    # Bucket the distances to both ends of the word, from zero to four or more.
    positions = np.arange(len(tokens))
    spaces = tokens == SPACE_INDEX
    previous_space = np.maximum.accumulate(np.where(spaces, positions, -1))
    next_space = np.minimum.accumulate(np.where(spaces, positions, len(tokens))[::-1])[::-1]
    from_start = np.minimum(positions[~spaces] - previous_space[~spaces] - 1, 4)
    from_end = np.minimum(next_space[~spaces] - positions[~spaces] - 1, 4)
    word_layout = np.zeros((ALPHABET_SIZE, 25))
    np.add.at(word_layout, (letters, from_start * 5 + from_end), 1)
    features.append(word_layout / denominators[:, None])

    return np.concatenate(features, axis=1).astype(np.float32), counts


def minimum_cost_assignment(cost):
    """Return the column chosen for each row by the Hungarian algorithm."""
    cost = np.asarray(cost)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Assignment cost must be a square matrix.")
    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("Assignment cost contains invalid values.")
    size = len(cost)
    row_potential = np.zeros(size + 1)
    col_potential = np.zeros(size + 1)
    column_to_row = np.zeros(size + 1, dtype=int)
    previous_column = np.zeros(size + 1, dtype=int)

    for row in range(1, size + 1):
        column_to_row[0] = row
        column = 0
        distances = np.full(size + 1, np.inf)
        visited = np.zeros(size + 1, dtype=bool)

        while True:
            visited[column] = True
            active_row = column_to_row[column]
            free = np.flatnonzero(~visited)
            reduced_cost = (
                cost[active_row - 1, free - 1]
                - row_potential[active_row]
                - col_potential[free]
            )
            improved = reduced_cost < distances[free]
            distances[free[improved]] = reduced_cost[improved]
            previous_column[free[improved]] = column

            next_column = free[np.argmin(distances[free])]
            delta = distances[next_column]
            if not np.isfinite(delta):
                raise ValueError("Assignment has no feasible solution.")
            row_potential[column_to_row[visited]] += delta
            col_potential[visited] -= delta
            distances[free] -= delta
            column = next_column
            if column_to_row[column] == 0:
                break

        while column:
            prior = previous_column[column]
            column_to_row[column] = column_to_row[prior]
            column = prior

    return np.argsort(column_to_row[1:])


def predict_key(scores):
    """Turn ciphertext-to-plaintext scores into a plaintext-to-ciphertext key."""
    cost = -scores.astype(np.float64, copy=True)
    # All training keys map each letter to a different letter.
    np.fill_diagonal(cost, np.inf)
    cipher_to_plain = minimum_cost_assignment(cost)
    return np.argsort(cipher_to_plain)


def tokenize_ciphertext(ciphertext: str) -> list[int]:
    """Match corpus word boundaries while keeping contractions together."""
    tokens = []
    text = ciphertext.lower().replace("'", "").replace("’", "")
    for word in re.findall(r"[a-z]+", text):
        if tokens:
            tokens.append(SPACE_ID)
        tokens.extend(LETTER_IDS[char] for char in word)
    return tokens


def predict_logits(features, weights):
    """Run the trained model's two SiLU layers and linear output layer."""
    for kernel, bias in zip(weights[:-2:2], weights[1:-2:2]):
        values = features @ kernel + bias
        features = values / (1 + np.exp(-np.clip(values, -80, 80)))
    return features @ weights[-2] + weights[-1]


def first_guess(tokens, weights):
    features, counts = extract_features(tokens - 1)
    scores = predict_logits(features, weights)
    scores[counts == 0] = 0
    return predict_key(scores), scores


def apply_key(ciphertext, plain_to_cipher):
    cipher_to_plain = np.argsort(plain_to_cipher)
    decoded_alphabet = "".join(ALPHABET[index] for index in cipher_to_plain)
    table = str.maketrans(
        ALPHABET + ALPHABET.upper(), decoded_alphabet + decoded_alphabet.upper()
    )
    return ciphertext.translate(table)


def guess_tokens(tokens, plain_to_cipher):
    cipher_to_guess = np.append(np.argsort(plain_to_cipher) + 1, SPACE_ID)
    return cipher_to_guess[tokens - 1]


def context_windows(tokens):
    padded = np.pad(tokens, CONTEXT_RADIUS)
    width = 2 * CONTEXT_RADIUS + 1
    return np.lib.stride_tricks.sliding_window_view(padded, width)


def refiner_logits(windows, weights):
    embedding, *layers = weights
    features = embedding[windows].reshape(len(windows), -1)
    return predict_logits(features, layers)


def correction_scores(tokens, first_key, refiner_weights):
    guessed = guess_tokens(tokens, first_key)
    letter_positions = tokens != SPACE_ID
    windows = context_windows(guessed)[letter_positions]
    # Identical contexts produce identical logits.  Reuse them when puzzles
    # contain repeated words or phrases, while preserving position order.
    unique_windows, inverse = np.unique(windows, axis=0, return_inverse=True)
    scores = refiner_logits(unique_windows, refiner_weights)[inverse.reshape(-1)]
    cipher_letters = tokens[letter_positions] - 1
    totals = np.zeros((ALPHABET_SIZE, ALPHABET_SIZE), dtype=np.float32)
    np.add.at(totals, cipher_letters, scores)
    counts = np.bincount(cipher_letters, minlength=ALPHABET_SIZE)
    return totals / np.maximum(counts, 1)[:, None]


def decoded_key_identity(key, observed):
    return np.argsort(key)[observed].tobytes()


def refinement_candidates(tokens, first_key, scores, refiner_weights, strength, pass_count):
    """Yield the initial key and one result for every requested pass."""
    observed = np.unique(tokens[tokens != SPACE_ID] - 1)
    visited = {decoded_key_identity(first_key, observed)}
    current = first_key
    yield current
    for pass_index in range(pass_count):
        corrections = correction_scores(tokens, current, refiner_weights)
        # Try a stronger correction if the usual blend repeats an earlier guess.
        for multiplier in (1.0, 2.0):
            candidate = predict_key(scores + multiplier * strength * corrections)
            identity = decoded_key_identity(candidate, observed)
            if identity not in visited:
                visited.add(identity)
                current = candidate
                break
        else:
            # Neither blend can advance the key.  With the same current key
            # and visited set, future passes produce the same result.
            yield current
            for _ in range(pass_index + 1, pass_count):
                yield current
            return
        yield current


def language_indices(tokens):
    symbols = np.pad(tokens.astype(int) - 1, (3, 1), constant_values=SPACE_INDEX)
    return (((symbols[:-3] * SPACE_ID + symbols[1:-2]) * SPACE_ID
             + symbols[2:-1]) * SPACE_ID + symbols[3:])


def candidate_scores(tokens, key, scores, language):
    decoded = guess_tokens(tokens, key)
    language_score = language[language_indices(decoded)].mean()
    positions = tokens != SPACE_ID
    evidence = scores[tokens[positions] - 1, decoded[positions] - 1].mean()
    return float(language_score), float(evidence)


@lru_cache(maxsize=1)
def get_language_search():
    """Build the word-pattern index once per process, on first use."""
    from build_lexicon import load_word_counts
    from word_search import WordSearch

    return WordSearch(load_word_counts())


def solve_tokens(tokens, weights, refiner_weights, strength, language, evidence_weight, pass_count,
                 search=None):
    """Return first, neural-refined, and final keys from ciphertext alone."""
    if pass_count < 0:
        raise ValueError("Refinement passes must be nonnegative.")
    first_key, scores = first_guess(tokens, weights)
    if not pass_count:
        return first_key, first_key, first_key
    refined_key = first_key
    best_score = -np.inf
    candidates = refinement_candidates(tokens, first_key, scores, refiner_weights, strength, pass_count)
    previous_key = None
    for key in candidates:
        if key is previous_key:
            # The generator only repeats an object after it has converged.
            break
        previous_key = key
        fluency, evidence = candidate_scores(tokens, key, scores, language)
        quality = fluency + evidence_weight * evidence
        if quality > best_score:
            best_score = quality
            refined_key = key
    final_key = refined_key
    if search is not None:
        from char_language import load_char_language
        from search_keys import polish_key

        def score_key(candidate):
            fluency, evidence = candidate_scores(tokens, candidate, scores, language)
            return fluency + evidence_weight * evidence

        _, word_key = search.search(tokens, refined_key, score_key)[0]
        local_key = polish_key(tokens, word_key, scores, language, evidence_weight, search.counts)
        final_key = polish_key(tokens, local_key, scores, language, evidence_weight, search.counts,
                               char_language=load_char_language())
    return first_key, refined_key, final_key


def decode(ciphertext, weights, refiner_weights, strength, language, evidence_weight, pass_count,
           *, neural_only=False):
    tokens = np.asarray(tokenize_ciphertext(ciphertext), dtype=int)
    if not len(tokens):
        raise ValueError("Enter a ciphertext containing letters A-Z.")
    search = get_language_search() if pass_count > 0 and not neural_only else None
    first_key, _, final_key = solve_tokens(tokens, weights, refiner_weights, strength,
                                           language, evidence_weight, pass_count, search)
    return apply_key(ciphertext, first_key), apply_key(ciphertext, final_key), pass_count


def read_ciphertext():
    ciphertext = input("\nCiphertext> ")
    if ciphertext.strip().lower() != "/paste":
        return ciphertext

    print("Paste your ciphertext across multiple lines. Type /solve on its own line to decode.")
    lines = []
    while True:
        line = input()
        command = line.strip().lower()
        if command == "/solve":
            return "\n".join(lines)
        if command == "/quit":
            return line
        lines.append(line)


def load_main_weights(path):
    with np.load(path, allow_pickle=False) as archive:
        weights = []
        for index in range(6):
            weight = archive[f"arr_{index}"].astype(np.float32)
            scale = f"scale_{index}"
            if scale in archive:
                weight *= archive[scale]
            weights.append(weight)
    return tuple(weights)


def default_model_dir():
    script_dir = Path(__file__).resolve().parent
    bundled = script_dir / "models" / "rtx5090"
    return bundled if all((bundled / f).is_file() for f in MODEL_FILES) else script_dir


def load_model(model_dir):
    """Load a model pair for repeated inference or evaluation."""
    model_dir = Path(model_dir)
    model_path, refiner_path = (model_dir / name for name in MODEL_FILES)
    missing = [str(path) for path in (model_path, refiner_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("model files not found: " + ", ".join(missing))
    weights = load_main_weights(model_path)
    with np.load(refiner_path, allow_pickle=False) as archive:
        refiner_weights = tuple(archive[f"weight_{i}"].astype(np.float32) for i in range(7))
        strength = float(archive["strength"].item())
        language = archive["language"]
        evidence_weight = float(archive["evidence_weight"].item())
        default_passes = int(archive["max_passes"].item())
    return weights, refiner_weights, strength, language, evidence_weight, default_passes


def main():
    parser = argparse.ArgumentParser(description="Decode Science Olympiad Aristocrat puzzles.")
    parser.add_argument(
        "--passes", type=int, metavar="N",
        help="run exactly N refinement passes per puzzle (default: model setting)",
    )
    parser.add_argument(
        "--model-dir", type=Path, metavar="PATH",
        help=("directory containing aristocrat_weights.npz and aristocrat_refiner.npz "
              "(default: models/rtx5090 beside this script when both files exist, "
              "otherwise the script directory)"),
    )
    parser.add_argument("--neural-only", action="store_true",
                        help="use neural refinement alone, skipping the language search")
    args = parser.parse_args()
    if args.passes is not None and args.passes < 0:
        parser.error("--passes must be zero or greater")

    try:
        weights, refiner_weights, strength, language, evidence_weight, default_passes = load_model(
            args.model_dir if args.model_dir is not None else default_model_dir()
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    pass_count = default_passes if args.passes is None else args.passes
    print("Aristocrat Decoder")
    print("Paste a cipher and press Enter. Use /paste for multiple lines or /quit to exit.")

    while True:
        try:
            ciphertext = read_ciphertext()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if ciphertext.strip().lower() == "/quit":
            break
        if not ciphertext.strip():
            continue
        try:
            first, refined, passes = decode(
                ciphertext, weights, refiner_weights, strength, language, evidence_weight, pass_count,
                neural_only=args.neural_only,
            )
        except ValueError as error:
            print(error)
            continue
        print(f"\nFirst pass\n{first}\n\nRefined\n{refined}\n\nRefinement passes: {passes}")


if __name__ == "__main__":
    main()
