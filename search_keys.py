"""Optional deterministic hill climbing for an Aristocrat decoder.

The objective combines the model's mean tetragram log probability and neural
score, plus an optional training-corpus word-frequency bonus. Only strict
improvements are accepted; each step examines every feasible transposition.
The word bonus is deliberately bounded to limit overfitting to a dictionary.
"""
import numpy as np

from aristocrat_chat import ALPHABET_SIZE, SPACE_ID, minimum_cost_assignment

# Mean bonus per letter. On 80 Gutenberg validation puzzles (seed 5092),
# weights 0.5, 1, 2 and 4 solved 68, 69, 69 and 69 respectively. We use 2
# rather than increase dictionary influence for a negligible accuracy gain.
WORD_WEIGHT = 2.0


def _word_bonus(decoded, word_counts, maximum):
    if not word_counts:
        return np.zeros(len(decoded))
    # Normalized log counts give common words more weight without letting a
    # single very frequent word dominate the tetragram model.
    if maximum <= 0:
        return np.zeros(len(decoded))
    result = np.empty(len(decoded))
    cache = {}
    for row, symbols in enumerate(decoded):
        total = 0.0
        letters = 0
        for word in symbols.astype(np.uint8, copy=False).tobytes().split(bytes([SPACE_ID])):
            if not word:
                continue
            value = cache.get(word)
            if value is None:
                text = ''.join(chr(c + 96) for c in word)
                value = np.log1p(word_counts.get(text, 0)) / maximum
                cache[word] = value
            total += len(word) * value
            letters += len(word)
        result[row] = total / max(letters, 1)
    return result


def _char_language_score(padded, char_language):
    """Mean 5-gram score with the same boundary padding as tetragrams."""
    indices = ((((padded[:, :-4] * 27 + padded[:, 1:-3]) * 27
                 + padded[:, 2:-2]) * 27 + padded[:, 3:-1]) * 27
               + padded[:, 4:])
    return char_language[indices].mean(axis=1)


def polish_key(tokens, key, scores, language, evidence_weight, word_counts=None, *, char_language=None):
    """Return a locally improved plaintext-to-ciphertext derangement.

    ``tokens`` uses the chat decoder's 1..26 letter IDs and 27 for spaces;
    ``scores`` is indexed by ciphertext then plaintext. ``word_counts`` is
    an optional mapping from lowercase training-corpus words to counts. Inputs
    are never modified. ``char_language``, if supplied, is a flat 27**5
    table of five-gram log probabilities, weighted by 0.5. Equal-scoring
    moves are not taken.
    """
    tokens = np.asarray(tokens, dtype=np.intp)
    key = np.asarray(key, dtype=np.intp)
    if (tokens.ndim != 1 or not len(tokens) or np.any((tokens < 1) | (tokens > SPACE_ID))
            or not np.any(tokens != SPACE_ID)):
        raise ValueError('tokens must contain at least one letter and valid symbol IDs')
    if key.shape != (26,) or not np.array_equal(np.sort(key), np.arange(26)) or np.any(key == np.arange(26)):
        raise ValueError('key must be a derangement of 0..25')
    scores = np.asarray(scores)
    if scores.shape != (26, 26):
        raise ValueError('scores must have shape (26, 26)')
    if char_language is not None:
        char_language = np.asarray(char_language)
        if char_language.shape != (27**5,):
            raise ValueError('char_language must have shape (27**5,)')
    observed = np.unique(tokens[tokens != SPACE_ID] - 1)
    unseen = np.setdiff1d(np.arange(26), observed)
    lookup = tokens - 1
    cipher = np.argsort(key)
    rows, cols = np.triu_indices(26, 1)
    observed_mask = np.zeros(26, dtype=bool)
    observed_mask[observed] = True
    use = observed_mask[rows] | observed_mask[cols]
    rows, cols = rows[use], cols[use]
    maximum = np.log1p(max(word_counts.values())) if word_counts else 0.0
    letter_mask = tokens != SPACE_ID
    cipher_letters = tokens[letter_mask] - 1
    # Use only swaps that affect text. Repair an invalid unseen assignment if
    # the remaining unseen symbols can be rearranged into a derangement.
    while True:
        variants = np.broadcast_to(cipher, (len(rows), 26)).copy()
        index = np.arange(len(rows))
        variants[index, rows] = cipher[cols]
        variants[index, cols] = cipher[rows]
        invalid = variants == np.arange(26)
        valid = ~invalid[:, observed].any(axis=1)
        # A swap can invalidate an unseen symbol. Reassign unseen symbols
        # with the same deterministic assignment as the original algorithm.
        for index in np.flatnonzero(valid & invalid.any(axis=1)):
            if len(unseen) < 2:
                valid[index] = False
                continue
            cost = np.zeros((len(unseen), len(unseen)))
            cost[unseen[:, None] == variants[index, unseen][None, :]] = np.inf
            try:
                assigned = minimum_cost_assignment(cost)
            except ValueError:
                valid[index] = False
                continue
            variants[index, unseen] = variants[index, unseen][assigned]
        variants = np.concatenate((cipher[None, :], variants[valid]))
        best = 0
        best_score = initial_score = None
        # Bound the temporary decoded/4-gram arrays for long ciphertexts.
        batch_size = max(1, min(len(variants), 262144 // max(len(tokens), 1)))
        for start in range(0, len(variants), batch_size):
            batch = variants[start:start + batch_size]
            maps = np.concatenate((batch, np.full((len(batch), 1), 26)), axis=1)
            decoded = maps[:, lookup]
            padded = np.pad(decoded, ((0, 0), (3, 1)), constant_values=26)
            indices = (((padded[:, :-3] * 27 + padded[:, 1:-2]) * 27
                        + padded[:, 2:-1]) * 27 + padded[:, 3:])
            objective = language[indices].mean(axis=1)
            if char_language is not None:
                objective = objective + 0.5 * _char_language_score(padded, char_language)
            if evidence_weight:
                objective = objective + evidence_weight * scores[cipher_letters[None, :], decoded[:, letter_mask]].mean(axis=1)
            if word_counts:
                objective = objective + WORD_WEIGHT * _word_bonus(decoded + 1, word_counts, maximum)
            local = int(np.argmax(objective))
            if start == 0:
                initial_score = objective[0]
            if best_score is None or objective[local] > best_score:
                best, best_score = start + local, objective[local]
        if best == 0 or best_score <= initial_score + 1e-10:
            break
        cipher = variants[best]
    return np.argsort(cipher)
