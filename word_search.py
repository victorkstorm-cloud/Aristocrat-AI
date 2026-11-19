"""Bounded word-pattern search for coordinated substitution-key proposals.

The lexicon must come from training text only. Keys use the same plaintext-to-
ciphertext convention as aristocrat_chat. This module has no runtime model files.
"""

from collections import defaultdict
import math
import numpy as np

import aristocrat_chat as chat


def _pattern(word):
    positions = {}
    return tuple(positions.setdefault(letter, len(positions)) for letter in word)


class WordSearch:
    """Reusable index of words for a bounded, deterministic key search."""

    def __init__(self, counts, *, max_words_per_pattern=2000):
        self.counts = {word: int(count) for word, count in counts.items()
                       if word.isascii() and word.isalpha() and word.islower() and count > 0}
        groups = defaultdict(list)
        for word, count in self.counts.items():
            if 3 <= len(word) <= 16:
                groups[_pattern(word)].append((word, count))
        self.groups = {pattern: [word for word, _ in sorted(items, key=lambda x: (-x[1], x[0]))[:max_words_per_pattern]]
                       for pattern, items in groups.items()}
        # Rows follow descending frequency and alphabetical tie order. Stable
        # sorting by distance below preserves the original full ordering.
        self.matrices = {pattern: np.frombuffer(''.join(words).encode('ascii'), dtype=np.uint8).reshape(len(words), -1)
                         for pattern, words in self.groups.items()}

    @staticmethod
    def _words(tokens):
        parts = []
        start = 0
        for end in np.flatnonzero(tokens == chat.SPACE_ID):
            if end > start:
                parts.append(tuple(int(x) - 1 for x in tokens[start:end]))
            start = end + 1
        if start < len(tokens):
            parts.append(tuple(int(x) - 1 for x in tokens[start:]))
        return parts

    def word_score(self, words, cipher_to_plain):
        """Average frequency evidence per word, with a penalty for unknown words."""
        total = 0.0
        for word in words:
            text = ''.join(chr(97 + cipher_to_plain[x]) for x in word)
            total += math.log1p(self.counts.get(text, 0)) if text in self.counts else -3.0
        return total / max(len(words), 1)

    @staticmethod
    def _force(mapping, cipher_word, plain_word):
        # Swaps maintain a permutation while enforcing every letter of a word.
        result = mapping.copy()
        fixed = set()
        for cipher, char in zip(cipher_word, plain_word):
            plain = ord(char) - 97
            if cipher in fixed:
                if result[cipher] != plain:
                    return None
                continue
            owner = int(np.flatnonzero(result == plain)[0])
            if owner in fixed and owner != cipher:
                return None
            result[cipher], result[owner] = result[owner], result[cipher]
            fixed.add(cipher)
        # Repair forbidden fixed points without disturbing forced assignments.
        for c in range(26):
            if result[c] != c:
                continue
            if c in fixed:
                return None
            for other in range(26):
                if (other != c and other not in fixed and result[other] != c
                        and result[c] != other):
                    result[c], result[other] = result[other], result[c]
                    break
            else:
                return None
        return result

    def search(self, tokens, key, score_key, *, beam_width=4, rounds=3,
               max_word_changes=3, candidates_per_word=12, word_weight=0.10):
        """Return ranked (combined score, key) proposals, including the input key.

        score_key(key) is the caller's language/evidence score. word_weight is
        an additive weight on average log word frequency. This search is bounded
        by rounds, beam_width, and candidates_per_word, and never changes inputs.
        """
        tokens = np.asarray(tokens)
        words = self._words(tokens)
        unique = list(dict.fromkeys(w for w in words if 3 <= len(w) <= 16))
        observed = np.unique(tokens[tokens != chat.SPACE_ID] - 1)
        def value(mapping):
            candidate = np.argsort(mapping)
            return float(score_key(candidate)) + word_weight * self.word_score(words, mapping)
        initial = np.argsort(np.asarray(key)).copy()
        beam = [(value(initial), initial)]
        seen = {initial[observed].tobytes()}
        all_results = list(beam)
        matches_cache = {}
        for _ in range(rounds):
            proposals = []
            for _, mapping in beam:
                for word in unique:
                    current = ''.join(chr(97 + mapping[c]) for c in word)
                    if current in self.counts:
                        continue
                    pattern = _pattern(word)
                    cache_key = (pattern, current)
                    matches = matches_cache.get(cache_key)
                    if matches is None:
                        matrix = self.matrices.get(pattern)
                        if matrix is None:
                            matches = []
                        else:
                            distances = np.count_nonzero(matrix != np.frombuffer(current.encode('ascii'), dtype=np.uint8), axis=1)
                            eligible = np.flatnonzero((distances >= 1) & (distances <= max_word_changes))
                            order = np.argsort(distances[eligible], kind='stable')[:candidates_per_word]
                            matches = [self.groups[pattern][index] for index in eligible[order]]
                        matches_cache[cache_key] = matches
                    for target in matches:
                        candidate = self._force(mapping, word, target)
                        if candidate is None:
                            continue
                        identity = candidate[observed].tobytes()
                        if identity in seen:
                            continue
                        seen.add(identity)
                        proposals.append((value(candidate), candidate))
            if not proposals:
                break
            proposals.sort(key=lambda item: item[0], reverse=True)
            beam = proposals[:beam_width]
            all_results.extend(beam)
        all_results.sort(key=lambda item: item[0], reverse=True)
        return [(score, np.argsort(mapping)) for score, mapping in all_results]
