import itertools
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import numpy as np

import aristocrat_chat as chat
import train_5090 as train
import training_data as data


class DataTests(unittest.TestCase):
    def test_words_normalization(self):
        self.assertEqual(data.words("Café can't—DON’T 123!"), ["cafe", "cant", "dont"])

    def test_puzzles_and_supervision(self):
        # Corpus words are alphabetic, as guaranteed by words().
        documents = [["alpha", "beta", "gamma", "delta", "epsilon"] * 80]
        puzzles = data.make_puzzles(documents, 4, 17)
        again = data.make_puzzles(documents, 4, 17)
        for (cipher, plain), (cipher2, plain2) in zip(puzzles, again):
            np.testing.assert_array_equal(cipher, cipher2)
            np.testing.assert_array_equal(plain, plain2)
            self.assertTrue(np.all((cipher == chat.SPACE_ID) == (plain == chat.SPACE_ID)))
            pairs = {(int(p), int(c)) for p, c in zip(plain, cipher) if p != chat.SPACE_ID}
            self.assertEqual(len(pairs), len({p for p, _ in pairs}))
            self.assertEqual(len(pairs), len({c for _, c in pairs}))
            self.assertTrue(all(p != c for p, c in pairs))
        features, labels = data.supervised_symbols(puzzles)
        expected = sum(len(set(c) - {chat.SPACE_ID}) for c, _ in puzzles)
        self.assertEqual(features.shape, (expected, 197))
        offset = 0
        for cipher, plain in puzzles:
            observed = sorted(set(cipher) - {chat.SPACE_ID})
            for i, symbol in enumerate(observed):
                self.assertEqual(labels[offset + i], plain[np.flatnonzero(cipher == symbol)[0]] - 1)
            offset += len(observed)

    def test_corrupt_cached_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "gutenberg.zip").write_bytes(b"corrupt")
            with patch.object(data, "urlopen", side_effect=AssertionError("network used")):
                with self.assertRaisesRegex(ValueError, "SHA-256 verification"):
                    data.read_book_archive(cache)

    def test_small_corpus_rejected_and_signature_changes(self):
        with self.assertRaises(ValueError):
            data.make_puzzles([["small"]], 1, 1)
        self.assertNotEqual(data.data_signature({"main": [["one"]]}),
                            data.data_signature({"main": [["two"]]}))


class MathTests(unittest.TestCase):
    def test_assignment_matches_brute_force(self):
        matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        assignment = chat.minimum_cost_assignment(matrix)
        actual = matrix[np.arange(3), assignment].sum()
        self.assertEqual(actual, min(sum(matrix[i, p[i]] for i in range(3))
                                     for p in itertools.permutations(range(3))))
        with self.assertRaises(ValueError):
            chat.minimum_cost_assignment(np.array([[np.inf, np.inf], [0, 1]]))

    def test_features_and_permutation_equivariance(self):
        tokens = np.array(chat.tokenize_ciphertext("ab ba cab")) - 1
        features, counts = chat.extract_features(tokens)
        self.assertEqual(features.shape, (26, 197))
        self.assertEqual(counts.sum(), 7)
        perm = np.roll(np.arange(26), 3)
        swapped = np.where(tokens == 26, 26, perm[np.minimum(tokens, 25)])
        changed, changed_counts = chat.extract_features(swapped)
        np.testing.assert_allclose(features, changed[perm])
        np.testing.assert_array_equal(counts, changed_counts[perm])

    def test_key_and_context(self):
        key = np.roll(np.arange(26), -1)
        self.assertEqual(chat.apply_key("Bcd!", key), "Abc!")
        tokens = np.array([2, 3, 27, 4])
        np.testing.assert_array_equal(chat.guess_tokens(tokens, key), [1, 2, 27, 3])
        windows = chat.context_windows(tokens)
        self.assertEqual(windows.shape, (4, 13))
        np.testing.assert_array_equal(windows[:, 6], tokens)
        scores = np.eye(26) * 100
        predicted = chat.predict_key(scores)
        self.assertTrue(np.all(predicted != np.arange(26)))
        self.assertEqual(len(set(predicted)), 26)


class ConvergenceTests(unittest.TestCase):
    def test_converged_generator_preserves_requested_count(self):
        key = np.roll(np.arange(26), -1)
        tokens = np.array([1, 2, 27, 1])
        with patch.object(chat, "correction_scores", return_value=np.zeros((26, 26))) as correction, \
             patch.object(chat, "predict_key", return_value=key) as predict:
            candidates = list(chat.refinement_candidates(tokens, key, np.zeros((26, 26)), None, 1, 5))
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(np.array_equal(candidate, key) for candidate in candidates))
        self.assertEqual(correction.call_count, 1)
        self.assertEqual(predict.call_count, 2)

    def test_cli_reports_requested_passes_after_convergence(self):
        key = np.roll(np.arange(26), -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "aristocrat_weights.npz", **{f"arr_{i}": np.zeros(1) for i in range(6)})
            values = {f"weight_{i}": np.zeros(1) for i in range(7)}
            values.update(strength=1., language=np.zeros(1), evidence_weight=0., max_passes=2)
            np.savez(root / "aristocrat_refiner.npz", **values)
            output = io.StringIO()
            with patch.object(sys, "argv", ["aristocrat_chat.py", "--model-dir", str(root), "--passes", "5"]), \
                 patch.object(chat, "read_ciphertext", side_effect=["ab", "/quit"]), \
                 patch.object(chat, "first_guess", return_value=(key, np.zeros((26, 26)))), \
                 patch.object(chat, "candidate_scores", return_value=(0., 0.)), \
                 patch.object(chat, "correction_scores", return_value=np.zeros((26, 26))) as correction, \
                 patch.object(chat, "predict_key", return_value=key), redirect_stdout(output):
                chat.main()
        self.assertIn("Refinement passes: 5", output.getvalue())
        self.assertEqual(correction.call_count, 1)


class FormatTests(unittest.TestCase):
    def weights(self):
        rng = np.random.default_rng(1)
        return tuple(rng.normal(size=shape).astype(np.float32) for shape in
                     [(3, 4), (4,), (4, 5), (5,), (5, 2), (2,)])

    def test_widening_preserves_logits(self):
        weights = self.weights()
        inputs = np.array([[1., -2., .5], [0., 0., 0.]], dtype=np.float32)
        widened = train.widened_layers(weights, (9, 11))
        np.testing.assert_allclose(chat.predict_logits(inputs, weights),
                                   chat.predict_logits(inputs, widened), rtol=1e-5, atol=1e-6)

    def test_quantized_archive_round_trip(self):
        weights = self.weights()
        archive, rounded = train.pack_main(weights)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            train.save_archive(path, archive)
            loaded = chat.load_main_weights(path)
            self.assertFalse(path.with_suffix(".tmp.npz").exists())
        for actual, expected in zip(loaded, rounded):
            np.testing.assert_array_equal(actual, expected)
        for original, actual in zip(weights[::2], loaded[::2]):
            self.assertLessEqual(float(np.max(np.abs(original - actual))),
                                 float(np.max(np.abs(original)) / 127 + 1e-6))


if __name__ == "__main__":
    unittest.main()
