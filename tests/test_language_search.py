import hashlib
from collections import Counter
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

import aristocrat_chat as chat
import build_lexicon
import char_language
import search_keys
import training_data
from word_search import WordSearch


class LexiconTests(unittest.TestCase):
    def test_training_only_counts_and_reproducible_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'books.zip'
            with ZipFile(archive, 'w') as z:
                for split, books in training_data.BOOKS.items():
                    for book in books:
                        z.writestr('gutenberg/' + book, b'Caf\xe9 cat cat' if split in ('main', 'refiner') else b'heldout')
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.object(training_data, 'CORPUS_SHA256', checksum):
                counts = build_lexicon.build_word_counts(archive)
            n = len(training_data.BOOKS['main']) + len(training_data.BOOKS['refiner'])
            self.assertEqual(counts, {'cafe': n, 'cat': 2 * n})
            first, second = root / 'one.gz', root / 'two.gz'
            build_lexicon.write_word_counts(counts, first)
            build_lexicon.write_word_counts(dict(reversed(list(counts.items()))), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(build_lexicon.load_word_counts(first), counts)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                build_lexicon.build_word_counts(archive)


class CharacterLanguageTests(unittest.TestCase):
    def test_synthetic_counts_scalar_scores_boundary_and_loader(self):
        documents = [["aba", "cab"], ["abc", "aba"]]
        with patch.object(char_language, "training_books", return_value=iter(documents)):
            table = char_language.build_char_language("unused")
        counts = {n: Counter() for n in (3, 4, 5)}
        for document in documents:
            text = "   " + " ".join(document) + " "
            for n in counts:
                counts[n].update(text[i:i+n] for i in range(len(text)-n+1))

        def index(gram):
            result = 0
            for c in gram:
                result = result * 27 + (26 if c == " " else ord(c) - 97)
            return result

        def expected(gram):
            lower = (counts[4][gram[1:]] + 1) / (counts[3][gram[1:4]] + 27)
            return np.log((counts[5][gram] + 10 * lower) / (counts[4][gram[:4]] + 10))

        for gram in ("   ab", "aba c", " cab ", "zzzzz", "abcab"):
            self.assertAlmostEqual(float(table[index(gram)]), expected(gram), places=6)
        text = "aba cab"
        padded = "   " + text + " "
        digits = np.array([[26 if c == " " else ord(c) - 97 for c in padded]])
        scalar = np.mean([expected(padded[i:i+5]) for i in range(len(padded)-4)])
        self.assertAlmostEqual(float(search_keys._char_language_score(digits, table)[0]), scalar, places=6)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.npz"
            char_language.save_char_language(table, path)
            char_language.load_char_language.cache_clear()
            loaded = char_language.load_char_language(path)
            self.assertIs(loaded, char_language.load_char_language(path))
            self.assertFalse(loaded.flags.writeable)
            with self.assertRaises(ValueError):
                loaded[0] = 0
            np.testing.assert_array_equal(loaded, table)
            bad = root / "bad.npz"
            np.savez(bad, log_probabilities=np.zeros(3, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Invalid character language"):
                char_language.load_char_language(bad)
            wrong_type = root / "wrong_type.npz"
            np.savez(wrong_type, log_probabilities=np.zeros(3, dtype=np.float64))
            with self.assertRaises(ValueError):
                char_language.load_char_language(wrong_type)
            char_language.load_char_language.cache_clear()


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.true_key = np.roll(np.arange(26), -1)
        self.plaintext = 'hello hello world'
        self.tokens = np.array([self.true_key[ord(c) - 97] + 1 if c != ' ' else 27 for c in self.plaintext])
        mapping = np.argsort(self.true_key)
        mapping[0], mapping[4] = mapping[4], mapping[0]
        self.start = np.argsort(mapping)

    def test_word_search_improves_planted_text_and_preserves_keys(self):
        search = WordSearch({'hello': 100, 'world': 100, 'INVALID': 3})
        original = self.start.copy()
        proposals = search.search(self.tokens, self.start, lambda key: 0.)
        words = search._words(self.tokens)
        before = search.word_score(words, np.argsort(self.start))
        after = search.word_score(words, np.argsort(proposals[0][1]))
        self.assertGreater(after, before)
        self.assertEqual(chat.apply_key('Ifmmp Ifmmp Xpsme!', proposals[0][1]), 'Hello Hello World!')
        self.assertTrue(any(np.array_equal(k, original) for _, k in proposals))
        for _, key in proposals:
            np.testing.assert_array_equal(np.sort(key), np.arange(26))
            self.assertTrue(np.all(key != np.arange(26)))
        np.testing.assert_array_equal(self.start, original)

    def test_polish_improves_planted_evidence_and_rejects_invalid_keys(self):
        scores = np.zeros((26, 26))
        inverse = np.argsort(self.true_key)
        scores[np.arange(26), inverse] = 1
        language = np.zeros(27 ** 4)
        original = self.start.copy()
        improved = search_keys.polish_key(self.tokens, self.start, scores, language, 1.)
        def evidence(key):
            return chat.candidate_scores(self.tokens, key, scores, language)[1]
        self.assertGreater(evidence(improved), evidence(self.start))
        self.assertTrue(np.all(improved != np.arange(26)))
        np.testing.assert_array_equal(np.sort(improved), np.arange(26))
        np.testing.assert_array_equal(self.start, original)
        with self.assertRaises(ValueError):
            search_keys.polish_key(self.tokens, np.arange(26), scores, language, 1.)

    def test_zero_pass_and_cli_use_baseline(self):
        scores = np.zeros((26, 26))
        with patch.object(chat, 'first_guess', return_value=(self.true_key, scores)), \
             patch.object(chat, 'get_language_search', side_effect=AssertionError('search used')):
            first, final, count = chat.decode('Ifmmp, Xpsme!', None, None, 0, None, 0, 0)
            self.assertEqual((first, final, count), ('Hello, World!', 'Hello, World!', 0))
            output = io.StringIO()
            with patch.object(sys, 'argv', ['aristocrat_chat.py', '--passes', '0']), \
                 patch.object(chat, 'load_model', return_value=(None, None, 0, None, 0, 3)), \
                 patch.object(chat, 'read_ciphertext', side_effect=['Ifmmp, Xpsme!', '/quit']), \
                 patch('sys.stdout', output):
                chat.main()
            self.assertIn('Refined\nHello, World!', output.getvalue())
            self.assertIn('Refinement passes: 0', output.getvalue())


if __name__ == '__main__':
    unittest.main()
