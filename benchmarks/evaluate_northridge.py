"""Evaluate the deployed solver on two 2025–26 Northridge tests.

No answers are sent to the solver. Use --prepare-only to verify the cohort
without loading a model or performing inference.
"""
import argparse
import hashlib
import json
import re
import string
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'B': ('https://raw.githubusercontent.com/toebes/ciphers/master/samples/2026_B_Invitational_Northridge.json',
          '9c682075c64819c2f5bbe0f7438cac3b39a0678d6468947ff39f9614ba6d85ff'),
    'C': ('https://raw.githubusercontent.com/toebes/ciphers/master/samples/2026_C_Invitational_Northridge.json',
          '9b3310d1d623d34adc896684e60ea251a82d8e379217b42fbc2ef26148840d7b'),
}
ORGANIZER = 'https://toebes.com/codebusters/samples.html'
ALPHABET = string.ascii_uppercase


def prepare(input_dir=None):
    """Load and verify official tests, reconstruct ciphertext, and deduplicate."""
    cases, seen, exclusions = [], set(), []
    for division, (url, expected_hash) in SOURCES.items():
        if input_dir is None:
            with urlopen(Request(url, headers={'User-Agent': 'aristocrat-benchmark/1'}), timeout=60) as response:
                raw = response.read()
        else:
            raw = (input_dir / f'{division}.json').read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_hash:
            raise ValueError(f'{division} source hash mismatch: expected {expected_hash}, got {actual}')
        test = json.loads(raw)
        metadata = test['TEST.0']
        ids = [metadata['timed'], *metadata['questions']]
        for position, number in enumerate(ids):
            item = test[f'CIPHER.{number}']
            if item.get('cipherType') != 'aristocrat':
                continue
            identifier = f'{division}-{number}'
            if item.get('curlang') != 'en':
                exclusions.append({'id': identifier, 'reason': 'non-English'})
                continue
            if item.get('operation') != 'decode':
                exclusions.append({'id': identifier, 'reason': 'not plaintext recovery'})
                continue
            if item.get('misspelled'):
                raise ValueError(f'{identifier}: misspelled puzzle needs separate verification')
            source, dest = item['alphabetSource'], item['alphabetDest']
            replacement = item['replacement']
            if (sorted(source) != list(ALPHABET) or sorted(dest) != list(ALPHABET)
                    or any(replacement.get(a) != b for a, b in zip(source, dest))
                    or any(a == b for a, b in zip(source, dest))):
                raise ValueError(f'{identifier}: inconsistent saved replacement mapping')
            # The organizer's makeReplacement encodes via getReverseReplacement;
            # its saved mapping is ciphertext -> plaintext, not the reverse.
            plain = item['cipherString'].upper()
            encoder = dict(zip(dest, source))
            ciphertext = ''.join(encoder.get(c, c) for c in plain)
            if ''.join(replacement.get(c, c) for c in ciphertext) != plain:
                raise ValueError(f'{identifier}: reconstruction round trip failed')
            identity = ' '.join(re.findall('[A-Z]+', plain.replace("'", '').replace('’', '')))
            if identity in seen:
                exclusions.append({'id': identifier, 'reason': 'duplicate plaintext'})
                continue
            seen.add(identity)
            cases.append({'id': identifier, 'division': division,
                          'question': 'timed' if position == 0 else position,
                          'ciphertext': ciphertext, 'plaintext': plain})
    return cases, exclusions


def score(cases, model_dir=None):
    # Resolve from this file, not the caller's working directory. Imports are
    # deferred so --prepare-only does not load NumPy, a model, or the solver.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np
    import aristocrat_chat as chat
    model = chat.load_model(model_dir or chat.default_model_dir())
    search = chat.get_language_search() if model[5] > 0 else None
    rows = []
    for case in cases:
        cipher, plain = case['ciphertext'], case['plaintext']
        tokens = np.asarray(chat.tokenize_ciphertext(cipher), dtype=int)
        _, neural_key, final_key = chat.solve_tokens(tokens, *model, search)
        neural = chat.apply_key(cipher, neural_key)
        final = chat.apply_key(cipher, final_key)
        # Compare exactly the CLI's formatted output, ignoring letter case only.
        rows.append({'id': case['id'], 'baseline_correct': neural.upper() == plain,
                     'enhanced_correct': final.upper() == plain})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, help='directory holding pinned B.json and C.json')
    parser.add_argument('--output', type=Path, help='write JSON report to this file')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='verify cohort without inference')
    args = parser.parse_args()
    cases, exclusions = prepare(args.input_dir)
    report = {'organizer_index': ORGANIZER,
              'sources': {division: {'url': url, 'sha256': digest}
                          for division, (url, digest) in SOURCES.items()},
              'eligible_count': len(cases), 'question_ids': [case['id'] for case in cases],
              'exclusions': exclusions}
    if not args.prepare_only:
        rows = score(cases, args.model_dir)
        report['results'] = rows
        report['aggregate'] = {
            'baseline_correct': sum(row['baseline_correct'] for row in rows),
            'enhanced_correct': sum(row['enhanced_correct'] for row in rows),
            'fixed': sum(not row['baseline_correct'] and row['enhanced_correct'] for row in rows),
            'broken': sum(row['baseline_correct'] and not row['enhanced_correct'] for row in rows),
        }
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
