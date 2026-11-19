"""Check the shipped model archives and exercise both CLI model choices."""
import hashlib
import string
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "models" / "rtx5090"
PLAINTEXT = "It was the best of times, it was the worst of times."
DIGESTS = {
    ROOT / "models/english_fivegrams.npz": "55453770a070b9fcaebd365d35f8df061f5e36a8b47b91d543aaf740a8b48abe",
    ROOT / "models/english_words.tsv.gz": "90ebfc5b5999c6f62b7ec45e8d4f4129c49cc4c751ee11119b548ade70e8b500",
    ROOT / "aristocrat_weights.npz": "be798fedb2eff294f35830afffaf6aae27e6f35a40646db668b24d193e05b710",
    ROOT / "aristocrat_refiner.npz": "0fda17aac9038daa690d547c4c936bb8a5f8314008d73f1ff9241cd40a7f3e22",
    TRAINED / "aristocrat_weights.npz": "922c181daaf95d63b6508fdc5b5b3530e19deecb121deafe4880f4b12f2760da",
    TRAINED / "aristocrat_refiner.npz": "b1480a5c037b4deeab523bf45cfa994dbac80076413182ebefbd76ad8502f517",
}


def exercise_cli(*options):
    lower = string.ascii_lowercase
    shifted = lower[7:] + lower[:7]
    ciphertext = PLAINTEXT.translate(str.maketrans(lower + lower.upper(), shifted + shifted.upper()))
    result = subprocess.run(
        [sys.executable, str(ROOT / "aristocrat_chat.py"), *options],
        input=ciphertext + "\n/quit\n", text=True, capture_output=True, check=True, timeout=90,
    )
    decoded = result.stdout.partition("\nRefined\n")[2].splitlines()
    if not decoded or len(decoded[0]) != len(PLAINTEXT) or "Refinement passes: 3" not in result.stdout:
        raise RuntimeError(f"Incomplete decode with options {options}: {result.stdout}")
    return decoded[0]


if __name__ == "__main__":
    for path, expected in DIGESTS.items():
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise SystemExit(f"Model checksum mismatch: {path.relative_to(ROOT)}")
    answer = exercise_cli()
    if answer != PLAINTEXT:
        raise SystemExit(f"The default decoder missed the release fixture: {answer}")
    if answer != exercise_cli("--model-dir", str(TRAINED)):
        raise SystemExit("Default CLI output differs from the shipped RTX 5090 pair.")
    exercise_cli("--model-dir", str(ROOT))
    print("Verified model and lexicon checksums, an exact solve, and both model pairs.")
