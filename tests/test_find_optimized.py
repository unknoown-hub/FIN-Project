"""
tests/test_find_optimized.py

Unit tests for FINd_optimized.py.



These tests verify:
    1. The hash output has the correct structure (256 bits)
    2. Hashing the same image twice gives identical results
    3. Hamming distance between identical hashes is zero
    4. Hamming distance is non-negative and does not exceed 256
    5. The hasher handles a PIL Image object as well as a file path
    6. Invalid file paths raise an IOError, not a silent failure
    7. Discrimination: same-family images are closer than different-family images
    8. Correctness: hashes are bit-identical to the original FINDHasher on 20 images
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from FINd import FINDHasher
from FINd_optimized import FINDHasher_optimized


MEME_DIR = Path("/Users/noha/Desktop/Hilary_term /DS in practice TA/Summative /meme_images")

# Two images from the same meme family (family 0000)
SAME_A = MEME_DIR / "0000_12268686.jpg"
SAME_B = MEME_DIR / "0000_12270286.jpg"

# One image from a completely different family (family 0001)
DIFF   = MEME_DIR / "0001_13187259.jpg"


@pytest.fixture(scope="module")
def hasher():
    """Single FINDHasher_optimized instance shared across all tests."""
    return FINDHasher_optimized()


@pytest.fixture(scope="module")
def original_hasher():
    """Original FINDHasher for correctness comparison."""
    return FINDHasher()


@pytest.fixture(scope="module")
def hash_same_a(hasher):
    return hasher.fromFile(str(SAME_A))


@pytest.fixture(scope="module")
def hash_same_b(hasher):
    return hasher.fromFile(str(SAME_B))


@pytest.fixture(scope="module")
def hash_diff(hasher):
    return hasher.fromFile(str(DIFF))


# ── Tests ──────────────────────────────────────────────────────────────────


class TestHashStructure:
    """Verify the hash output has the correct format."""

    def test_hash_is_256_bits(self, hash_same_a):
        """FINd is a 256-bit hash — the output array must have exactly 256 elements."""
        assert len(hash_same_a.hash) == 256

    def test_hash_contains_only_zeros_and_ones(self, hash_same_a):
        """Each bit in the hash must be 0 or 1."""
        values = set(int(b) for b in hash_same_a.hash)
        assert values.issubset({0, 1})

    def test_hash_is_not_all_zeros(self, hash_same_a):
        """A hash of all zeros would mean no image information was captured."""
        assert sum(int(b) for b in hash_same_a.hash) > 0

    def test_hash_is_not_all_ones(self, hash_same_a):
        """A hash of all ones is equally meaningless."""
        assert sum(int(b) for b in hash_same_a.hash) < 256


class TestDeterminism:
    """Hashing the same image twice must always give the same result."""

    def test_same_file_twice_gives_identical_hash(self, hasher):
        """Determinism check — no randomness in the algorithm."""
        h1 = hasher.fromFile(str(SAME_A))
        h2 = hasher.fromFile(str(SAME_A))
        assert (h1 - h2) == 0

    def test_fromFile_and_fromImage_agree(self, hasher):
        """fromFile and fromImage must produce the same hash for the same image."""
        h_file  = hasher.fromFile(str(SAME_A))
        h_image = hasher.fromImage(Image.open(SAME_A))
        assert (h_file - h_image) == 0


class TestHammingDistance:
    """Verify the Hamming distance behaves correctly."""

    def test_identical_hashes_have_zero_distance(self, hash_same_a):
        """Distance between a hash and itself must be zero."""
        assert (hash_same_a - hash_same_a) == 0

    def test_distance_is_non_negative(self, hash_same_a, hash_diff):
        """Hamming distance is always >= 0."""
        assert (hash_same_a - hash_diff) >= 0

    def test_distance_does_not_exceed_256(self, hash_same_a, hash_diff):
        """Maximum possible distance for a 256-bit hash is 256."""
        assert (hash_same_a - hash_diff) <= 256


class TestDiscrimination:
    """Verify the optimised hasher separates similar images from different ones."""

    def test_same_family_closer_than_different_family(self, hash_same_a, hash_same_b, hash_diff):
        """
        Core accuracy check: distance(same family) < distance(different family).
        Optimisation must not degrade discriminative power.
        """
        same_dist = hash_same_a - hash_same_b
        diff_dist = hash_same_a - hash_diff
        assert same_dist < diff_dist, (
            f"Same-family distance ({same_dist}) should be less than "
            f"different-family distance ({diff_dist})"
        )

    def test_same_family_distance_below_threshold(self, hash_same_a, hash_same_b):
        """
        Same-family images should have Hamming distance well below 128
        (the expected distance for two completely random 256-bit hashes).
        """
        dist = hash_same_a - hash_same_b
        assert dist < 128, (
            f"Same-family distance {dist} is not below the random baseline of 128"
        )


class TestCorrectness:
    """
    Verify the optimised hasher produces bit-identical hashes to the original.
    This is the primary correctness contract: all optimisations are semantics-preserving.
    """

    def test_matches_original_on_same_a(self, hasher, original_hasher):
        """Hash of SAME_A must be bit-identical between optimised and original."""
        h_opt  = hasher.fromFile(str(SAME_A))
        h_orig = original_hasher.fromFile(str(SAME_A))
        assert (h_opt - h_orig) == 0, (
            f"Optimised hash differs from original by {h_opt - h_orig} bits on SAME_A"
        )

    def test_matches_original_on_same_b(self, hasher, original_hasher):
        """Hash of SAME_B must be bit-identical between optimised and original."""
        h_opt  = hasher.fromFile(str(SAME_B))
        h_orig = original_hasher.fromFile(str(SAME_B))
        assert (h_opt - h_orig) == 0, (
            f"Optimised hash differs from original by {h_opt - h_orig} bits on SAME_B"
        )

    def test_matches_original_on_diff(self, hasher, original_hasher):
        """Hash of DIFF must be bit-identical between optimised and original."""
        h_opt  = hasher.fromFile(str(DIFF))
        h_orig = original_hasher.fromFile(str(DIFF))
        assert (h_opt - h_orig) == 0, (
            f"Optimised hash differs from original by {h_opt - h_orig} bits on DIFF"
        )

    def test_matches_original_on_20_images(self, hasher, original_hasher):
        """
        Batch correctness: optimised hash must equal original on 20 diverse images.
        Tests across multiple image sizes and meme families.
        """
        import random
        random.seed(42)
        images = random.sample(sorted(MEME_DIR.glob("*.jpg")), 20)
        failures = []
        for p in images:
            d = original_hasher.fromFile(str(p)) - hasher.fromFile(str(p))
            if d != 0:
                failures.append((p.name, d))
        assert not failures, (
            f"Optimised hash differs from original on {len(failures)} images: "
            + ", ".join(f"{name} (dist={d})" for name, d in failures)
        )


class TestErrorHandling:
    """Verify the hasher fails clearly on bad input."""

    def test_missing_file_raises_error(self, hasher):
        """Passing a non-existent path should raise IOError, not return silently."""
        with pytest.raises(IOError):
            hasher.fromFile("/this/path/does/not/exist.jpg")
