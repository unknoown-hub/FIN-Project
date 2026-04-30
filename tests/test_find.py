"""
tests/test_find.py

Unit tests for FINd.py — the original FINd image hashing algorithm.



These tests verify:
    1. The hash output has the correct structure (256 bits)
    2. Hashing the same image twice gives identical results
    3. Hamming distance between identical hashes is zero
    4. Hamming distance is non-negative and does not exceed 256
    5. The hasher handles a PIL Image object as well as a file path
    6. Invalid file paths raise an IOError, not a silent failure
"""

import sys
import io
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from FINd import FINDHasher



MEME_DIR = Path("/Users/noha/Desktop/Hilary_term /DS in practice TA/Summative /meme_images")

# Two images from the same meme family (family 0000)
SAME_A = MEME_DIR / "0000_12268686.jpg"
SAME_B = MEME_DIR / "0000_12270286.jpg"

# One image from a completely different family (family 0001)
DIFF   = MEME_DIR / "0001_13187259.jpg"


@pytest.fixture(scope="module")
def hasher():
    """Single FINDHasher instance shared across all tests in this module."""
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



# Tests


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


class TestErrorHandling:
    """Verify the hasher fails clearly on bad input."""

    def test_missing_file_raises_error(self, hasher):
        """Passing a non-existent path should raise IOError, not return silently."""
        with pytest.raises(IOError):
            hasher.fromFile("/this/path/does/not/exist.jpg")
