"""
tests/test_api.py

Integration tests for the FINd REST API (/compare endpoint).



import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from find_api.app import app

client = TestClient(app)

MEME_DIR = Path(".../DS in practice TA/Summative /meme_images") # Chang the path upon replication
IMG_A = MEME_DIR / "0000_12268686.jpg"
IMG_B = MEME_DIR / "0000_12270286.jpg"
IMG_C = MEME_DIR / "0001_13187259.jpg"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post(path_a, path_b):
    with open(path_a, "rb") as f1, open(path_b, "rb") as f2:
        return client.post("/compare", files={"image1": f1, "image2": f2})


# ── Response structure ────────────────────────────────────────────────────────

class TestResponseStructure:

    def test_status_200(self):
        assert _post(IMG_A, IMG_B).status_code == 200

    def test_has_image1_hash(self):
        assert "image1_hash" in _post(IMG_A, IMG_B).json()

    def test_has_image2_hash(self):
        assert "image2_hash" in _post(IMG_A, IMG_B).json()

    def test_has_distance(self):
        assert "distance" in _post(IMG_A, IMG_B).json()

    def test_hash_is_64_char_hex(self):
        data = _post(IMG_A, IMG_B).json()
        assert len(data["image1_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in data["image1_hash"])

    def test_distance_is_integer(self):
        assert isinstance(_post(IMG_A, IMG_B).json()["distance"], int)

    def test_distance_in_valid_range(self):
        d = _post(IMG_A, IMG_B).json()["distance"]
        assert 0 <= d <= 256


# ── Correctness ───────────────────────────────────────────────────────────────

class TestCorrectness:

    def test_same_image_distance_zero(self):
        data = _post(IMG_A, IMG_A).json()
        assert data["distance"] == 0

    def test_same_image_identical_hashes(self):
        data = _post(IMG_A, IMG_A).json()
        assert data["image1_hash"] == data["image2_hash"]

    def test_same_family_closer_than_different_family(self):
        same_dist = _post(IMG_A, IMG_B).json()["distance"]
        diff_dist = _post(IMG_A, IMG_C).json()["distance"]
        assert same_dist < diff_dist

    def test_distance_is_symmetric(self):
        d_ab = _post(IMG_A, IMG_B).json()["distance"]
        d_ba = _post(IMG_B, IMG_A).json()["distance"]
        assert d_ab == d_ba


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_missing_image1_returns_422(self):
        with open(IMG_B, "rb") as f:
            r = client.post("/compare", files={"image2": f})
        assert r.status_code == 422

    def test_missing_image2_returns_422(self):
        with open(IMG_A, "rb") as f:
            r = client.post("/compare", files={"image1": f})
        assert r.status_code == 422

    def test_invalid_file_returns_400(self):
        fake = b"this is not an image"
        r = client.post("/compare", files={
            "image1": ("fake.jpg", fake, "image/jpeg"),
            "image2": ("fake2.jpg", fake, "image/jpeg"),
        })
        assert r.status_code == 400
