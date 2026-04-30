# FINd Image Hashing — SDS in Practice 2026

This repository contains the FINd  image hashing algorithm, an
optimised implementation, a comparative evaluation against dHash, and a
deployable REST API.

> This code and data is intended for the OII SDS in Practice course only.
> The code and data should not be used for any other purposes.


---

## Dataset

The meme generator dataset from the Library of Congress is used for
evaluation. It consists of 55,972 JPEG images belonging to 1,035 meme
families, located in `/data/meme_images`.

Filenames follow the convention `XXXX_imageid.jpg` where the first four
digits identify the meme family. Images within the same family produce
lower Hamming distances than images across families.


## Project Structure

```
FIN project/
├── FINd.py                     # Original algorithm —  NOT MODIFY
├── FINd_optimized.py           # Optimised hasher (6 optimisations, 10.49× speedup)
├── matrix.py                   # Matrix utilities — NOT MODIFY
│
├── optimisations/              # Individual optimisation variants (v1–v6)
│   ├── v1_luminance.py         # np.asarray luma conversion
│   ├── v2_boxfilter.py         # Integral image box filter
│   ├── v3_median.py            # np.median
│   ├── v4_hash.py              # Vectorised threshold
│   ├── v5_decimate.py          # np.ix_ decimation
│   └── v6_dct.py               # Matrix multiply DCT (@ operator)
│
├── notebooks/
│   ├── 01_evaluation...ipynb   # Profiling and evaluation of original FINd
│   ├── 02_optimization.ipynb   # Benchmarking all optimisation variants
│   ├── 03_optimized...ipynb    # Evaluation of FINDHasher_optimized
│   └── 04_comparison.ipynb     # FINDHasher_optimized vs dHash comparison
│
├── find_api/                   # REST API package
│   ├── __init__.py
│   └── app.py                  # FastAPI application — POST /compare on port 8945
│
├── tests/
│   ├── test_find.py            # Unit tests — original FINDHasher (12 tests)
│   ├── test_find_optimized.py  # Unit tests — FINDHasher_optimized (16 tests)
│   └── test_api.py             # Integration tests — REST API (14 tests)
│
├── Dockerfile                  # Multi-stage container build
├── .dockerignore               # Files excluded from Docker image
├── requirements.txt            # Runtime dependencies
├── pyproject.toml              # Package configuration
└── output/                     # Generated figures and results
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the original FINd algorithm

```bash
python FINd.py meme_images/0297_21195384.jpg
```

### 3. Use the optimised hasher in Python

```python
from FINd_optimized import FINDHasher_optimized

hasher = FINDHasher_optimized()
h1 = hasher.fromFile("image1.jpg")
h2 = hasher.fromFile("image2.jpg")
print(h1 - h2)  # Hamming distance, 0–256
```

### 4. Run all tests

```bash
pytest tests/ -v
```

### 5. Start the API server

```bash
uvicorn find_api.app:app --host 0.0.0.0 --port 8945
```

Interactive docs available at: `http://localhost:8945/docs`

### 6. Deploy with Docker

```bash
docker build -t find-api .
docker run -p 8945:8945 find-api
```

---

## API Reference

**Endpoint:** `POST /compare`
**Port:** 8945
**Content-Type:** `multipart/form-data`

| Field  | Type | Description      |
|--------|------|------------------|
| image1 | file | First image file |
| image2 | file | Second image file |

**Response (HTTP 200):**

```json
{
  "image1_hash": "a3f1...",
  "image2_hash": "b9c2...",
  "distance": 42
}
```

| Status | Condition                          |
|--------|------------------------------------|
| 200    | Success                            |
| 400    | File is not a valid image          |
| 422    | Missing image1 or image2 field     |

---

## Known Limitations

- `app.py` uses a `sys.path` injection to import `FINd_optimized` from
  the project root. If the directory structure changes, replace with a
  proper package installation.
- The optimised hasher uses ~2.2× more memory than the original due to
  the integral image allocated during the box filter optimisation.
  Converting intermediate arrays from `float64` to `float32` would halve
  this overhead.
