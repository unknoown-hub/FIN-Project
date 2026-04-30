"""
FINd REST API — FastAPI application.

Endpoint
--------
POST /compare
    Accepts two images (multipart/form-data) and returns the Hamming distance
    between their FINd hashes.
"""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# Allow importing FINd_optimized from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from FINd_optimized import FINDHasher_optimized

app = FastAPI(
    title="FINd Image Hashing API",
    description="Compare two images using the FINd perceptual hashing algorithm.",
    version="1.0.0",
)

_hasher = FINDHasher_optimized()


def _load_image(upload: UploadFile) -> Image.Image:
    """Read an UploadFile and return a PIL Image, raising HTTP 400 on failure."""
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=f"'{upload.filename}' is not a valid image file.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/compare")
def compare(
    image1: UploadFile = File(..., description="First image"),
    image2: UploadFile = File(..., description="Second image"),
) -> JSONResponse:
    """
    Hash both images with FINd and return their Hamming distance.

    Returns
    -------
    JSON with keys: image1_hash, image2_hash, distance
    """
    img1 = _load_image(image1)
    img2 = _load_image(image2)

    h1 = _hasher.fromImage(img1)
    h2 = _hasher.fromImage(img2)

    return JSONResponse({
        "image1_hash": str(h1),
        "image2_hash": str(h2),
        "distance": int(h1 - h2),
    })
