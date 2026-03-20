"""Paths to committed test assets (see testing/fixtures/README.md)."""

from pathlib import Path

_TESTING_DIR = Path(__file__).resolve().parent

FIXTURE_IMAGES_DIR = _TESTING_DIR / "fixtures" / "images"
