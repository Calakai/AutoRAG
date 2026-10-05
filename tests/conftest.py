"""Shared fixtures."""

import pytest

from tests.helpers import HashingEmbedder


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
