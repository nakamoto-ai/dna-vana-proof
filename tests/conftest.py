
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_requester():
    """Fixture to mock requests."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_hasher():
    """Fixture to mock hasher (e.g., hashlib)."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_proof():
    mock = MagicMock()
    return mock