import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def tmp_path():
    path = ROOT / ".tmp" / f"test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path
