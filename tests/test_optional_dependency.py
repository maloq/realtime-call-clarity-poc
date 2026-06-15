import pytest

from callclarity.registry import create_method
from callclarity.types import MethodUnavailable


def test_missing_optional_dependency_gives_clean_error():
    with pytest.raises(MethodUnavailable, match="checkpoint"):
        create_method("denoise", "tiny_mask_gru", {"checkpoint_path": None})
