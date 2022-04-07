import pytest

import merlin.io
from merlin.datasets.synthetic import generate_data

pytestmark = pytest.mark.datasets


def test_social_data():
    dataset = generate_data("social", 100)

    assert isinstance(dataset, merlin.io.Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 18
