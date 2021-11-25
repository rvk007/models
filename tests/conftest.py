#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
from merlin_standard_lib import Schema

from merlin_models.data import tabular_testing_data


@pytest.fixture
def tabular_data_file() -> str:
    return tabular_testing_data.path


@pytest.fixture
def tabular_schema_file() -> str:
    return tabular_testing_data.schema_path


@pytest.fixture
def tabular_schema() -> Schema:
    return tabular_testing_data.schema.remove_by_name(["session_id", "session_start", "day_idx"])


@pytest.fixture
def yoochoose_schema() -> Schema:
    return tabular_testing_data.schema.remove_by_name(["session_id", "session_start", "day_idx"])


try:
    import tensorflow as tf  # noqa

    from tests.tf.conftest import *  # noqa
except ImportError:
    pass

try:
    import torch  # noqa

    from tests.torch.conftest import *  # noqa
except ImportError:
    pass
