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
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.predictions.classification import CategoricalTarget, EmbeddingTablePrediction
from merlin.models.tf.predictions.sampling.popularity import PopularityBasedSamplerV2
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_binary_prediction_block(ecommerce_data: Dataset, run_eagerly):
    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([8]),
        mm.BinaryPrediction("click"),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "precision",
        "recall",
        "binary_accuracy",
        "auc",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_prediction_block(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema
    schema["item_category"] = schema["item_category"].with_tags(
        schema["item_category"].tags + "target"
    )
    ecommerce_data.schema = schema
    model = mm.Model(
        mm.InputBlock(schema),
        mm.MLPBlock([8]),
        mm.CategoricalPrediction(
            prediction=schema["item_category"],
            negative_samplers=PopularityBasedSamplerV2(max_id=100, max_num_samples=20),
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "ndcg_at_10",
        "precision_at_10",
        "map_at_10",
        "mrr_at_10",
        "recall_at_10",
        "regularization_loss",
    }


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_next_item_prediction(sequence_testing_data: Dataset, run_eagerly):
    dataloader, schema = _next_item_loader(sequence_testing_data)
    embeddings = mm.Embeddings(
        schema,
        sequence_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    )

    predictions = [
        schema.select_by_name("item_id_seq"),
        schema["item_id_seq"],
        CategoricalTarget(schema["item_id_seq"]),
        embeddings["item_id_seq"],
        EmbeddingTablePrediction(embeddings["item_id_seq"]),
    ]

    for target in predictions:
        model = mm.Model(
            mm.InputBlockV2(
                schema,
                embeddings=embeddings,
            ),
            mm.MLPBlock([32]),
            mm.CategoricalPrediction(
                prediction=target,
                negative_samplers=PopularityBasedSamplerV2(max_id=51996, max_num_samples=20),
            ),
        )
        _, history = testing_utils.model_test(model, dataloader, run_eagerly=run_eagerly)


def test_setting_negative_sampling_strategy(sequence_testing_data: Dataset):
    dataloader, schema = _next_item_loader(sequence_testing_data)
    model = mm.Model(
        mm.InputBlockV2(
            schema,
        ),
        mm.MLPBlock([32]),
        mm.CategoricalPrediction(prediction=schema["item_id_seq"]),
    )
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1], training=True)
    assert output.shape == (batch[1].shape[0], 51997)

    model.compile(
        optimizer="adam",
        negative_sampling=[PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )

    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 21)

    model.compile(
        optimizer="adam",
        negative_sampling=["in-batch", PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )
    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 71)


def _next_item_loader(sequence_testing_data: Dataset):
    def _last_interaction_as_target(inputs, targets):
        inputs = mm.AsRaggedFeatures()(inputs)
        items = inputs["item_id_seq"]
        _items = items[:, :-1]
        targets = items[:, -1:].flat_values
        inputs["item_id_seq"] = _items
        return inputs, targets

    schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    sequence_testing_data.schema = schema
    dataloader = BatchedDataset(sequence_testing_data, batch_size=50)
    dataloader = dataloader.map(_last_interaction_as_target)
    return dataloader, schema
