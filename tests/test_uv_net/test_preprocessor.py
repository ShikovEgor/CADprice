import json
import os

from pytest import mark

from machining_features_inference.datasets.universal_dataset import \
    UniversalDataset
from machining_features_inference.preprocessor import Preprocessor
from tests.base import (cleanup_dir, inference_settings, learning_settings,
                        logger)


def test_processor():
    cleanup_dir(inference_settings.output_collection)
    processor = Preprocessor(inference_settings, logger)

    file = list(inference_settings.input_collection.glob("*.st*p"))[0].stem + ".bin"
    processor.process()
    assert file in os.listdir(inference_settings.output_collection)


def test_processor_labels():
    cleanup_dir(learning_settings.output_collection)

    processor = Preprocessor(learning_settings, logger)
    file = list(learning_settings.input_collection.glob("*.st*p"))[0].stem
    processor.process()
    assert file + ".bin" in os.listdir(learning_settings.output_collection)
    assert file + ".json" in os.listdir(
        learning_settings.output_collection.joinpath("labels")
    )
    with open(
        learning_settings.output_collection.joinpath("labels").joinpath(file + ".json"),
        "r",
    ) as f:
        labels = json.load(f)
        assert len(labels) > 0
        assert all([isinstance(x, int) for x in labels])


@mark.parametrize("split,expected", [(None, 51), ("train", 41), ("val", 10)])
def test_universal_dataset(split, expected):
    ls = learning_settings.copy()
    ls.train_eval_share = 0.8 if split is not None else None
    dataset = UniversalDataset(settings=ls, split=split, logger=logger)
    assert len(dataset) == expected
    for g in dataset.data:
        assert len(g["graph"].ndata["y"]) > 0
