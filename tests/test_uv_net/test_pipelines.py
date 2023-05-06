import os

from machining_features_inference.pipelines import (InferencePipeline,
                                                    LearningPipeline)
from tests.base import (cleanup_dir, inference_settings, learning_settings,
                        logger)


def test_learning_pipeline():
    cleanup_dir(learning_settings.output_collection.parent)
    p = LearningPipeline(learning_settings, logger)
    p.run()
    assert learning_settings.output_collection.exists()
    assert learning_settings.model_repository_path.exists()
    model_path = learning_settings.model_repository_path.joinpath(
        os.listdir(learning_settings.model_repository_path)[0]
    )
    assert "model" in os.listdir(model_path)


def test_inference_pipeline():
    inference_settings.model = inference_settings.model_repository_path.joinpath(
        os.listdir(inference_settings.model_repository_path)[0]
    )
    p = InferencePipeline(inference_settings, logger)
    p.run()
    assert "inferred_labels.json" in os.listdir(
        inference_settings.output_collection.parent
    )
