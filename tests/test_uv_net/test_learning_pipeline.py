from tests.base import cleanup_dir, learning_settings, logger
from uv_net_pipeline.uv_net_learning_pipeline import UVNetLearningPipeline


def test_learning_pipeline():
    cleanup_dir(learning_settings.output_collection)
    p = UVNetLearningPipeline(learning_settings, logger)
    p.run()
