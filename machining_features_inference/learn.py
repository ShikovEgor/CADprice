import pathlib
from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger

from machining_features_inference.pipelines import LearningPipeline
from machining_features_inference.settings import UVNetPipelineSettings

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(message)s"
    basicConfig(format=FORMAT, level=INFO)
    logger = getLogger("uvnet")

    parser = ArgumentParser()
    parser.add_argument("-n", "--num_classes", type=int)
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    learning_settings = UVNetPipelineSettings(
        input_collection=pathlib.Path(args.input_path),
        output_collection=pathlib.Path(args.output_path),
        model_repository_path=pathlib.Path(args.output_path).joinpath("models"),
        num_classes=args.num_classes,
        train_eval_share=0.8,
        convert_labels=True,
        num_processes=8,
        device="cpu",
        batch_size=8,
        num_epochs=2,
    )

    inference_settings = learning_settings.copy()
    inference_settings.convert_labels = False
    inference_settings.train_eval_share = None
    p = LearningPipeline(learning_settings, logger)
    p.run()
