import os
import pathlib
import shutil
from logging import basicConfig, getLogger

from machining_features_inference.settings import UVNetPipelineSettings


def cleanup_dir(path: pathlib.Path):
    if path.exists():
        shutil.rmtree(path)


FORMAT = "%(asctime)s %(message)s"
basicConfig(format=FORMAT)
logger = getLogger("uvnet")

learning_settings = UVNetPipelineSettings(
    input_collection=pathlib.Path(os.path.abspath(__file__)).parent.joinpath("inputs"),
    output_collection=pathlib.Path(os.path.abspath(__file__)).parent.joinpath(
        "outputs/converted"
    ),
    model_repository_path=pathlib.Path(os.path.abspath(__file__)).parent.joinpath(
        "outputs/models"
    ),
    num_classes=25,
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
