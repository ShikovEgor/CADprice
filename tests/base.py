import os
import pathlib
import shutil
from logging import basicConfig, getLogger

from uv_net_pipeline.settings import (UVNetModelFactorySettings,
                                      UVNetPipelineSettings)


def cleanup_dir(path: pathlib.Path):
    if path.exists():
        shutil.rmtree(path)


FORMAT = "%(asctime)s %(clientip)-15s %(user)-8s %(message)s"
basicConfig(format=FORMAT)
logger = getLogger("uvnet")

learning_settings = UVNetPipelineSettings(
    input_collection=pathlib.Path(os.path.abspath(__file__)).parent.joinpath("inputs"),
    output_collection=pathlib.Path(os.path.abspath(__file__)).parent.joinpath(
        "outputs"
    ),
    num_classes=25,
    train_eval_share=0.8,
    convert_labels=True,
    num_processes=8,
    device="cpu",
    model_factory_settings=UVNetModelFactorySettings(batch_size=8, num_epochs=2),
)

inference_settings = learning_settings.copy()
inference_settings.convert_labels = False
inference_settings.train_eval_share = None
