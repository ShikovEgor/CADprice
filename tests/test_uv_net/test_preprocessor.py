import os
import pathlib
from uv_net_pipeline.settings import UVNetPipelineSettings
from uv_net_pipeline.preprocessor import Preprocessor

from logging import getLogger, basicConfig

FORMAT = "%(asctime)s %(clientip)-15s %(user)-8s %(message)s"
basicConfig(format=FORMAT)
logger = getLogger("uvnet")
settings = UVNetPipelineSettings(
    input_collection=pathlib.Path(os.path.abspath(__file__)).parent.parent.joinpath("inputs"),
    output_collection=pathlib.Path(os.path.abspath(__file__)).parent.parent.joinpath("outputs"),
)

processor = Preprocessor(settings, logger)


def test_processor():
    file = os.listdir(settings.input_collection)[0][:-4]+".bin"
    processor.process()
    assert file in os.listdir(settings.output_collection)
