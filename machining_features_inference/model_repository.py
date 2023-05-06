import json
import os
import pathlib
from datetime import datetime
from logging import Logger

from machining_features_inference.settings import UVNetPipelineSettings
from machining_features_inference.uvnet.mfdetector import MFDetector


class ModelRepository:
    def __init__(self, settings: UVNetPipelineSettings, logger: Logger):
        self._settings = settings
        self._logger = logger

    def load_model(self, model_name: str) -> MFDetector:
        model_path = self._settings.model_repository_path.joinpath(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model is not exist in path {model_path}!")

        with open(model_path.joinpath("pipeline_settings.json"), "r") as f:
            model_settings = UVNetPipelineSettings.parse_obj(dict(json.load(f)))
        model = MFDetector(settings=model_settings, logger=self._logger)
        model.load_model(model_path.joinpath("model"))
        return model

    def save_model(self, model: MFDetector) -> None:
        if not self._settings.model_repository_path.exists():
            os.mkdir(self._settings.model_repository_path)
            self._logger.info(
                f"Created model repository directory: {self._settings.model_repository_path}"
            )

        iou = round(float(model.val_iou.compute()), 3)

        model_name = f"MFD_{datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S')}_iou_{iou}"
        model_path = self._settings.model_repository_path.joinpath(model_name)
        os.mkdir(model_path)

        with open(model_path.joinpath("pipeline_settings.json"), "w") as f:
            json.dump(
                {
                    k: (v if not isinstance(v, pathlib.Path) else str(v))
                    for k, v in model._settings.dict().items()
                },
                f,
            )

        model.save_model(model_path.joinpath("model"))
