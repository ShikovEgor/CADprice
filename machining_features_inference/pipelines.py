from abc import ABC, abstractmethod
from logging import Logger

from machining_features_inference.data_repository import DataRepository
from machining_features_inference.model_repository import ModelRepository
from machining_features_inference.preprocessor import Preprocessor
from machining_features_inference.settings import UVNetPipelineSettings
from machining_features_inference.uvnet.mfdetector import MFDetector


class BasePipeline(ABC):
    def __init__(self, settings: UVNetPipelineSettings, logger: Logger):
        self._logger = logger
        self._settings = settings
        self._model = MFDetector(settings, logger)
        self._preprocessor = Preprocessor(settings, logger)
        self._model_repository = ModelRepository(settings, logger)
        self._data_repository = DataRepository(settings, logger)

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class LearningPipeline(BasePipeline):
    def run(self):
        self._logger.info("Preprocessing data...")
        self._preprocessor.process()

        self._logger.info(f"Train/eval set as: {self._settings.train_eval_share}")
        self._logger.info("Loading train data...")
        train_data = self._data_repository.load_graphs("train")
        self._logger.info("Loading eval data...")
        val_data = self._data_repository.load_graphs("val")

        self._logger.info("Start learning process...")
        for epoch in range(1, self._settings.num_epochs + 1):
            self._logger.info(f"Epoch: {epoch}")
            self._model.train(train_data)
            self._model.valid(val_data)

        self._model_repository.save_model(self._model)


class InferencePipeline(BasePipeline):
    def run(self):
        if self._settings.model is None:
            raise ValueError("Model name should be specified!")
        self._logger.info("Preprocessing data...")
        self._preprocessor.process()

        self._logger.info("Loading inference data...")
        data = self._data_repository.load_graphs()
        model = self._model_repository.load_model(self._settings.model)
        self._logger.info("Start machining features inference process...")
        machining_features_labels = model.get_labels(data)
        self._data_repository.save_labels(machining_features_labels)
