from logging import Logger

from uv_net_pipeline.datasets.universal_dataset import UniversalDataset
from uv_net_pipeline.preprocessor import Preprocessor
from uv_net_pipeline.settings import UVNetPipelineSettings
from uv_net_pipeline.uvnet.segmentation import Segmentation


class UVNetLearningPipeline:
    def __init__(self, settings: UVNetPipelineSettings, logger: Logger):
        self._logger = logger
        self._settings = settings
        self._model = Segmentation(settings, logger)
        self._preprocessor = Preprocessor(settings, logger)

    def run(self):
        self._logger.info("Preprocessing data...")
        self._preprocessor.process()

        self._logger.info(f"Train/eval set as: {self._settings.train_eval_share}")
        self._logger.info("Loading train data...")
        train_data = UniversalDataset(
            settings=self._settings, logger=self._logger, split="train"
        ).get_dataloader()
        self._logger.info("Loading eval data...")
        val_data = UniversalDataset(
            settings=self._settings, logger=self._logger, split="val"
        ).get_dataloader()

        self._logger.info("Start learning process...")
        for epoch in range(1, self._settings.model_factory_settings.num_epochs):
            self._logger.info(f"Epoch: {epoch}")
            self._model.train(train_data)
            self._model.valid(val_data)

        # self._model.save_model()
