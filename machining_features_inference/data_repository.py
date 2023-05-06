import json
import os
from logging import Logger
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from machining_features_inference.datasets.universal_dataset import \
    UniversalDataset
from machining_features_inference.settings import UVNetPipelineSettings


class DataRepository:
    def __init__(self, settings: UVNetPipelineSettings, logger: Logger):
        self._settings = settings
        self._logger = logger

    def load_graphs(self, split: Optional[str] = None) -> DataLoader:
        data = UniversalDataset(
            settings=self._settings, logger=self._logger, split=split
        )
        if split == "train" and len(data) < self._settings.batch_size:
            self._settings.batch_size = len(data)
            self._logger.warning(
                f"Batch size is set to {len(data)} because of small dataset."
            )
        return data.get_dataloader()

    def save_labels(self, labels: Dict[str, List[int]]) -> None:
        output_dir = self._settings.output_collection.parent
        if not output_dir.exists():
            os.mkdir(output_dir)
        with open(output_dir.joinpath("inferred_labels.json"), "w") as f:
            json.dump(labels, f)
        self._logger.info(
            f"Labels saved to path {output_dir.joinpath('inferred_labels.json')}"
        )

    def save_graphs(self, graphs) -> None:
        raise NotImplementedError
