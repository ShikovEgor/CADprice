import json
import logging
import pathlib
from typing import Optional

import dgl
import torch
from dgl.data.utils import load_graphs
from torch import FloatTensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from uv_net_pipeline.datasets import util
from uv_net_pipeline.settings import UVNetPipelineSettings


class UniversalDataset(Dataset):
    def __init__(
        self,
        settings: UVNetPipelineSettings,
        logger: logging.Logger,
        split: Optional[str] = None,
    ):
        """
        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
        """
        self._logger = logger
        self._settings = settings
        path = pathlib.Path(settings.output_collection)
        self.raw_dir = path.parent.joinpath("raw")
        self.path = path
        self.labels_path = path.joinpath("labels")
        self._num_classes = settings.num_classes

        all_files = path.glob("*.bin")

        if split is not None:
            splits = util.split_list_by_share(
                all_files,
                ratios=[
                    self._settings.train_eval_share,
                    1 - self._settings.train_eval_share,
                ],
            )
            if split == "train":
                all_files = splits[0]
            elif split == "val":
                all_files = splits[1]
            else:
                raise ValueError(f"{split} is incorrect split name!")

        self.random_rotate = settings.random_rotate
        # Load graphs
        self.load_graphs(all_files)
        self._logger.info("Done loading {} files".format(len(self.data)))

    def num_classes(self):
        return self._num_classes

    def load_graphs(self, file_paths):
        self.data = []
        for fn in tqdm(file_paths):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)

            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        if self._settings.center_and_scale:
            self._center_and_scale()
        self._convert_to_float32()

    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        sample = {"graph": graph, "filename": file_path.stem}
        if self._settings.convert_labels:
            label_file = self.labels_path.joinpath(file_path.stem + ".json")
            with open(str(label_file), "r") as read_file:
                labels_data = [int(x) for x in json.load(read_file)]
            sample["graph"].ndata["y"] = torch.tensor(labels_data).long()
        return sample

    def _center_and_scale(self):
        for i in range(len(self.data)):
            (
                self.data[i]["graph"].ndata["x"],
                center,
                scale,
            ) = util.center_and_scale_uvgrid(
                self.data[i]["graph"].ndata["x"], return_center_scale=True
            )
            self.data[i]["graph"].edata["x"][..., :3] -= center
            self.data[i]["graph"].edata["x"][..., :3] *= scale

    def _convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = (
                self.data[i]["graph"].ndata["x"].type(FloatTensor)
            )
            self.data[i]["graph"].edata["x"] = (
                self.data[i]["graph"].edata["x"].type(FloatTensor)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.random_rotate:
            rotation = util.get_random_rotation()
            sample["graph"].ndata["x"] = util.rotate_uvgrid(
                sample["graph"].ndata["x"], rotation
            )
            sample["graph"].edata["x"] = util.rotate_uvgrid(
                sample["graph"].edata["x"], rotation
            )
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self._settings.model_factory_settings.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self._settings.num_processes,  # Can be set to non-zero on Linux
            drop_last=True,
        )
