import json
import os
import pathlib

import torch

from uv_net_pipeline.datasets.base import BaseDataset


class MFCADPSDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 10

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the MFCAD dataset from:
        Weijuan Cao, Trevor Robinson, Yang Hua, Flavien Boussuge,
        Andrew R. Colligan, and Wanbin Pan. "Graph representation
        of 3d cad models for machining feature recognition with deep
        learning." In Proceedings of the ASME 2020 International
        Design Engineering Technical Conferences and Computers
        and Information in Engineering Conference, IDETC-CIE.
        ASME, 2020.

        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        path = pathlib.Path(root_dir).joinpath(split)
        self.labels_path = pathlib.Path(str(path) + "_labels")
        self.path = path

        assert split in ("train", "val", "test")
        split_filelist = os.listdir(path)

        self.random_rotate = random_rotate

        all_files = []
        for fn in split_filelist:
            all_files.append(path.joinpath(fn))

        # Load graphs
        print(f"Loading {split} data...")
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    @staticmethod
    def reduce_map(labels):
        new_map = {
            0: 2,
            1: 5,
            2: 1,
            3: 1,
            4: 1,
            5: 3,
            6: 3,
            7: 3,
            8: 9,
            9: 9,
            10: 9,
            11: 4,
            12: 4,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 0,
            18: 0,
            19: 0,
            20: 7,
            21: 7,
            22: 6,
            23: 2,
            24: 8,
        }
        return [new_map[x] for x in labels]

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally load the label and store it as node data
        label_file = self.labels_path.joinpath(file_path.stem + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = self.reduce_map([int(x) for x in json.load(read_file)])
        sample["graph"].ndata["y"] = torch.tensor(labels_data).long()
        return sample
