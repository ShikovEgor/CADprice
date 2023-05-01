import json
import os
import pathlib
from multiprocessing import Pool

import torch
from occwl.compound import Compound
from tqdm import tqdm

from uv_net_pipeline.datasets.base import BaseDataset


class MFTestDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 2

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
        random_rotate=False,
        allow_list=None,
    ):
        """
        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.allow_list = allow_list
        self.raw_dir = path.parent.joinpath("raw")
        self.path = path

        split_filelist = [x for x in os.listdir(path) if ".bin" in x]

        self.random_rotate = random_rotate

        all_files = []
        for fn in split_filelist:
            if self.allow_list is not None:
                if path.joinpath(fn).stem in self.allow_list:
                    all_files.append(path.joinpath(fn))

        # Load graphs
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def get_mechanical_features(self, filename):
        features = []
        with open(self.raw_dir.joinpath(f"{filename}.FRT"), "r") as f:
            for line in f:
                if line:
                    features.append([x for x in line.split(",") if x and x != "\n"])
        return features

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        pool = Pool(processes=50)
        try:
            samples = list(tqdm(pool.imap(self.load_one_graph, file_paths), total=len(file_paths)))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

        for sample in samples:
            if sample is None or sample["graph"].edata["x"].size(0) == 0:
                continue
            self.data.append(sample)
        if center_and_scale:
            self._center_and_scale()
        self._convert_to_float32()

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        file_path = pathlib.Path(file_path)
        shape, mapping = Compound.load_step_with_attributes(self.raw_dir.joinpath(f"{file_path.stem}.stp"))
        mechanical_features = self.get_mechanical_features(file_path.stem)
        flat_faces = []
        [flat_faces.extend(x) for x in mechanical_features]
        truth = [0] * shape.num_faces()

        index_map = {}
        for i, face in enumerate(shape.faces()):
            try:
                index_map[mapping[face]["name"]] = i
                if mapping[face]["name"] in flat_faces:
                    truth[i] = 1
            except KeyError as e:
                pass
        sample["graph"].ndata["y"] = torch.tensor(truth).long()
        sample["mechanical_features"] = [[index_map[y] for y in x] for x in mechanical_features]
        sample["shape"] = shape
        return sample
