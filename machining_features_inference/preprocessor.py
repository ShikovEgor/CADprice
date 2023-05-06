import json
import os
import pathlib
import signal
from logging import Logger
from multiprocessing.pool import Pool

import dgl
import numpy as np
import torch
from occwl.compound import Compound
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm

from machining_features_inference.settings import UVNetPipelineSettings


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Preprocessor:
    def __init__(self, settings: UVNetPipelineSettings, logger: Logger) -> None:
        self._settings = settings
        self._logger = logger

    def build_graph(self, solid: Compound) -> dgl.graph:
        # Build face adjacency graph with B-rep entities as node and edge features
        graph = face_adjacency(solid)

        # Compute the UV-grids for faces
        graph_face_feat = []
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]
            # Compute UV-grids
            points = uvgrid(
                face,
                method="point",
                num_u=self._settings.surf_num_u_samples,
                num_v=self._settings.surf_num_v_samples,
            )

            normals = uvgrid(
                face,
                method="normal",
                num_u=self._settings.surf_num_u_samples,
                num_v=self._settings.surf_num_v_samples,
            )
            visibility_status = uvgrid(
                face,
                method="visibility_status",
                num_u=self._settings.surf_num_u_samples,
                num_v=self._settings.surf_num_v_samples,
            )
            mask = np.logical_or(
                visibility_status == 0, visibility_status == 2
            )  # 0: Inside, 1: Outside, 2: On boundary
            # Concatenate channel-wise to form face feature tensor
            face_feat = np.concatenate((points, normals, mask), axis=-1)
            graph_face_feat.append(face_feat)
        graph_face_feat = np.asarray(graph_face_feat)

        # Compute the U-grids for edges
        graph_edge_feat = []
        for edge_idx in graph.edges:
            # Get the B-rep edge
            edge = graph.edges[edge_idx]["edge"]
            # Ignore dgenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # Compute U-grids
            points = ugrid(
                edge,
                method="point",
                num_u=self._settings.curv_num_u_samples,
            )
            tangents = ugrid(
                edge,
                method="tangent",
                num_u=self._settings.curv_num_u_samples,
            )
            # Concatenate channel-wise to form edge feature tensor
            edge_feat = np.concatenate((points, tangents), axis=-1)
            graph_edge_feat.append(edge_feat)
        graph_edge_feat = np.asarray(graph_edge_feat)

        # Convert face-adj graph to DGL format
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
        dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
        dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
        return dgl_graph

    def process_one_file(self, filename: pathlib.Path) -> None:
        try:
            fn_stem = filename.stem
            output_path = pathlib.Path(self._settings.output_collection)
            solid, mapping = Compound.load_step_with_attributes(str(filename))

            if self._settings.convert_labels:
                label_dir = output_path.joinpath("labels")
                if not label_dir.exists():
                    os.mkdir(label_dir)

                labels = [
                    int(mapping[face]["name"])
                    for face in solid.faces()
                    if face in mapping
                ]
                with open(label_dir.joinpath(fn_stem + ".json"), "w") as f:
                    json.dump(labels, f)
            graph = self.build_graph(solid)
            dgl.data.utils.save_graphs(
                str(output_path.joinpath(fn_stem + ".bin")), [graph]
            )  # _color
        except Exception as e:
            self._logger.error(f"Error processing {filename}: {str(e)}")

    def process(self) -> None:
        input_path = self._settings.input_collection
        output_path = self._settings.output_collection
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Created output directory {output_path}")
        step_files = list(input_path.glob("*.st*p"))
        self._logger.info(f"Found {len(step_files)} STEP files to preprocess")
        pool = Pool(processes=self._settings.num_processes, initializer=initializer)
        try:
            results = list(
                tqdm(
                    pool.imap(self.process_one_file, step_files), total=len(step_files)
                )
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        self._logger.info(f"Processed {len(results)} files.")
