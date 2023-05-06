from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Extra


class UVNetPipelineSettings(BaseModel):
    input_collection: Path
    output_collection: Path
    model_repository_path: Path
    model: Optional[str] = None

    num_processes: int = 20

    # dataset settings
    num_classes: int
    convert_labels: bool = False
    center_and_scale: bool = True
    random_rotate: bool = True

    train_eval_share: Optional[float] = None
    batch_size: int = 64

    # preprocessor settings
    curv_num_u_samples: int = 20
    surf_num_u_samples: int = 20
    surf_num_v_samples: int = 20

    # model settings
    device: str = "cuda:0"
    num_epochs: int = 10

    crv_in_channels: int = 6
    srf_emb_dim: int = 64
    crv_emb_dim: int = 128
    graph_emb_dim: int = 128
    dropout: float = 0.3

    skip_labels: List[int] = []  # [18, 11, 5]
    # optimizer
    lr: float = 0.0001
    weight_decay: float = 0.0005

    class Config:
        extra = Extra.forbid
