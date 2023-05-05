from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class UVNetPreprocessorSettings(BaseModel):
    curv_num_u_samples: int = 20
    surf_num_u_samples: int = 20
    surf_num_v_samples: int = 20


class UVNetModelFactorySettings(BaseModel):
    batch_size: int = 64
    crv_in_channels: int = 6
    srf_emb_dim: int = 64
    crv_emb_dim: int = 128
    graph_emb_dim: int = 128
    dropout: float = 0.3
    num_epochs: int = 10
    skip_labels: List[int] = []  # [18, 11, 5]
    # optimizer
    lr: float = 0.0001
    weight_decay: float = 0.0005


class UVNetPipelineSettings(BaseModel):
    device: str = "cuda:0"
    input_collection: Path
    output_collection: Path
    num_classes: int
    convert_labels: bool = False
    center_and_scale: bool = True
    random_rotate: bool = False
    num_processes: int = 20
    train_eval_share: Optional[float] = None
    preprocessor_settings: UVNetPreprocessorSettings = UVNetPreprocessorSettings()
    model_factory_settings: UVNetModelFactorySettings = UVNetModelFactorySettings()
