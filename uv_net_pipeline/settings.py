from pydantic import BaseModel
from pathlib import Path


class UVNetPreprocessorSettings(BaseModel):
    curv_num_u_samples: int = 20
    surf_num_u_samples: int = 20
    surf_num_v_samples: int = 20


class UVNetModelFactorySettings(BaseModel):
    batch_size: int = 512
    crv_in_channels: int = 6
    srf_emb_dim: int = 64
    graph_emb_dim: int = 128
    dropout: float = 0.3
    num_epochs: int = 10


class UVNetPipelineSettings(BaseModel):
    input_collection: Path
    output_collection: Path
    convert_labels: bool = False
    center_and_scale: bool = True
    random_rotate: bool = False
    num_processes: int = 20
    preprocessor_settings: UVNetPreprocessorSettings = UVNetPreprocessorSettings()
    model_factory_settings: UVNetModelFactorySettings = UVNetModelFactorySettings()
