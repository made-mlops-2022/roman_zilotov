from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelParams:
    grid_n_neighbors: List[int]
    model: str = field(default = 'knn')
    n_neighbors: int = field(default = 5)

    n_estimators: int = field(default = 100)
    max_depth: int = field(default = 5)
    random_state: int = field(default = 42)
    grid_search: bool = field(default = True)

@dataclass
class FeatureParams:
    categorical: List[str]
    numerical: List[str]
    target_col: str

@dataclass
class SplittingParams:
    test_size: float = field(default = 0.3)
    random_state: int = field(default = 42)

@dataclass
class TrainPipelineCfg:
    model: ModelParams
    splitting_params: SplittingParams
    feature_params: FeatureParams
