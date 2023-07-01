import dataclasses

import numpy as np


@dataclasses.dataclass
class RouteResult:
    org_data: np.ndarray
    order: list[int]
    length: float
    search_time_sec: float

    def get_path(self) -> np.ndarray:
        return self.org_data[self.order]
