import dataclasses

import numpy as np


@dataclasses.dataclass
class RouteResult:
    org_data: np.ndarray
    order: np.ndarray
    length: float
    search_time_sec: float

    def get_route(self) -> np.ndarray:
        return self.org_data[self.order]
