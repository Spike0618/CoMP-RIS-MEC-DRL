from __future__ import annotations

"""
PyTorch 


-  PyTorch  `weights_only` 
-  `weights_only=True`
-  checkpoint  `weights_only=False` 
"""

from pathlib import Path
from typing import Any, Union

import torch


def safe_torch_load(
    path: Union[str, Path],
    map_location: Any = "cpu",
    *,
    allow_unsafe_fallback: bool = True,
) -> Any:
    """
     torch.load 

    
    - path: checkpoint 
    - map_location: torch.load  map_location
    - allow_unsafe_fallback:  weights_only=False
    """
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        
        return torch.load(p, map_location=map_location)
    except Exception:
        if not bool(allow_unsafe_fallback):
            raise
        try:
            return torch.load(p, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(p, map_location=map_location)
