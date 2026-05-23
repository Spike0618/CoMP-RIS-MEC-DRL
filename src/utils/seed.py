from __future__ import annotations

"""



-  python / numpy / torch cuda
-  cudnn 
"""

import random
import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    

    
    - seed
    - deterministic cudnn True 
    """
    seed = int(seed)

    
    
    
    random.seed(seed)

    
    np.random.seed(seed)

    
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        
        pass
