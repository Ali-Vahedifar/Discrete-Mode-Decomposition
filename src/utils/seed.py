"""
Reproducibility Utilities
=========================

Author: Ali Vahedi (Mohammad Ali Vahedifar)
Affiliation: DIGIT and Department of ECE, Aarhus University, Denmark
Email: av@ece.au.dk
IEEE INFOCOM 2025
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Used to ensure reproducible results as per the paper's
    methodology of averaging over 10 independent runs.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    IEEE INFOCOM 2025
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    Seed function for DataLoader workers.
    
    Author: Ali Vahedi (Mohammad Ali Vahedifar)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
