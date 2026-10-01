from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# Abdominal soft-tissue window. Pancreas ~ 40-150 HU, tumour is hypo-attenuating.
HU_MIN, HU_MAX = -160, 240


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cases():
    return pd.read_csv(PROC / "cases.csv")
