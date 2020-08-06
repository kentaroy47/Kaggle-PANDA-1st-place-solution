import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def src_backup(input_dir: Path, output_dir: Path):
    for src_path in input_dir.glob("**/*.py"):
        new_path = output_dir / src_path.name
        shutil.copy2(str(src_path), str(new_path))
