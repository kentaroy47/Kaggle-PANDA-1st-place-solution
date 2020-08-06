import argparse
import os
import shutil
import ssl
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from factory import read_yaml
from trainer import LightningModuleReg
from utils import seed_torch, src_backup

# For downloading PyTorch weight
ssl._create_default_https_context = ssl._create_unverified_context


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--config", default="./configs/sample.yaml", type=str, help="config path")
    arg("--gpus", default="0", type=str, help="gpu numbers")
    arg("--kfold", type=int, default=1)
    return parser


def make_output_path(output_path: Path, debug: bool) -> Path:
    if debug:
        name_tmp = output_path.name
        output_path = Path("../output/tmp") / name_tmp
        cnt = 0
        while output_path.exists():
            output_path = output_path.parent / (name_tmp + f"_{cnt}")
            cnt += 1

    output_path.mkdir(parents=True, exist_ok=True)
    output_path.chmod(0o777)
    return output_path


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(
        self, model_name: str, kfold: int, cfg_name: str, filepath: str, **args
    ):
        super(MyModelCheckpoint, self).__init__(**args)
        self.monitor = "val_loss"
        self.model_name = model_name
        self.kfold = kfold
        self.cfg_name = cfg_name
        self.file_path = filepath
        self.latest_path = (
            f"{self.file_path}/{cfg_name}_{model_name}_kfold_{kfold}_latest.pt"
        )
        self.bestloss_path = (
            f"{self.file_path}/{cfg_name}_{model_name}_kfold_{kfold}_bestloss.pt"
        )
        self.mode = "min"
        if self.mode == "min":
            self.monitor_op = np.less
            self.best_model_score = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            self.best_model_score = -np.Inf

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Save latest
        if self.verbose > 0:
            epoch = trainer.current_epoch
            print(f"\nEpoch: {epoch}: Saving latest model to {self.latest_path}")
        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        self._save_model(self.latest_path)

        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor).cpu().numpy()
        if self.monitor_op(current, self.best):
            self.best_model_score = current
            if self.verbose > 0:
                print(f"Saving best model to {self.bestloss_path}")
            if os.path.exists(self.bestloss_path):
                os.remove(self.bestloss_path)
            self._save_model(self.bestloss_path)


class MyLogger(LightningLoggerBase):
    def __init__(self, logger_df_path: Path):
        super(MyLogger, self).__init__()
        self.all_metrics = defaultdict(list)
        self.df_path = logger_df_path

    def experiment(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if step is None:
            step = len(self.all_metrics["step"])

        if "created_at" not in metrics:
            metrics["created_at"] = str(datetime.utcnow())

        if "v_loss" in metrics:
            self.all_metrics["step"].append(step)
            for k, v in metrics.items():
                self.all_metrics[k].append(v)
            metrics_df = pd.DataFrame(self.all_metrics)
            metrics_df = metrics_df[sorted(metrics_df.columns)]
            metrics_df.to_csv(self.df_path, index=False)

    @property
    def name(self) -> str:
        return ""

    @property
    def version(self):
        return ""


def train_a_kfold(cfg: Dict, cfg_name: str, output_path: Path) -> None:
    # Checkpoint callback
    kfold = cfg.Data.dataset.kfold
    checkpoint_callback = MyModelCheckpoint(
        model_name=cfg.Model.base,
        kfold=kfold,
        cfg_name=cfg_name,
        filepath=str(output_path),
        verbose=True,  # print when save result, not must
    )

    # Logger
    logger_name = f"kfold_{str(kfold).zfill(2)}.csv"
    mylogger = MyLogger(logger_df_path=output_path / logger_name)

    # Trainer
    seed_torch(cfg.General.seed)
    seed_everything(cfg.General.seed)
    debug = cfg.General.debug
    trainer = Trainer(
        logger=mylogger,
        max_epochs=5 if debug else cfg.General.epoch,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=False,
        train_percent_check=0.02 if debug else 1.0,
        val_percent_check=0.06 if debug else 1.0,
        gpus=cfg.General.gpus,
        use_amp=cfg.General.fp16,
        amp_level=cfg.General.amp_level,
        distributed_backend=cfg.General.multi_gpu_mode,
        log_save_interval=5 if debug else 200,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
    )

    # # Lightning module and start training
    model = LightningModuleReg(cfg)
    trainer.fit(model)


def main():
    args = make_parse().parse_args()

    # Read Config
    cfg = read_yaml(fpath=args.config)
    cfg.Data.dataset.kfold = args.kfold
    cfg.General.debug = args.debug
    for key, value in cfg.items():
        print(f"    {key.ljust(30)}: {value}")

    # Set gpu
    cfg.General.gpus = list(map(int, args.gpus.split(",")))

    # Make output path
    output_path = Path("../output/model") / Path(args.config).stem
    output_path = make_output_path(output_path, args.debug)

    # Source code backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))
    src_backup_path = output_path / "src_backup"
    src_backup_path.mkdir(exist_ok=True)
    src_backup(input_dir=Path("./"), output_dir=src_backup_path)

    # Train start
    # seed_torch(seed=cfg.General.seed)
    train_a_kfold(cfg, Path(args.config).stem, output_path)


if __name__ == "__main__":
    main()
