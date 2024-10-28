import os
import sys
from pathlib import Path

import cpuinfo
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict


def initialize_config(config: DictConfig) -> None:
    """Initializes the experiment setup, including device configuration and WandB tracking."""

    # Resolve and set the device
    config.device_suggested = config.device_suggested
    if (
        config.device_suggested.startswith("cuda") and not torch.cuda.is_available()
    ) or (config.device_suggested == "mps" and not torch.backends.mps.is_available()):
        config.device = "cpu"
    else:
        config.device = config.device_suggested

    # Record hardware specs
    cpu_specs = cpuinfo.get_cpu_info()
    cpu_specs = f'{cpu_specs["brand_raw"]}, {cpu_specs["count"]} cpus'

    if config.device.startswith("cuda"):
        device_specs = torch.cuda.get_device_name()
    elif config.device == "mps":
        gpu_model = (
            os.popen('system_profiler SPDisplaysDataType | grep "Model"')
            .read()
            .split()[-1]
        )
        n_gpu_cores = (
            os.popen('system_profiler SPDisplaysDataType | grep "Cores"')
            .read()
            .split()[-1]
        )
        device_specs = f"{gpu_model}, {n_gpu_cores} GPU cores"
    else:
        device_specs = cpu_specs

    with open_dict(config):
        config.cpu_specs = cpu_specs
        config.device_specs = device_specs

    # Check for conda or non-conda environment
    executable = sys.executable
    is_conda = "mamba" in executable or "conda" in executable
    with open_dict(config):
        config.is_conda = is_conda

    # Initialize WandB
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        tags=config.wandb.tags,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )
