import argparse
import logging
import os
import time
from pathlib import Path
from tqdm import tqdm


import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb

from deq_reasoning import utils, dataset, models

logger = logging.getLogger(__name__)
os.environ['WANDB_MODE'] = 'disabled'

OmegaConf.register_new_resolver("eval", eval, use_cache=True)
OmegaConf.register_new_resolver(
    "generate_random_seed", utils.seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="digit_multiplication")
def main(config: DictConfig):

    utils.config.initialize_config(config)

    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config, resolve=True)}")

    # Record total runtime.
    total_runtime = time.time()

    utils.seeding.seed_everything(config)

    device = config.device

    tokenizer, train_loader, val_loader, val_dataset = dataset.data.load_dataloader(config)

    vocab_size = tokenizer.vocab_size

    # Initialize the model
    if config.model.arch == "deq":
        from deq_reasoning.models.implicit.deq import GPT2CustomModel
    elif config.model.arch == "baseline_1":
        from deq_reasoning.models.explicit.baseline_1 import GPT2CustomModel
    elif config.model.arch == "baseline_2":
        from deq_reasoning.models.implicit.baseline_2 import GPT2CustomModel
    elif config.model.arch == "baseline_3":
        from deq_reasoning.models.implicit.baseline_3 import GPT2CustomModel
    else:
        raise NotImplementedError
    
    model = GPT2CustomModel(
        vocab_size=vocab_size,
        max_seq_len=config.dataset.max_seq_len,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
    )
    model.to(device)

    # For generating examples
    val_examples = [val_dataset[i] for i in range(min(2, len(val_dataset)))]  # Take first 5 examples

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens

    for epoch in range(config.epochs):
        models.train(config, model, train_loader, criterion, optimizer, epoch)
        models.val(config, model, val_loader, criterion, epoch)
        logger.info(f'\nExample Generations after Epoch {epoch + 1}:')
        utils.generate.generate_examples_after_epoch(val_examples, tokenizer, model, config.model.max_chunk, device=config.device)

    # Save the final model
    if config.save_model:
        os.makedirs(config.outputs_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.outputs_dir, 'pytorch_model.bin'))

    total_runtime = time.time() - total_runtime
    wandb.log({"total_runtime": total_runtime})


if __name__ == '__main__':
    main()
