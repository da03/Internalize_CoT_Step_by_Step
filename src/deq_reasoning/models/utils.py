import logging
from tqdm import tqdm

import torch
import wandb

logger = logging.getLogger(__name__)


def train(config, model, train_loader, criterion, optimizer, epoch):
    model.train()

    logger.info(f'\nEpoch {epoch + 1}/{config.epochs}')
    epoch_loss = 0
    total_correct = 0
    total_count = 0

    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(config.device)  # (batch_size, seq_len)
        labels = batch['labels'].to(config.device)  # (batch_size, seq_len)
        attention_mask = batch['attention_mask'].to(config.device)  # (batch_size, seq_len, seq_len)

        optimizer.zero_grad()
        # Get logits and loss from the model
        logits, loss = model.compute_loss(input_ids, attention_mask=attention_mask, labels=labels, criterion=criterion,
                            chunk_size=config.model.max_chunk,max_chunks=config.model.chunk_size,predict_start_idx=batch['input_len'])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
        mask = labels != -100  # Ignore padding tokens (-100 is ignored in CrossEntropyLoss)
        correct = (predictions == labels) & mask  # Only count non-padding tokens
        total_correct += correct.sum().item()
        total_count += mask.sum().item()  # Total number of valid tokens (non-padding)

    avg_loss = epoch_loss / len(train_loader)
    accuracy = total_correct / total_count if total_count > 0 else 0

    logger.info(
            f'Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}'
        )

    wandb.log(
        {
            "epoch": epoch,
            "average_train_loss": avg_loss,
            "training_acc": accuracy,
        }
    )


def val(config, model, val_loader, criterion, epoch):
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(config.device)  # (batch_size, seq_len)
            labels = batch['labels'].to(config.device)  # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(config.device)  # (batch_size, seq_len, seq_len)

            logits = model(input_ids, attention_mask=attention_mask,num_iter=config.model.max_chunk)  # (batch_size, seq_len, vocab_size)

            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            labels = labels.view(-1)  # (batch_size * seq_len)
            
            # Calculate validation loss
            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Calculate validation accuracy
            predictions = torch.argmax(logits, dim=-1)  # (batch_size * seq_len)
            mask = labels != -100  # Ignore padding tokens
            correct = (predictions == labels) & mask  # Only count non-padding tokens
            val_correct += correct.sum().item()
            val_total += mask.sum().item()  # Total number of valid tokens (non-padding)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    logger.info(f'Average Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    wandb.log(
        {
            "epoch": epoch,
            "average_val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        }
    )

