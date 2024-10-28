import torch
import logging
logger = logging.getLogger(__name__)


def generate_sequence(model, tokenizer, input_ids, max_target_length=50, max_chunk=12, device='cpu'):
    model.eval()

    # Convert the input_ids to tensor if it's not already a tensor
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)  # (1, seq_len)

    input_len = input_ids.size(1)  # Length of the input sequence

    # Prepare input_ids with placeholders for target tokens (add padding for target length)
    total_len = input_len + max_target_length  # Total sequence length (input + target)
    input_ids_padded = torch.full((1, total_len), tokenizer.pad_token_id, dtype=torch.long).to(device)
    input_ids_padded[:, :input_len] = input_ids  # Fill the input part, leave target part as padding

    # Prepare attention mask: Input tokens attend to themselves, target tokens attend to input tokens
    attention_mask = torch.ones((1, total_len, total_len), dtype=torch.long).to(device)

    # Run the model to get logits
    with torch.no_grad():
        logits = model(input_ids_padded, attention_mask=attention_mask, num_iter=max_chunk)  # (1, total_len, vocab_size)

    # Get logits for the target portion (starting after input_len)
    target_logits = logits[:, input_len:, :]  # (1, max_target_length, vocab_size)

    # Predict tokens for the target portion
    predicted_tokens = torch.argmax(target_logits, dim=-1)  # (1, max_target_length)

    # Optionally stop at the first EOS token
    eos_positions = (predicted_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
    if len(eos_positions) > 0:
        first_eos = eos_positions[0]
        predicted_tokens = predicted_tokens[:, :first_eos]  # Stop at EOS

    # Decode the predicted tokens into text
    generated_text = tokenizer.decode(predicted_tokens[0].tolist(), skip_special_tokens=True)

    return generated_text


def generate_examples_after_epoch(val_examples, tokenizer, model, max_chunk=12, device='cpu'):
    for idx, example in enumerate(val_examples):
        input_len = example['input_len']
        input_tokens = example['input_ids'][:input_len]  # Only the input portion (without padding)
        input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)

        # Generate the prediction
        generated_text = generate_sequence(model, tokenizer, input_tokens, max_target_length=len(example['target_ids']), max_chunk=max_chunk, device=device)

        # Extract the target answer (what the model should generate)
        target_tokens = example['target_ids']  # Get the target tokens directly from the example
        target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
        
        logger.info(f'\nExample {idx + 1}')
        logger.info(f'Input Text: {input_text}')
        logger.info(f'Prediction: {generated_text}')
        logger.info(f'Target Answer: {target_text}')
