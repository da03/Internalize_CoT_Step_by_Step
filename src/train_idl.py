import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer
from dataclasses import dataclass
import copy
import math
from tqdm import tqdm


# Define custom vocabulary and mapping
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '*', '+', '(', ')', '#', '<PAD>', '<EOS>']
char2id = {char: idx for idx, char in enumerate(vocab)}
id2char = {idx: char for char, idx in char2id.items()}
vocab_size = len(vocab)

# Custom tokenizer
class CustomTokenizer:
    def __init__(self, vocab, char2id, id2char):
        self.vocab = vocab
        self.char2id = char2id
        self.id2char = id2char
        self.pad_token_id = char2id['<PAD>']
        self.eos_token_id = char2id['<EOS>']
        self.eos_token = '<EOS>'
        
    def encode(self, text, add_special_tokens=True):
        ids = [self.char2id[char] for char in text if char in self.char2id]
        if add_special_tokens:
            ids.append(self.eos_token_id)  # Add EOS token at the end
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id2char[token_id] for token_id in token_ids if token_id in self.id2char]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in ['<PAD>', '<EOS>']]
        return ''.join(tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=-1, max_size=-1):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print(f'Creating features from dataset file at {file_path}')
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip().split('||') for line in f.readlines() if (len(line.strip()) > 0 and not line.strip().isspace()
                                                                             and len(line.strip().split('||')) == 2)]
        if max_size > 0:
            print(f'Truncated to {max_size}')
            lines = lines[:max_size]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        self.examples = []
        for src, tgt in zip(src_lines, tgt_lines):
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)
            input_text = src.strip()
            target_text = cot + ' ' + ans

            # Tokenize input and target separately
            input_ids = tokenizer.encode(input_text, add_special_tokens=True)
            target_ids = tokenizer.encode(target_text, add_special_tokens=True)

            # Truncate if necessary
            if max_length > 0:
                input_ids = input_ids[:max_length]
                target_ids = target_ids[:max_length]

            self.examples.append({
                'input_ids': input_ids,
                'target_ids': target_ids,
                'input_len': len(input_ids),
                'target_len': len(target_ids)
            })

            if len(self.examples) % 10000 == 0:
                print(f'Processed {len(self.examples)} examples')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

@dataclass
class CoTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # Get input_ids and target_ids separately
        batch_input_ids = [torch.tensor(e['input_ids'], dtype=torch.long) for e in examples]
        batch_target_ids = [torch.tensor(e['target_ids'], dtype=torch.long) for e in examples]

        # Combine input_ids with padding tokens (instead of actual target_ids)
        combined_ids = []
        labels = []

        for input_ids, target_ids in zip(batch_input_ids, batch_target_ids):
            # Create padding for target_ids
            padding = torch.full((len(target_ids),), self.tokenizer.pad_token_id, dtype=torch.long)

            # Concatenate input_ids and padding
            combined = torch.cat((input_ids, padding))
            combined_ids.append(combined)

            # Create labels: input tokens are ignored (-100), target tokens remain for learning
            input_len = len(input_ids)
            label = torch.full((input_len,), -100, dtype=torch.long)  # Ignore input part in the labels
            label = torch.cat((label, target_ids))  # Target tokens are placed after input tokens
            labels.append(label)

        # Pad the combined sequences and labels to the maximum length in the batch
        batch_combined_ids = pad_sequence(combined_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Pad with -100 to ignore during loss

        # Create an attention mask for combined sequence
        batch_size, max_len = batch_combined_ids.size()
        attention_mask = torch.ones((batch_size, max_len, max_len), dtype=torch.long)
        # attention_mask[batch_combined_ids == self.tokenizer.pad_token_id] = 0  # Padding tokens are not attended to
        # print(attention_mask)

        return {
            'input_ids': batch_combined_ids,
            'attention_mask': attention_mask,
            'labels': batch_labels
        }


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * n_heads * d_head)
        qkv = qkv.view(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(2, 0, 1, 3).contiguous()  # (n_heads, batch_size, seq_len, 3 * d_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each is (n_heads, batch_size, seq_len, d_head)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (n_heads, batch_size, seq_len, seq_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)  # Expand for n_heads
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (n_heads, batch_size, seq_len, d_head)
        context = context.permute(1, 2, 0, 3).contiguous()  # (batch_size, seq_len, n_heads, d_head)
        context = context.view(batch_size, seq_len, self.n_heads * self.d_head)  # (batch_size, seq_len, d_model)
        output = self.out_proj(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_head, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadAttention(n_heads, d_model, d_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output = self.attn(self.ln_1(x), attn_mask)
        x = x + self.dropout(attn_output)
        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.dropout(mlp_output)
        return x

class GPT2CustomModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, n_heads, d_model, d_ff, dropout=0.1):
        super(GPT2CustomModel, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, d_model, d_model // n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def generate_sequence(model, tokenizer, input_ids, max_target_length=50, device='cpu'):
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
        logits = model(input_ids_padded, attention_mask=attention_mask)  # (1, total_len, vocab_size)

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


def extract_generated_answer(text):
    split_pattern = '####'
    if split_pattern in text:
        return text.split(split_pattern, 1)[1].strip()
    else:
        return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Train custom GPT-2 model on custom dataset")
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='./trained_model', help='Directory to save the model')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_size', type=int, default=-1, help='Maximum size of the dataset to use')
    parser.add_argument('--truncation', type=int, default=-1, help='Sequence truncation length')

    # Model configuration parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model hidden size')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of hidden layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=128*4, help='Feedforward network hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = CustomTokenizer(vocab, char2id, id2char)

    vocab_size = tokenizer.vocab_size
    max_seq_len = args.max_length

    # Initialize the model
    model = GPT2CustomModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model.to(device)

    # Load the training dataset
    train_dataset = CoTDataset(tokenizer, args.train_path, max_length=args.max_length, max_size=args.max_size)
    train_data_collator = CoTDataCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_data_collator)

    # Load the validation dataset
    val_dataset = CoTDataset(tokenizer, args.val_path, max_length=args.max_length)
    val_data_collator = CoTDataCollator(tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_data_collator)

    # For generating examples
    val_examples = [val_dataset[i] for i in range(min(2, len(val_dataset)))]  # Take first 5 examples

    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        epoch_loss = 0
        total_correct = 0
        total_count = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)  # (batch_size, seq_len)
            labels = batch['labels'].to(device)  # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (batch_size, seq_len, seq_len)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)  # (batch_size, seq_len, vocab_size)

            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            labels = labels.view(-1)  # (batch_size * seq_len)

            # Calculate loss
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)  # (batch_size * seq_len)
            mask = labels != -100  # Ignore padding tokens (-100 is ignored in CrossEntropyLoss)
            correct = (predictions == labels) & mask  # Only count non-padding tokens
            total_correct += correct.sum().item()
            total_count += mask.sum().item()  # Total number of valid tokens (non-padding)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f'Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)  # (batch_size, seq_len)
                labels = batch['labels'].to(device)  # (batch_size, seq_len)
                attention_mask = batch['attention_mask'].to(device)  # (batch_size, seq_len, seq_len)

                logits = model(input_ids, attention_mask=attention_mask)  # (batch_size, seq_len, vocab_size)

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
        print(f'Average Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Generate examples after each epoch
        print(f'\nExample Generations after Epoch {epoch + 1}:')
        for idx, example in enumerate(val_examples):
            input_len = example['input_len']
            input_tokens = example['input_ids'][:input_len]  # Only the input portion (without padding)
            input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)

            # Generate the prediction
            generated_text = generate_sequence(model, tokenizer, input_tokens, max_target_length=128, device=device)

            # Extract the target answer (what the model should generate)
            target_tokens = example['target_ids']  # Get the target tokens directly from the example
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            print(f'\nExample {idx + 1}')
            print(f'Input Text: {input_text}')
            print(f'Prediction: {generated_text}')
            print(f'Target Answer: {target_text}')


    # Save the final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))

if __name__ == '__main__':
    main()
