from dataclasses import dataclass
import os

import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '*', '+', '(', ')', '#', '<PAD>', '<EOS>']
char2id = {char: idx for idx, char in enumerate(vocab)}
id2char = {idx: char for char, idx in char2id.items()}
vocab_size = len(vocab)

def extract_generated_answer(text):
    """
    Only used for inference on digit multiplication dataset.
    """
    split_pattern = '####'
    if split_pattern in text:
        return text.split(split_pattern, 1)[1].strip()
    else:
        return text.strip()
    

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
    def __init__(self, tokenizer, file_path, max_length=-1, max_size=-1, with_cot=False):
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
            input_text = src.strip()

            if with_cot:
                cot = extract_cot(tgt)
                target_text = cot + ' ' + ans
            else:
                target_text = ans

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
            'labels': batch_labels,
            'input_len': input_len,
        }
