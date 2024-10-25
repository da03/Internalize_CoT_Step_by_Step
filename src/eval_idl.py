import torch
from transformers import GPT2Tokenizer
import torch.nn.functional as F

# Import your custom GPT2 model and CoTDataset
# Assuming your custom model code is saved in custom_gpt2_model.py
from train_idl import GPT2CustomModel, CoTDataset, generate_sequence, CustomTokenizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom vocabulary and mapping
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '*', '+', '(', ')', '#', '<PAD>', '<EOS>']
char2id = {char: idx for idx, char in enumerate(vocab)}
id2char = {idx: char for char, idx in char2id.items()}
vocab_size = len(vocab)
tokenizer = CustomTokenizer(vocab, char2id, id2char)

vocab_size = tokenizer.vocab_size


# Model parameters (ensure these match the ones used during training)
d_model = 128
n_layers = 4
n_heads = 4
d_ff = 128 * 4
dropout = 0.1
max_seq_len = 256

# Initialize the model
model = GPT2CustomModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    n_layers=n_layers,
    n_heads=n_heads,
    d_model=d_model,
    d_ff=d_ff,
    dropout=dropout,
)
model.load_state_dict(torch.load('./trained_model/pytorch_model.bin', map_location=device))
model.to(device)
model.eval()

# Load validation dataset
val_dataset = CoTDataset(tokenizer, 'data/4_by_4_mult/valid.txt', max_length=max_seq_len, max_size=100)  # Load only 10 examples for testing

# Function to decode tokens
def decode_tokens(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Function to extract the final answer after '####'
def extract_generated_answer(text):
    split_pattern = '####'
    if split_pattern in text:
        return text.split(split_pattern, 1)[1].strip()
    else:
        return text.strip()

# Test the model on the examples
correct = 0
total = 0

for idx, example in enumerate(val_dataset):
    input_len = example['input_len']
    input_tokens = example['input_ids'][:input_len]  # Exclude the EOS token at the end
    input_text = decode_tokens(input_tokens)

    print(f"\nExample {idx + 1}")
    print(f"Input Text: {input_text}")

    # Generate prediction

    generated_text = generate_sequence(model, tokenizer, input_tokens, max_target_length=128, device=device)
    print(f"Generated Text: {generated_text}")

    # Extract the final answer from the generated text
    generated_answer = extract_generated_answer(generated_text)
    print(f"Extracted Generated Answer: {generated_answer}")

    # Extract the target answer from the example
    target_tokens = example['target_ids']  # Include the EOS token
    target_text = decode_tokens(target_tokens)
    target_answer = extract_generated_answer(target_text)
    print(f"Target Answer: {target_answer}")

    # Compare the generated answer with the target answer
    if generated_answer == target_answer:
        print("Result: Correct")
        correct += 1
    else:
        print("Result: Incorrect")
    total += 1

# Calculate and print the accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nTotal Examples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")