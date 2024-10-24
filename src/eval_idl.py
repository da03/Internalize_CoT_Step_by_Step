import torch
from transformers import GPT2Tokenizer
import torch.nn.functional as F

# Import your custom GPT2 model and CoTDataset
# Assuming your custom model code is saved in custom_gpt2_model.py
from train_idl import GPT2CustomModel, CoTDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is defined

# Model parameters (ensure these match the ones used during training)
d_model = 64
n_layers = 1
n_heads = 1
d_ff = 64 * 4
dropout = 0.1
max_seq_len = 512
vocab_size = len(tokenizer)

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
val_dataset = CoTDataset(tokenizer, 'data/9_by_9_mult/valid.txt', max_length=max_seq_len, max_size=10)  # Load only 10 examples for testing

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
    input_tokens = example['input_ids'][:input_len - 1]  # Exclude the EOS token at the end
    input_text = decode_tokens(input_tokens)

    print(f"\nExample {idx + 1}")
    print(f"Input Text: {input_text}")

    # Generate prediction
    def generate_sequence(model, tokenizer, input_text, max_length=100):
        model.eval()
        device = next(model.parameters()).device

        # Tokenize input text
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        input_len = input_ids.size(1)

        # Prepare attention mask
        max_len = input_len + max_length
        attention_mask = torch.ones((1, max_len, max_len), dtype=torch.long, device=device)
        attention_mask = torch.tril(attention_mask)
        attention_mask[:, input_len:, :input_len] = 0  # Prevent target tokens from attending to input tokens

        generated = input_ids
        for _ in range(max_length):
            outputs = model(generated, attention_mask=attention_mask[:, :generated.size(1), :generated.size(1)])
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text

    generated_text = generate_sequence(model, tokenizer, input_text, max_length=100)
    print(f"Generated Text: {generated_text}")

    # Extract the final answer from the generated text
    generated_answer = extract_generated_answer(generated_text)
    print(f"Extracted Generated Answer: {generated_answer}")

    # Extract the target answer from the example
    target_tokens = example['input_ids'][input_len - 1:]  # Include the EOS token
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