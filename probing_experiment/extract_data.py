import numpy as np
import os

# Path to the activations
activations_path = 'cached_activations/final/transformer_layer_5_first_pred.npy'

# Load the activations (assuming they are saved as a numpy array)
activations = np.load(activations_path)

# Number of examples
num_examples = activations.shape[0]

# Initialize lists to store expected outputs for each probe
probe1_outputs = []
probe2_outputs = []
probe3_outputs = []
probe4_outputs = []

# Assuming your dataset is a list of strings (one per example)
# Load dataset from file
with open('data/4_by_4_mult/test_bigbench.txt', 'r') as f:
    dataset = [line.strip() for line in f.readlines()]

for idx, data_point in enumerate(dataset):
    tokens = data_point.split(" ")
    
    if len(tokens) < 9:
        print(f"Data point {idx} does not have enough tokens.")
        continue
    
    first_four_digits = tokens[0:4]  # ['a', 'b', 'c', 'd']
    # Reverse them to get ['d', 'c', 'b', 'a']
    reversed_digits = first_four_digits[::-1]
    dcba_str = ''.join(reversed_digits)
    try:
        dcba = int(dcba_str)
        print(dcba)
    except ValueError:
        print(f"Data point {idx}: Unable to convert dcba '{dcba_str}' to integer.")
        continue
    
    try:
        star_index = tokens.index('*')
    except ValueError:
        print(f"Data point {idx} does not contain '*' token.")
        continue

    if len(tokens) <= star_index + 4:
        print(f"Data point {idx} does not have four digits after '*'.")
        continue
    
    next_four_digits = tokens[star_index + 1 : star_index + 5]
    try:
        e = int(next_four_digits[0])
        f = int(next_four_digits[1])
        g = int(next_four_digits[2])
        h = int (next_four_digits[3].split('||')[0])
    except ValueError:
        print(f"Data point {idx}: Non-integer value found in digits after '*'.")
        continue
    

    output1 = dcba * e
    output2 = dcba * (f * 10)
    output3 = dcba * (g * 100)
    output4 = dcba * (h * 1000)
    

    probe1_outputs.append(output1)
    probe2_outputs.append(output2)
    probe3_outputs.append(output3)
    probe4_outputs.append(output4)
    

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{num_examples} data points.")


probe1_outputs = np.array(probe1_outputs)
probe2_outputs = np.array(probe2_outputs)
probe3_outputs = np.array(probe3_outputs)
probe4_outputs = np.array(probe4_outputs)


np.save('probe1_expected_outputs.npy', probe1_outputs)
np.save('probe2_expected_outputs.npy', probe2_outputs)
np.save('probe3_expected_outputs.npy', probe3_outputs)
np.save('probe4_expected_outputs.npy', probe4_outputs)
