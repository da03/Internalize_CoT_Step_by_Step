import random
import operator
from tqdm import tqdm
import os

COUNT = 8

def generate_countdown_examples(num_examples=100, min_target=10, max_target=100):
    """Generate training examples for the countdown game."""
    examples = []
    operations = ['+', '-', '*']
    unique_examples = set()  # To track unique examples
    
    pbar = tqdm(total=num_examples, desc="Generating examples")
    
    while len(examples) < num_examples:
        # Decide if we want 3 or 4 numbers
        #count = random.randint(3, 4)
        count = COUNT
        
        # Generate unique random numbers (sample without replacement)
        numbers = random.sample(range(1, 101), count)
        
        # Create a random permutation of the numbers
        nums = random.sample(numbers, len(numbers))
        
        # Generate random operations
        ops = [random.choice(operations) for _ in range(len(nums)-1)]
        
        # Build the expression string
        expression = ''
        for i in range(len(nums)):
            expression += str(nums[i])
            if i < len(ops):
                expression += f" {ops[i]} "
        expression_out = ''
        for i in range(len(nums)):
            expression_out += ' '.join(f'{nums[i]:03d}')
            if i < len(ops):
                expression_out += f' {ops[i]} '
        
        # Evaluate the expression with proper order of operations
        try:
            # Replace * with * to ensure proper evaluation
            eval_expr = expression.replace(' * ', '*').replace(' + ', '+').replace(' - ', '-')
            result = eval(eval_expr)
            
            # Check if the result is in the desired range
            if min_target <= result <= max_target and result == int(result):
                target = int(result)
                target_str = f'{target:03d}'
                target_str = ' '.join(target_str)
                # Format the example
                #numbers_str = ' '.join(f'{num:03d}' for num in numbers)
                numbers_str = ' , '.join([' '.join(str(num).zfill(3)) for num in numbers])

                input_str = f" {target_str} : {numbers_str}"
                example = f"{input_str}|| #### {expression_out}"
                
                # Only add if it's a unique example
                if example not in unique_examples:
                    unique_examples.add(example)
                    examples.append(example)
                    pbar.update(1)  # Update progress bar
        except:
            # Skip expressions that cause evaluation errors
            continue
    
    pbar.close()
    return examples

# Generate and save examples with train/valid/test splits
def save_examples_with_splits(base_filename, total_examples=100000, valid_size=1000, test_size=1000):
    """Generate examples and split into train, validation, and test sets."""
    train_size = total_examples - valid_size - test_size
    
    print(f"Generating {total_examples} examples (Train: {train_size}, Valid: {valid_size}, Test: {test_size})")
    examples = generate_countdown_examples(total_examples)
    
    # Shuffle examples
    random.shuffle(examples)
    
    # Split into train, validation, and test sets
    train_examples = examples[:train_size]
    valid_examples = examples[train_size:train_size+valid_size]
    test_examples = examples[train_size+valid_size:]
    
    os.makedirs(f"data/countdown{COUNT}", exist_ok=True)
    # Save to separate files
    with open(f"data/countdown{COUNT}/{base_filename}_train.txt", 'w') as f:
        for example in train_examples:
            f.write(example + '\n')
    
    with open(f"data/countdown{COUNT}/{base_filename}_valid.txt", 'w') as f:
        for example in valid_examples:
            f.write(example + '\n')
    
    with open(f"data/countdown{COUNT}/{base_filename}_test.txt", 'w') as f:
        for example in test_examples:
            f.write(example + '\n')
    
    print(f"Saved {len(train_examples)} training examples to {base_filename}_train.txt")
    print(f"Saved {len(valid_examples)} validation examples to {base_filename}_valid.txt")
    print(f"Saved {len(test_examples)} test examples to {base_filename}_test.txt")

# Example usage
save_examples_with_splits('countdown', 500000, 1000, 1000)