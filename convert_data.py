import json
import re
from datasets import load_dataset

def extract_cot_and_answer(solution_text):
    # Find all occurrences of \boxed{...}
    boxed_expressions = list(re.finditer(r'\\boxed\{(.+)\}', solution_text, re.DOTALL))
    if boxed_expressions:
        # Get the last \boxed{...}
        last_box = boxed_expressions[-1]
        answer = last_box.group(1).strip()
        # CoT is everything before the last \boxed{...}
        cot = solution_text[:last_box.start()].strip()
        #return cot, answer
        return solution_text.strip(), answer
    else:
        # If no \boxed{} is found, return the whole solution as CoT and an empty answer
        return solution_text.strip(), ''

def sanitize_text(text):
    return text
    # Replace newline characters with spaces
    #return ' '.join(text.splitlines())

# Load the MATH dataset with the 'all' subset
dataset = load_dataset('lighteval/MATH', 'all')

# Split the train split into train and valid (500 examples for valid)
train_valid_split = dataset['train'].train_test_split(test_size=500, seed=42)
train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']  # This is the valid set now

# Open files to write the outputs
with open('train.txt', 'w', encoding='utf-8') as train_file, \
     open('valid.txt', 'w', encoding='utf-8') as valid_file, \
     open('test.txt', 'w', encoding='utf-8') as test_file:

    # Function to process and write each entry
    def process_and_write(entry, file):
        problem = sanitize_text(entry['problem'].strip())
        solution = entry['solution'].strip()

        # Extract CoT and answer from the solution
        cot, answer = extract_cot_and_answer(solution)
        if answer == '':
            return
        cot = sanitize_text(cot)
        answer = sanitize_text(answer)

        # Prepare the line to write
        d = {'problem': problem, 'cot': cot, 'answer': answer}
        line = json.dumps(d) + '\n' #f"{problem}||||||||{cot} ######## {answer}\n"

        # Write the line to the file
        file.write(line)

    # Process train dataset
    for entry in train_dataset:
        process_and_write(entry, train_file)

    # Process valid dataset
    for entry in valid_dataset:
        process_and_write(entry, valid_file)

    # Process test dataset
    for entry in dataset['test']:
        process_and_write(entry, test_file)
