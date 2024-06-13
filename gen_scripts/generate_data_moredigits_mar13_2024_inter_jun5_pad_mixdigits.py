# Generate data
import os
import copy
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

from numpy.random import default_rng

def generate_rand_ids(num_digits, size):
    # The first digit ranges from 1 to 9
    #first_digits = np.random.randint(1, 10, size=(size, 1))
    first_digits = np.random.randint(0, 10, size=(size, 1))
    # Generate the remaining 19 digits, which can be 0-9
    remaining_digits = np.random.randint(0, 10, size=(size, num_digits-1))
    # Combine the first digits and remaining digits
    numbers = np.hstack((first_digits, remaining_digits))
    # Convert each row of digits into a single 20-digit number string
    #number_strings = [int(''.join(map(str, row))) for row in numbers]
    number_strings = [''.join(map(str, row)) for row in numbers]
    return number_strings


min_digits = 1
max_digits = 9
num_splits = (max_digits - min_digits + 1) ** 2
#train_lines = []
#val_lines = []
#test_lines = []
top_folder_name = f'data/long_mult_mixed_{min_digits}_to_{max_digits}_inter_mar1824_includingzero_padinput'
os.makedirs(top_folder_name, exist_ok=True)
train_filenames = [f'{top_folder_name}/train_{split_id}.txt' for split_id in range(num_splits)]
#train_files = [open(filename, 'w') for filename in train_filenames]
train_tmp_files = [open(filename+'.tmp', 'w') for filename in train_filenames]
for num_digits_1 in range(min_digits, max_digits+1):
    for num_digits_2 in range(min_digits, max_digits+1):
        print (num_digits_1, num_digits_2)
        folder_name = f'{top_folder_name}/{num_digits_1}_by_{num_digits_2}'
        output_folder = folder_name #os.path.join('data', folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        OVERWRITE_DATA = True
        
        #train_file = f'{output_folder}/train.txt'
        val_file = f'{output_folder}/valid.txt'
        test_file = f'{output_folder}/test_bigbench.txt'
        
        if OVERWRITE_DATA:
            numbers = set([])
            size = (10**3 - 10**2)**2
            gen_size = int(size*1.1)
            if num_digits_1 < 3 or num_digits_2 < 3:
                old_size = size
                size = min(size, (10**num_digits_1 - 10**(num_digits_1-1)) * (10**num_digits_1 - 10**(num_digits_1-1)))
                if size < old_size:
                    gen_size = max(1000, int(size*1.5))
            rand_ids_x = generate_rand_ids(num_digits_1, gen_size)
            rand_ids_y = generate_rand_ids(num_digits_2, gen_size)
        
            def left_pad(a, i):
                if i == 0:
                    return a
                return '0' * i + a
                #return a
        
            def right_pad(a, i):
                if i == 0:
                    return a
                return a + '0' * i
                #return a
            def right_pad_real(a, i):
                if i == 0:
                    return a
                return a + '0' * i
        
            def verify(num1, num2, s):
                first, second = s.split('||')
                a, b = first.strip().split(' * ')
                assert int(a.replace(' ', '')[::-1]) == num1
                assert int(b.replace(' ', '')[::-1]) == num2
                result = second.split(' #### ')[-1]
                assert int(result.replace(' ', '')[::-1]) == num1*num2
        
            def write_file(data, fname):
                with open(fname, 'w') as fout:
                    for d in data:
                        fout.write(d + '\n')
        
            def construct_example(num1, num2):
                s = ' '.join(str(num1)[::-1]) + ' * ' + ' '.join(str(num2)[::-1])
                s += '||'
        
                prods = []
                #import pdb; pdb.set_trace()
                for i, c in enumerate(str(num2)[::-1]):
                    c = int(c)
                    prod = c * int(num1)
                    prod = str(prod)
                    prod = left_pad(prod, 1 + len(str(num1)) - len(prod))
                    prods.append(int(right_pad_real(prod, i)))
                    prod = right_pad(prod, i)
                    #prods.append(int(prod))
                    prod = list(prod)[::-1]
                    prod = ' '.join(prod)
                    if i > 0:
                        prod = ' + ' + prod
                        partial_sum = sum(prods)
                        partial_sum = str(partial_sum)
                        assert len(partial_sum) <= i+1+len(str(num1)), partial_sum
                        partial_sum = left_pad(partial_sum, i+1+len(str(num1)) - len(partial_sum))
                        partial_sum = list(str(partial_sum))[::-1]
                        partial_sum = ' '.join(partial_sum)
                        if i != len(str(num2)[::-1])-1:
                            prod = prod + f' ( {partial_sum} ) '
                    s += prod
                s += ' #### '
                total = sum(prods)
                total = str(total)
                #max_digits = num_digits * 2
                #total = left_pad(total, max_digits-len(total))
                s += ' '.join(total[::-1])
                s = s.replace('  ', ' ')
                verify(int(num1), int(num2), s)
                return s
        
        
            data_size = size
            val_size = 1000
            test_size = 1000
            if size < old_size:
                val_size = min(val_size, int(0.1 * size))
                test_size = min(test_size, int(0.1 * size))
            train_size = data_size - val_size - test_size
            data = []
            i = 0
            seen = set([])
            #with open(train_file, 'w') as ftrain:
            with open(val_file, 'w') as fval:
                with open(test_file, 'w') as ftest:
                    for num1, num2 in zip(rand_ids_x, rand_ids_y):
                        if (num1, num2) in seen:
                            continue
                        seen.add((num1, num2))
                        example = construct_example(num1, num2)
                        if i < train_size:
                            fout = train_tmp_files[i % num_splits]
                            #fout = None #ftrain
                            #train_lines.append(example)
                        elif i < train_size + val_size:
                            fout = fval
                        elif i < train_size + val_size + test_size:
                            fout = ftest
                        else:
                            break
                        i += 1
                        fout.write(example+'\n')
[fout.close() for fout in train_tmp_files]
for filename in train_filenames:
    lines = open(filename+'.tmp').readlines()
    random.shuffle(lines)
    with open(filename, 'w') as fout:
        for line in lines:
            fout.write(line)
[os.remove(filename+'.tmp') for filename in train_filenames]
