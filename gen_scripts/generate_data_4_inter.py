# Generate data
import os
import copy
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

from numpy.random import default_rng

np.random.seed(1234)
num_digits = 9
folder_name = f'long_mult_{num_digits}_inter'
output_folder = os.path.join('data', folder_name)
os.makedirs(output_folder, exist_ok=True)

OVERWRITE_DATA = True

train_file = f'{output_folder}/src1_train.txt'
val_file = f'{output_folder}/src1_valid.txt'
test_file = f'{output_folder}/src1_test.txt'
files = [train_file, val_file, test_file]

if not all([os.path.exists(d) for d in files]):
    if not OVERWRITE_DATA:
        print ('rewriting data since it doesn\'t exist!')
        OVERWRITE_DATA = True

if OVERWRITE_DATA:
    numbers = set([])
    #dataset_size = 2000
    min_num = 10**(num_digits-1)
    max_num = 10**(num_digits)
    span = max_num - min_num
    rng = default_rng(1234)
    #import pdb; pdb.set_trace()
    rand_ids = rng.choice(span*span, size=(10**3 - 10**2)**2, replace=False)

    #numbers = np.random.randint(min_num, max_num, size=dataset_size)
    #numbers = range(min_num, max_num)

    #numbers = set(numbers)#
    #numbers = list(numbers)
    
    #random.seed(1234)
    #random.shuffle(numbers)

    #numbers_reverse = copy.deepcopy(numbers)
    #random.shuffle(numbers_reverse)

    #Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?||<<16-3-4=9>> <<9*2=18>> #### 18

    def left_pad(a, i):
        if i == 0:
            return a
        return '0' * i + a

    def right_pad(a, i):
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
        #num1 = 123
        #num2 = 345
        s = ' '.join(str(num1)[::-1]) + ' * ' + ' '.join(str(num2)[::-1])
        s += '||'

        prods = []
        #import pdb; pdb.set_trace()
        for i, c in enumerate(str(num2)[::-1]):
            c = int(c)
            prod = c * num1
            prod = str(prod)
            prod = left_pad(prod, 1 + len(str(num1)) - len(prod))
            prod = right_pad(prod, i)
            prods.append(int(prod))
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
        max_digits = num_digits * 2
        total = left_pad(total, max_digits-len(total))
        s += ' '.join(total[::-1])
        s = s.replace('  ', ' ')
        verify(num1, num2, s)
        return s


    data = []
    for rand_id in rand_ids:
        count1 = rand_id % span
        count2 = rand_id // span
        num1 = min_num + count1
        num2 = min_num + count2
        example = construct_example(num1, num2)
        data.append(example)
    #for num1 in numbers:
    #    for num2 in numbers:
    #        example = construct_example(num1, num2)
    #        data.append(example)

    #random.seed(1234)
    #random.shuffle(data)

    data_size = len(data)
    val_size = 1000
    test_size = 1000
    train_size = data_size - val_size - test_size

    write_file(data[:train_size], train_file)
    write_file(data[train_size:train_size+val_size], val_file)
    write_file(data[train_size+val_size:], test_file)


    #os.system(f'mkdir -p {os.path.dirname(train_file)}')
    #with open(f'{train_file}.tmp', 'w') as ftrain:
    #    with open(f'{val_file}.tmp', 'w') as fval:
    #        with open(f'{test_file}.tmp', 'w') as ftest:
    #            for num in numbers[:700]:
    #                ftest.write(f'{" ".join([c for c in str(num)])}\n')
    #            for num in numbers[700:1400]:
    #                fval.write(f'{" ".join([c for c in str(num)])}\n')
    #            for num in numbers[1400:]:
    #                ftrain.write(f'{" ".join([c for c in str(num)])}\n')

    #def transform_data(input_file, output_file):
    #    with open(input_file) as fin:
    #      with open(output_file, 'w') as fout:
    #        lines = fin.readlines()
    #        numbers1 = lines[:(len(lines)//2)]
    #        numbers2 = lines[(len(lines)//2):]
    #        for number1, number2 in zip(numbers1, numbers2):
    #          num1 = int(number1.replace(' ', '').strip())
    #          num2 = int(number2.replace(' ', '').strip())
    #          result = num1 + num2
    #          result = str(result)
    #          result = ' '.join([i for i in result])
    #          if True:
    #            number1 = '%010d' % num1
    #            number1 = ' '.join([i for i in number1])
    #            number2 = '%010d' % num2
    #            number2 = ' '.join([i for i in number2])
    #            result = int(result.replace(' ', ''))
    #            result = '%011d' % result
    #            result = ' '.join([i for i in result])
    #          fout.write(f'{number1.strip()} + {number2.strip()} = {result}\n')

    #transform_data(f'{train_file}.tmp', train_file)
    #transform_data(f'{val_file}.tmp',   val_file)
    #transform_data(f'{test_file}.tmp',  test_file)
