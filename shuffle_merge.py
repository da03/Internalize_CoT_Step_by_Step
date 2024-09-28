import sys, os


base = open('data/math/train.txt').readlines()


new = open(sys.argv[1]).readlines()

import random
random.seed(1234)

random.shuffle(new)
new = new[:100000]

new  = new + base
random.shuffle(new)

with open(sys.argv[1]+'.merged', 'w') as fout:
    for line in new:
        fout.write(line.strip()+'\n')
