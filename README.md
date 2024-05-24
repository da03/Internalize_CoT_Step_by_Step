# Internalize CoT Step by Step

This repository contains code to reproduce the results from our paper "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step".

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers) (`pip install transformers`)

## Data & Pretrained Models & Logs

All dataset files and log files during inference are included in this repository, with the exception of large training files maintained under Git LFS. Model checkpoints are stored on Google Drive. The folder containing all checkpoints can be found at [this link]().

* 4 X 4 Mult - GPT-2: [data](data/4_by_4_mult/) [model]() [log](logs/4_by_4_mult/gpt2/log.generate)
* 5 X 5 Mult - GPT-2: [data](data/5_by_5_mult/) [model]() [log](logs/5_by_5_mult/gpt2/log.generate)
* 7 X 7 Mult - GPT-2: [data](data/7_by_7_mult/) [model]() [log](logs/7_by_7_mult/gpt2/log.generate)
* 9 X 9 Mult - GPT-2: [data](data/9_by_9_mult/) [model]() [log](logs/9_by_9_mult/gpt2/log.generate)
* 11 X 11 Mult - GPT-2: [data](data/11_by_11_mult/) [model]() (partially internalized) [log](logs/11_by_11_mult/gpt2/log.generate)
* GSM8K - GPT-2: [data](data/gsm8k/) [model]() [log](logs/gsm8k/gpt2/log.generate)
* GSM8K - GPT-2 Medium: [data](data/gsm8k/) [model]() [log](logs/gsm8k/gpt2-medium/log.generate)
* GSM8K - Phi-3 3.8B: [data](data/gsm8k/) [model]() [log](logs/gsm8k/phi3-3.8B/log.generate)
* GSM8K - Mistral 7B: [data](data/gsm8k/) [model]() [log](logs/gsm8k/mistral-7B/log.generate)

## Additional Datasets

We have included more multiplication datasets than those used in the paper to encourage future research that might yield even better results.

* 6 X 6 Mult: [data](data/6_by_6_mult/)
* 8 X 8 Mult: [data](data/8_by_8_mult/)
* 10 X 10 Mult: [data](data/10_by_10_mult/)
* 12 X 12 Mult: [data](data/12_by_12_mult/)
* 13 X 12 Mult: [data](data/13_by_13_mult/)
* 14 X 12 Mult: [data](data/14_by_14_mult/)
* 15 X 12 Mult: [data](data/15_by_15_mult/)
* 16 X 12 Mult: [data](data/16_by_16_mult/)
* 17 X 12 Mult: [data](data/17_by_17_mult/)
* 18 X 12 Mult: [data](data/18_by_18_mult/)
* 19 X 12 Mult: [data](data/19_by_19_mult/)
* 20 X 12 Mult: [data](data/20_by_20_mult/)

## Usage

We use 7 X 7 Mult with GPT-2 as an example. We assume that the working directory is `Internalize_CoT_Step_by_Step` throughout this document.

### Data Format

The format of training, validation, and test files is as follows:

```
[input 1]||[chain-of-thought 1] #### [output 1]
[input 2]||[chain-of-thought 2] #### [output 3]
[input 3]||[chain-of-thought 2] #### [output 3]
...
```

For example, the first line from the 4 X 4 Mult test set in [data/4_by_4_mult/test_bigbench.txt](data/4_by_4_mult/test_bigbench.txt) is:

```
9 1 7 3 * 9 4 3 3||1 7 4 3 3 + 0 6 7 8 4 1 ( 1 3 2 2 8 1 ) + 0 0 7 5 1 1 1 ( 1 3 9 7 9 2 1 ) + 0 0 0 7 5 1 1 1 #### 1 3 9 4 5 4 2 1
```

In this example, the input is `9 1 7 3 * 9 4 3 3` (corresponding to `3719*3349`, note that we reversed the digits), the chain-of-thought is `1 7 4 3 3 + 0 6 7 8 4 1 ( 1 3 2 2 8 1 ) + 0 0 7 5 1 1 1 ( 1 3 9 7 9 2 1 ) + 0 0 0 7 5 1 1 1`, and the output is `1 3 9 4 5 4 2 1` (corresponding to `12454931`).

Note that the chain-of-thought steps are only used for training, not for generation.

### Training

![](imgs/stepwise_internalization.png)

To train the model, run the following commands. The example uses 7 X 7 Mult with GPT-2:

```
export D=7
export FOLDER=data/${D}_by_${D}_mult/
export MODEL=gpt2
export EPOCHS=200
export LR=5e-5
export BSZ=32
export DELETE_SIDE=left
export DELETE_PER_EPOCH=8
export DELETE_ALL_BEYOND=inf
export DELETE_SCHEDULE_TYPE=step
export DELETION_SMOOTHING_LAMBDA=4
export PRETRAIN_EPOCHS=1
export SEED=1234
export SAVE=train_models/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --remove_per_epoch $DELETE_PER_EPOCH \
    --pretrain_epoch $PRETRAIN_EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --removal_side $DELETE_SIDE \
    --remove_all_beyond $DELETE_ALL_BEYOND \
    --remove_schedule_type $DELETE_SCHEDULE_TYPE \
    --removal_smoothing_lambda $DELETION_SMOOTHING_LAMBDA \
    --reset_optimizer \
    --seed $S \
    --save_model $SAVE \
    > ${SAVE}/log.train 2>&1
```

### Generation & Evaluation

Here we use a pretrained model as an example. Download the folder `models/7_by_7_mult/gpt2`, then the following command will run inference and evaluate both accuracy and throughput, logged in file `generation_logs/7_by_7_mult/log.generate`.

```
export D=7
export START_FROM=inf
export FOLDER=data/${D}_by_${D}_mult/
export MODEL=models/${D}_by_${D}_mult/gpt2
export BSZ=1
export SIDE=left
export SAVE=generation_logs/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --from_pretrained ${MODEL} \
    --start_from ${START_FROM} \
    --test_path ${FOLDER}/test_bigbench.txt \
    --removal_side ${SIDE} \
    --batch_size $BSZ \
    > ${SAVE}/log.generate 2>&1&
```
