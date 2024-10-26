# Internalize CoT Step by Step

```
4 8 7 7 1 8 4 2 8 1 1 * 1 2 1 5 1 3 4 3 6 6 2||4 8 7 7 1 8 4 2 8 1 1 0 + 0 8 6 5 5 3 6 9 4 6 3 2 0 ( 4 6 4 3 7 1 1 2 3 8 4 2 0 ) + 0 0 4 8 7 7 1 8 4 2 8 1 1 0 ( 4 6 8 1 5 9 2 0 8 0 3 4 1 0 ) + 0 0 0 0 2 9 8 8 0 4 2 1 9 5 0 ( 4 6 8 1 7 8 1 9 8 4 5 5 0 6 0 ) + 0 0 0 0 4 8 7 7 1 8 4 2 8 1 1 0 ( 4 6 8 1 1 7 9 6 0 3 0 8 8 7 1 0 ) + 0 0 0 0 0 2 5 3 3 5 4 4 7 4 5 3 0 ( 4 6 8 1 1 9 4 0 4 8 4 2 6 2 7 3 0 ) + 0 0 0 0 0 0 6 3 1 1 7 2 9 9 2 7 4 0 ( 4 6 8 1 1 9 0 4 5 9 1 5 5 2 0 1 5 0 ) + 0 0 0 0 0 0 0 2 5 3 3 5 4 4 7 4 5 3 0 ( 4 6 8 1 1 9 0 6 0 3 5 0 0 7 7 5 0 4 0 ) + 0 0 0 0 0 0 0 0 4 0 7 6 0 9 8 4 9 0 7 0 ( 4 6 8 1 1 9 0 6 4 3 2 7 0 6 6 0 0 5 7 0 ) + 0 0 0 0 0 0 0 0 0 4 0 7 6 0 9 8 4 9 0 7 0 ( 4 6 8 1 1 9 0 6 4 7 2 4 7 6 5 9 4 4 8 7 0 ) + 0 0 0 0 0 0 0 0 0 0 8 6 5 5 3 6 9 4 6 3 2 0 #### 4 6 8 1 1 9 0 6 4 7 0 1 3 2 9 5 4 9 4 1 3 0

7 6 9 2 * 1 6 3 6||7 6 9 2 0 + 0 2 0 8 7 1 ( 7 8 9 0 8 1 ) + 0 0 1 0 9 8 0 ( 7 8 0 1 7 0 1 ) + 0 0 0 2 0 8 7 1 #### 7 8 0 3 7 8 8 1


CUDA_VISIBLE_DEVICES=2 python src/train_idl.py --train_path data/11_by_11_mult/train.txt --val_path data/11_by_11_mult/valid.txt --epochs 20  --batch_size 16 --d_model 128 --d_ff 512 --n_layers 1 --n_heads 1 --learning_rate 3e-4 --max_length 1024 --chunk_size 80 --max_chunk 12 --output_dir models/11_by_11_mult/ --max_size -1
```

This repository contains code to reproduce the results from our paper [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/pdf/2405.14838).

## Online Demos

For multiplication, an online demo is available at [gpt2-multiplication](https://huggingface.co/spaces/yuntian-deng/gpt2-multiplication). The inference code and model behind this demo can be accessed in the [app.py](https://huggingface.co/spaces/yuntian-deng/gpt2-multiplication/blob/main/app.py) file. Please note that this online demo uses a slightly different data format than the one used in this repository. Specifically, it uses `=` instead of `<|endoftext|>` as the separator to ensure compatibility with standard Hugging Face transformer generation code, and the `####` symbols in the answer part are removed to provide a cleaner output. Despite these changes, the proposed approach still works effectively, demonstrating the generality of the method. Additionally, the online demo merges data with varying input digit lengths for training, allowing a single model to handle inputs of different digit counts. In contrast, the code in this repository trains one model for each number of digits for controlled experiments.

For GSM8K, an online demo is available at [implicit-cot-math](https://huggingface.co/spaces/yuntian-deng/implicit-cot-math). Note that this online demo internalizes one `</s>` token that separates the CoT from the answer (while keeping the `</s>` token that separates the input from the CoT intact), ensuring compatibility with standard Hugging Face transformer generation code. The final accuracy remains at 52%, nearly the same as the number reported in the paper (51%).

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers) (`pip install transformers`)

## Data & Pretrained Models & Logs

All dataset files and log files during inference are included in this repository, with the exception of large training files maintained under Git LFS. Model checkpoints are stored on Google Drive. The folder containing all model checkpoints and datasets can be found at [this link](https://drive.google.com/drive/folders/1_riWn76CiUWcc-fB6KL__PSBF5zstF1T?usp=sharing).

* 4 X 4 Mult - GPT-2 (Acc: 1.00): [data](data/4_by_4_mult/) [model](https://drive.google.com/drive/folders/1cCqJmmwJA7wg0Q64f51WgASSnLVJ4ahO?usp=sharing) [log](logs/4_by_4_mult/gpt2/log.generate)
* 5 X 5 Mult - GPT-2 (Acc: 1.00): [data](data/5_by_5_mult/) [model](https://drive.google.com/drive/folders/1r80h5rLgAa1NpAzIknuvHB92nDT2j0Hz?usp=sharing) [log](logs/5_by_5_mult/gpt2/log.generate)
* 7 X 7 Mult - GPT-2 (Acc: 0.95): [data](data/7_by_7_mult/) [model](https://drive.google.com/drive/folders/1a8jbCV9d9o243qqsxKvgnwDVT27PfJry?usp=sharing) [log](logs/7_by_7_mult/gpt2/log.generate)
* 9 X 9 Mult - GPT-2 (Acc: 0.99): [data](data/9_by_9_mult/) [model](https://drive.google.com/drive/folders/1WQn00xYt_fuFaw-hsl9u-QavmBn8NUaD?usp=sharing) [log](logs/9_by_9_mult/gpt2/log.generate)
* 11 X 11 Mult - GPT-2 (Acc: 0.74): [data](data/11_by_11_mult/) [model](https://drive.google.com/drive/folders/1JuHal52yxN0oO0Q_FMhjp6Enr54wAVy6?usp=sharing) (partially internalized) [log](logs/11_by_11_mult_320_tokens_removed/gpt2/log.generate)
* GSM8K - GPT-2 (Acc: 0.30): [data](data/gsm8k/) [model](https://drive.google.com/drive/folders/1IUQ26DNqDrX0mfTSm19gqG5k1FxLZsr2?usp=sharing) [log](logs/gsm8k/gpt2/log.generate)
* GSM8K - GPT-2 Medium (Acc: 0.35): [data](data/gsm8k/) [model](https://drive.google.com/drive/folders/1-mJ8h9O8ztx0iWJxtgcP4LEJlxj6Cyln?usp=sharing) [log](logs/gsm8k/gpt2-medium/log.generate)
* GSM8K - Phi-3 3.8B (Acc: 0.31): [data](data/gsm8k/) [model](https://drive.google.com/drive/folders/1Jm1sk62WhDX2exiQmRKI53aW5jKXAZE9?usp=sharing) [log](logs/gsm8k/phi3-3.8B/log.generate)
* GSM8K - Mistral 7B (Acc: 0.51): [data](data/gsm8k/) [model](https://drive.google.com/drive/folders/1azfzWxf2jy1H7XAe-dAhtYFTPuh7tfmd?usp=sharing) [log](logs/gsm8k/mistral-7B/log.generate)

## Additional Datasets

We have included more multiplication datasets than those used in the paper to encourage future research that might yield even better results. The folder containing all datasets can be found at [this link](https://drive.google.com/drive/folders/1dbQYoZNhu4h7aQzp6b6s3CUOd2pMX0sB?usp=sharing).

* 6 X 6 Mult: [data](https://drive.google.com/drive/folders/1FkGl3r4nxJZfbmh_cfNIrdC6GVFqeE_N?usp=sharing)
* 8 X 8 Mult: [data](https://drive.google.com/drive/folders/1F_k292AiCTSSHyLSejQYhApdnQzs57F7?usp=sharing)
* 10 X 10 Mult: [data](https://drive.google.com/drive/folders/15CDOwZIHioEmgno8XlJ1xHqvAtyorBz_?usp=sharing)
* 12 X 12 Mult: [data](https://drive.google.com/drive/folders/1WBwVezO4IdtQAsndebprjiBHq7qiCwFH?usp=sharing)
* 13 X 13 Mult: [data](https://drive.google.com/drive/folders/1XLBOxh-wQZFXMZn5CDuHnj4fmhA0iV4M?usp=sharing)
* 14 X 14 Mult: [data](https://drive.google.com/drive/folders/1mmZH03btFnUFg94rGs5OdUA6Usajb8rq?usp=sharing)
* 15 X 15 Mult: [data](https://drive.google.com/drive/folders/1jw4lTvzoGhnFIec5QCsukKK14_sUWE9w?usp=sharing)
* 16 X 16 Mult: [data](https://drive.google.com/drive/folders/1V8dl6mOfgsdTvraY7S7ypx8fQ76a4aAr?usp=sharing)
* 17 X 17 Mult: [data](https://drive.google.com/drive/folders/1-tepTXJqjJcFbzPrpSI12_lhMTHOEn2T?usp=sharing)
* 18 X 18 Mult: [data](https://drive.google.com/drive/folders/1vc3aCAz7ypv2G2RgpSjKQRVqew0kxe8Q?usp=sharing)
* 19 X 19 Mult: [data](https://drive.google.com/drive/folders/1lL88kkGBI6umMVMs7LB0xw6CHvifcSFF?usp=sharing)
* 20 X 20 Mult: [data](https://drive.google.com/drive/folders/1dRav5OysRX2L-nOpgYpi0fOpDEbgtU_f?usp=sharing)

## Usage

We use 9 X 9 Mult with GPT-2 as an example. We assume that the working directory is `Internalize_CoT_Step_by_Step` throughout this document.

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

To train the model, run the following commands. The example uses 9 X 9 Mult with GPT-2:

```
export D=9
export FOLDER=data/${D}_by_${D}_mult/
export MODEL=gpt2
export EPOCHS=200
export LR=5e-5
export BSZ=32
export ACCUMULATE=1
export REMOVE_PER_EPOCH=8
export REMOVE_ALL_WHEN_REMOVE_BEYOND=inf
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=3456
export SAVE=train_models/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --accumulate ${ACCUMULATE} \
    --remove_per_epoch ${REMOVE_PER_EPOCH} \
    --remove_all_when_remove_beyond ${REMOVE_ALL_WHEN_REMOVE_BEYOND} \
    --removal_smoothing_lambda ${REMOVAL_SMOOTHING_LAMBDA} \
    --removal_side ${REMOVAL_SIDE} \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --seed ${SEED} \
    --reset_optimizer \
    --save_model ${SAVE} \
    > ${SAVE}/log.train 2>&1
```

### Generation & Evaluation

Here we use a pretrained model as an example. Download the folder `models/9_by_9_mult/gpt2`, then the following command will run inference and evaluate both accuracy and throughput, logged in file `generation_logs/9_by_9_mult/log.generate`.

```
export D=9
export FOLDER=data/${D}_by_${D}_mult/
export MODEL=models/${D}_by_${D}_mult/gpt2
export BSZ=1
export SAVE=generation_logs/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --from_pretrained ${MODEL} \
    --test_path ${FOLDER}/test_bigbench.txt \
    --batch_size ${BSZ} \
    > ${SAVE}/log.generate 2>&1&
```

### Command for GSM8K

```
export FOLDER=data/gsm8k
export MODEL=mistralai/Mistral-7B-v0.1
export EPOCHS=80
export LR=1e-5
export BSZ=16
export ACCUMULATE=2
export REMOVE_PER_EPOCH=8
export REMOVE_ALL_WHEN_REMOVE_BEYOND=39
export MAX_LEN_TRAIN=150
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=1234
export SAVE=train_models/gsm8k
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --accumulate ${ACCUMULATE} \
    --remove_per_epoch ${REMOVE_PER_EPOCH} \
    --remove_all_when_remove_beyond ${REMOVE_ALL_WHEN_REMOVE_BEYOND} \
    --removal_smoothing_lambda ${REMOVAL_SMOOTHING_LAMBDA} \
    --removal_side ${REMOVAL_SIDE} \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --seed ${SEED} \
    --reset_optimizer \
    --bf16 \
    --max_len_train ${MAX_LEN_TRAIN} \
    --save_model ${SAVE} \
    > ${SAVE}/log.train 2>&1
```
