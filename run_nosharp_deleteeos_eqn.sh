#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 7-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
export BASE=/n/shieber_lab/Lab/yuntian/cascade/Internalize_CoT_Step_by_Step_gsm8k
cd $BASE
conda activate /n/shieber_lab/Lab/yuntian/icml/conda_icml
which python


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
export SAVE=train_models/gsm8k_nosharp_deleteeos_nospace0
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_nosharp_deleteeos_nospace.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/${SAVE} \
    > ${SAVE}/log.train 2>&1
