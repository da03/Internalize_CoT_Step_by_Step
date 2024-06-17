#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 7-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
export BASE=/n/shieber_lab/Lab/yuntian/cascade/Internalize_CoT_Step_by_Step
cd $BASE
conda activate /n/shieber_lab/Lab/yuntian/icml/conda_icml
which python

export NUM_DIGITS_MIN=1
export NUM_DIGITS_MAX=15
export FOLDER=/n/rush_lab/Lab/Users/yuntian/implicit/data/long_mult_mixed_${NUM_DIGITS_MIN}_to_${NUM_DIGITS_MAX}_inter_mar1824_includingzero_padinput
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
export MAX_VAL_SIZE=16
export SAVE=train_models/mix_${NUM_DIGITS_MIN}_to_${NUM_DIGITS_MAX}_includingzero_padinput_d${REMOVE_PER_EPOCH}_fixed_val16/cont_gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_mix_fixed.py \
    --model ${MODEL} \
    --from_pretrained /n/holyscratch01/rush_lab/Users/yuntian/cascade_jun12/train_models/mix_1_to_15_includingzero_padinput_d8_fixed_val16/gpt2/checkpoint_7 \
    --remove_start_from 63 \
    --start_epoch_from 8 \
    --data_folder ${FOLDER} \
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
    --max_val_size ${MAX_VAL_SIZE} \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/cascade_jun12/${SAVE} \
    > ${SAVE}/log.train.rerun 2>&1
