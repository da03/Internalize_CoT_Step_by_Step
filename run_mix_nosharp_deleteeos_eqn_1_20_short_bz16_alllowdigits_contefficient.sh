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
export NUM_DIGITS_MAX=20
#export FOLDER=/n/rush_lab/Lab/Users/yuntian/implicit/data/long_mult_mixed_${NUM_DIGITS_MIN}_to_${NUM_DIGITS_MAX}_inter_mar1824_includingzero_padinput
export FOLDER=/n/holyscratch01/rush_lab/Users/yuntian/cascade_jun12/data/long_mult_mixed_${NUM_DIGITS_MIN}_to_${NUM_DIGITS_MAX}_inter_mar1824_includingzero_padinput_short_alllowdigits
export MODEL=gpt2
export EPOCHS=200
export LR=5e-5
export BSZ=16
export ACCUMULATE=2
export REMOVE_PER_EPOCH=8
export REMOVE_ALL_WHEN_REMOVE_BEYOND=inf
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=3456
export MAX_VAL_SIZE=16
export SAVE=train_models/mix_${NUM_DIGITS_MIN}_to_${NUM_DIGITS_MAX}_includingzero_padinput_d${REMOVE_PER_EPOCH}_short_nosharp_deleteeos_eqn_val16_bsz16_acc2_alllowdigits_contefficient/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_mix_nosharp_deleteeos_eqn.py \
    --from_pretrained /n/holyscratch01/rush_lab/Users/yuntian/cascade_jun12/train_models/mix_1_to_20_includingzero_padinput_d8_short_nosharp_deleteeos_eqn_val16_bsz16_acc2_alllowdigits/gpt2/checkpoint_24 \
    --remove_start_from 200 \
    --start_epoch_from 25 \
    --model ${MODEL} \
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
    --truncation 1120 \
    --max_new_tokens 1050 \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/cascade_jun22/${SAVE} \
    > ${SAVE}/log.train.rerun 2>&1
