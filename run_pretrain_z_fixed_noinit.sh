

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
export THOUGHT=5
export NUM_ZS=60
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/noinit_zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --with_z \
    --num_zs ${NUM_ZS} \
    --entropy_loss_lambda ${ENTROPY_LOSS_LAMBDA} \
    --thought_length ${THOUGHT} \
    --remove_start_from 999999999 \
    --train_path ${FOLDER}/countdown_train.txt \
    --val_path ${FOLDER}/countdown_valid.txt \
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
    --max_new_tokens 34 \
    > ${SAVE}/log.train 2>&1&