export FOLDER=data/math
export MODEL=mistralai/Mathstral-7B-v0.1
export EPOCHS=80
export LR=5e-6
export BSZ=4
export ACCUMULATE=8
export REMOVE_PER_EPOCH=16
export REMOVE_ALL_WHEN_REMOVE_BEYOND=inf
export MAX_LEN_TRAIN=630
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=1234
export SAVE=train_models/math_icot_d${REMOVE_PER_EPOCH}_mathstral_lr${LR}
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
    --remove_start_from 0 \
    --remove_all_when_remove_beyond ${REMOVE_ALL_WHEN_REMOVE_BEYOND} \
    --removal_smoothing_lambda ${REMOVAL_SMOOTHING_LAMBDA} \
    --removal_side ${REMOVAL_SIDE} \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --seed ${SEED} \
    --reset_optimizer \
    --bf16 \
    --max_new_tokens ${MAX_LEN_TRAIN} \
    --max_len_train ${MAX_LEN_TRAIN} \
    --save_model ${SAVE} \
    > ${SAVE}/log.train 2>&1
