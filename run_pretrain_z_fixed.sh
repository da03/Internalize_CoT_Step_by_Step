
export FOLDER=data/countdown8/
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
export THOUGHT=5
export NUM_ZS=10
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=6 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    > ${SAVE}/log.train 2>&1&


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
export THOUGHT=5
export NUM_ZS=20
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    > ${SAVE}/log.train 2>&1&




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
export THOUGHT=5
export NUM_ZS=30
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    > ${SAVE}/log.train 2>&1&




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
export THOUGHT=5
export NUM_ZS=30
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=5.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=4 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    --max_new_tokens 40 \
    > ${SAVE}/log.train 2>&1&







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
export THOUGHT=5
export NUM_ZS=30
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=25.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=5 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    --max_new_tokens 40 \
    > ${SAVE}/log.train 2>&1&





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
export FROM_PRETRAINED=train_models/countdown8/think5/gpt2/checkpoint_25
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    > ${SAVE}/log.train 2>&1&



















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
export THOUGHT=5
export NUM_ZS=30
export FROM_PRETRAINED=train_models/countdown8/zs_think5_zs30_lamb25.0_withlog/gpt2/checkpoint_2
export ENTROPY_LOSS_LAMBDA=25.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_debug/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
    --with_z \
    --num_zs ${NUM_ZS} \
    --entropy_loss_lambda ${ENTROPY_LOSS_LAMBDA} \
    --thought_length ${THOUGHT} \
    --remove_start_from 999999999 \
    --train_path ${FOLDER}/countdown_valid_debug.txt \
    --val_path ${FOLDER}/countdown_valid_debug.txt \
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
    --save_model ${SAVE}


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
export FROM_PRETRAINED=train_models/countdown8/zs_think5_zs60_lamb1.0_withlog_fixed/gpt2/checkpoint_0
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}_withlog_fixed_cont/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    --max_new_tokens 40 \
    > ${SAVE}/log.train 2>&1&




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
export THOUGHT=1
export NUM_ZS=60
export FROM_PRETRAINED=train_models/countdown8/think1/gpt2/checkpoint_5
export ENTROPY_LOSS_LAMBDA=1.0
export SAVE=train_models/fixeval_countdown8/zs_think${THOUGHT}_zs${NUM_ZS}_lamb${ENTROPY_LOSS_LAMBDA}/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained ${FROM_PRETRAINED} \
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
    --max_new_tokens 40 \
    > ${SAVE}/log.train 2>&1&

