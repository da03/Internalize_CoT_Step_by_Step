export D=4
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
export L=6
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
export D=4
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
export L=3
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
export D=4
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
export L=0
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
export D=4
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
export L=1
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
export D=4
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
export L=2
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
export D=4
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
export L=4
export SAVE=train_models/${D}_by_${D}_mult/gpt2/${L}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --from_pretrained train_models/4_by_4_mult/gpt2 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --layer ${L} \
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
