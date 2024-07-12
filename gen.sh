export FOLDER=data/gsm8k
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/train_models/gsm8k_nosharp_deleteeos/checkpoint_8
export BSZ=1
export SAVE=generation_logs/gsm8k_nosharp_deleteeos/checkpoint_8
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --from_pretrained ${MODEL} \
    --test_path ${FOLDER}/test.txt \
    --batch_size ${BSZ} \
    > ${SAVE}/log.generate 2>&1&


export FOLDER=data/gsm8k
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/train_models/gsm8k_nosharp_deleteeos_nospace/checkpoint_8
export BSZ=1
export SAVE=generation_logs/gsm8k_nosharp_deleteeos_nospace/checkpoint_8
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate_nospace.py \
    --from_pretrained ${MODEL} \
    --test_path ${FOLDER}/test.txt \
    --batch_size ${BSZ} \
    > ${SAVE}/log.generate 2>&1&


export FOLDER=data/gsm8k
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/train_models/gsm8k_nosharp_deleteeos/checkpoint_7
export BSZ=1
export SAVE=generation_logs/gsm8k_nosharp_deleteeos/checkpoint_7
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --from_pretrained ${MODEL} \
    --test_path ${FOLDER}/test.txt \
    --batch_size ${BSZ} \
    > ${SAVE}/log.generate.jul12 2>&1&
