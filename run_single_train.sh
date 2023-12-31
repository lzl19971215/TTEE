export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2

SAVE_DIR=${2:- ""}
DATASET="laptop_acos"
TASK_NAME="ttee_${DATASET}_30epoch_256d_1aug_2e-5lr_5000decaystep_0.91decayrate_0.2dropout_1loss_BIO"
OUTPUT_DIR="./output"
echo ${TASK_NAME}
echo ${SAVE_DIR}

python train.py \
    --do_train \
    --do_valid \
    --do_test \
    --mask_sb \
    --cased=0 \
    --dataset="${DATASET}" \
    --model_type="end_to_end" \
    --task_name="${TASK_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --save_dir="${SAVE_DIR}" \
    --epochs=50 \
    --dropout_rate=0.2 \
    --valid_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --aspect_senti_batch_size=72 \
    --aspect_senti_test_batch_size=144 \
    --lr=2e-5 \
    --decay_steps=5000 \
    --decay_rate=0.9 \
    --d_block=128 \
    --fuse_strategy="gate" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug=1 \
    --loss_ratio=1