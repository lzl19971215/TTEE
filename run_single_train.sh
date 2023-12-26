export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2

LOG_PATH=${2:- ""}
SAVE_DIR=${3:- ""}
echo ${LOG_PATH}
echo ${SAVE_DIR}

python train.py \
    --task_name="debug" \
    --do_train \
    --do_valid \
    --do_test \
    --cased=0 \
    --schema="BIO" \
    --dataset="laptop_acos" \
    --model_type="end_to_end" \
    --output_dir="${LOG_PATH}" \
    --save_dir="${SAVE_DIR}" \
    --epochs=30 \
    --dropout_rate=0.2 \
    --valid_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=16 \
    --aspect_senti_batch_size=12 \
    --lr=2e-5 \
    --d_block=128 \
    --fuse_strategy="gate" \
    --block_att_head_num=1 \
    --language=en \
    --drop_null_data \
    --extra_attention \
    --data_aug=1 \
    --loss_ratio=1