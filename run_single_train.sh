export CUDA_VISIBLE_DEVICES=${1:-0}

LOG_PATH=${2:- ""}
SAVE_DIR=${3:- ""}
echo ${LOG_PATH}
echo ${SAVE_DIR}

python train.py \
    --do_train \
    --do_valid \
    --do_test \
    --cased \
    --mask_sb \
    --model_type="end_to_end" \
    --log_path="${LOG_PATH}" \
    --save_dir="${SAVE_DIR}" \
    --epochs=70 \
    --dropout_rate=0.1 \
    --valid_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --lr=2e-5 \
    --d_block=256 \
    --fuse_strategy="update" \
    --block_att_head_num=1 \
    --language=en \
    --drop_null_data \
    --extra_attention \
    --data_aug=2 \
    --loss_ratio=8