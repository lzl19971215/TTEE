export CUDA_VISIBLE_DEVICES=${1:-0}
#export TF_GPU_ALLOCATOR="cuda_malloc_async"


TASK_NAME=ttee_res15_100epoch_nod_tanh_2aug_2e-5lr_uncased_0.3dropout_dropnull_1loss_BIO
SAVE_DIR=""
OUTPUT_DIR="./output"
echo ${TASK_NAME}

python train.py \
    --do_train \
    --do_valid \
    --do_test \
    --mask_sb \
    --dataset="res15" \
    --model_type="end_to_end" \
    --task_name="${TASK_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --save_dir="${SAVE_DIR}" \
    --epochs=100 \
    --dropout_rate=0.3 \
    --valid_freq=5 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --lr=2e-5 \
    --fuse_strategy="update" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --drop_null_data \
    --extra_attention \
    --loss_ratio=1

#    --init_model_dir=checkpoint/v2_e2e_40_2_layer_pool_256_h_2e-5_aug2_cased_update_dropout_0.1_drop_null

# --init_model_dir \
