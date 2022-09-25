export CUDA_VISIBLE_DEVICES=${1:-0}
#export TF_GPU_ALLOCATOR="cuda_malloc_async"

for lr in 6e-5 7e-5 8e-5 9e-5 1e-4 2e-4
do
    TASK_NAME=ttee_res15_100epoch_256d_2aug_${lr}lr_cased_0.2dropout_dropnull_1loss
    SAVE_DIR=""
    OUTPUT_DIR="./output"
    echo "lr: ${lr}; dropout: ${dropout}"
    echo ${TASK_NAME}

    python train.py \
        --do_train \
        --do_valid \
        --do_test \
        --cased \
        --mask_sb \
        --dataset="res15" \
        --model_type="end_to_end" \
        --task_name="${TASK_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --save_dir="${SAVE_DIR}" \
        --epochs=100 \
        --dropout_rate=0.2 \
        --valid_freq=1 \
        --train_batch_size=16 \
        --test_batch_size=32 \
        --lr=${lr} \
        --d_block=256 \
        --fuse_strategy="update" \
        --block_att_head_num=1 \
        --language=en \
        --drop_null_data \
        --extra_attention \
        --data_aug=2 \
        --loss_ratio=1
done
#    --init_model_dir=checkpoint/v2_e2e_40_2_layer_pool_256_h_2e-5_aug2_cased_update_dropout_0.1_drop_null

# --init_model_dir \
