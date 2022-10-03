export CUDA_VISIBLE_DEVICES=${1:-0}
#export TF_GPU_ALLOCATOR="cuda_malloc_async"


for dropout in 0.2
do
    for ds in 15 16
    do
        TASK_NAME=ttee_res${ds}_100epoch_0d_2aug_2e-5lr_uncased_${dropout}dropout_dropnull_1loss_BIO
        SAVE_DIR=""
        OUTPUT_DIR="./output"
        echo "dataset: res${ds}; d_block: 0; dropout: ${dropout}; num augment: 2"
        echo ${TASK_NAME}

        python train.py \
            --do_train \
            --do_valid \
            --do_test \
            --mask_sb \
            --dataset="res${ds}" \
            --model_type="end_to_end" \
            --task_name="${TASK_NAME}" \
            --output_dir="${OUTPUT_DIR}" \
            --save_dir="${SAVE_DIR}" \
            --epochs=100 \
            --dropout_rate=${dropout} \
            --valid_freq=1 \
            --train_batch_size=16 \
            --test_batch_size=32 \
            --lr=2e-5 \
            --d_block=0 \
            --fuse_strategy="update" \
            --schema="BIO" \
            --block_att_head_num=1 \
            --language=en \
            --drop_null_data \
            --extra_attention \
            --data_aug=2 \
            --loss_ratio=1
    done
done
#    --init_model_dir=checkpoint/v2_e2e_40_2_layer_pool_256_h_2e-5_aug2_cased_update_dropout_0.1_drop_null

# --init_model_dir \
