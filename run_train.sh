export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2
#export TF_GPU_ALLOCATOR="cuda_malloc_async"

cased=0
for dropout in 0.3
do
    for ds in 15
    do
        TASK_NAME=ttee_res${ds}_100epoch_0d_2aug_2e-5lr_${cased}cased_gate_${dropout}dropout_dropnull_1loss_BIO
        SAVE_DIR=""
        OUTPUT_DIR="./output"
        echo "dataset: res${ds}; d_block: 0; dropout: 0.2; num augment: 2; loss_ratio:${loss_ratio}"
        echo ${TASK_NAME}

        python train.py \
            --do_train \
            --do_valid \
            --do_test \
            --mask_sb \
            --cased=${cased} \
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
            --decay_steps=0 \
            --decay_rate=0.92 \
            --d_block=0 \
            --fuse_strategy="gate" \
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
