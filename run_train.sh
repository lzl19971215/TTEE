export CUDA_VISIBLE_DEVICES=${1:-0}
#export TF_GPU_ALLOCATOR="cuda_malloc_async"

for loss_ratio in 3 4
do
#LOG_PATH=${2:- ""}
#SAVE_DIR=${3:- ""}
LOG_PATH=./logs/v2_e2e_70_2_layer_pool_256_h_aug2_2e-5_cased_dropout_0.1_drop_null_${loss_ratio}loss.log
SAVE_DIR=checkpoint/v2_e2e_70_2_layer_pool_256_h_aug2_2e-5_cased_dropout_0.1_drop_null_${loss_ratio}loss
echo "loss_ratio: ${loss_ratio}"
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
    --loss_ratio=${loss_ratio}
done
#    --init_model_dir=checkpoint/v2_e2e_40_2_layer_pool_256_h_2e-5_aug2_cased_update_dropout_0.1_drop_null
#    --cased \

# --init_model_dir \
