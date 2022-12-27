export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2
#export TF_GPU_ALLOCATOR="cuda_malloc_async"

epochs=50
cased=0
ds=15
d_block=0
dropout=0.2
decay_steps=0
fuse=update
TASK_NAME=ttee_res${ds}_70w11mean6wstepspretrain_${epochs}epoch_${d_block}d_2aug_2e-5lr_${decay_steps}_schedule_${cased}cased_${fuse}_${dropout}dropout_dropnull_1loss_BIO
SAVE_DIR="./checkpoint"
OUTPUT_DIR="./output"
echo "dataset: res${ds}; d_block: ${d_block}; dropout: ${dropout}; num augment: 2"
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
    --epochs=${epochs} \
    --dropout_rate=${dropout} \
    --valid_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --lr=2e-5 \
    --decay_steps=${decay_steps} \
    --decay_rate=0.9 \
    --d_block=${d_block} \
    --fuse_strategy="${fuse}" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --drop_null_data \
    --extra_attention \
    --data_aug=2 \
    --loss_ratio=1 \
    --init_model_dir=checkpoint/ttee_pretrain_70w.csv_pretrain_120000steps_0d_2e-5lr_3000_0.96_schedule_0cased_update_meanpool_0.3dropout/ckpt-60000
# --init_model_dir \
