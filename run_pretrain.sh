export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2


d_block=0
fuse_strategy=gate
cased=0
dropout=0.3
steps=50000
save_steps=5000
log_steps=1000
epoch=1



TASK_NAME=ttee_pretrain_${steps}steps_${d_block}d_2e-5lr_2000_0.96_schedule_${cased}cased_${fuse_strategy}_${dropout}dropout_1loss
SAVE_DIR="./checkpoint"
OUTPUT_DIR="./output"
echo ${TASK_NAME}

python train.py \
    --do_train \
    --mask_sb \
    --cased=${cased} \
    --dataset="pretrain" \
    --model_type="end_to_end" \
    --task_name="${TASK_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --save_dir="${SAVE_DIR}" \
    --epochs=${epoch} \
    --dropout_rate=${dropout} \
    --valid_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --lr=2e-5 \
    --decay_steps=2000 \
    --decay_rate=0.96 \
    --d_block=${d_block} \
    --fuse_strategy=${fuse_strategy} \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --extra_attention \
    --loss_ratio=1 \
    --pretrain \
    --pretrain_steps=${steps} \
    --pretrain_save_steps=${save_steps} \
    --pretrain_log_steps=${log_steps}


# ds=16
# d_block=512
# fuse_strategy=gate
# cased=1
# dropout=0.2
# epoch=200

# TASK_NAME=ttee_res${ds}_${epoch}epoch_${d_block}d_2aug_2e-5lr_5000_0.9_${cased}cased_${fuse_strategy}_${dropout}dropout_dropnull_1loss_BIO
# echo ${TASK_NAME}
# python train.py \
#     --do_train \
#     --do_valid \
#     --do_test \
#     --mask_sb \
#     --cased=${cased} \
#     --dataset="res${ds}" \
#     --model_type="end_to_end" \
#     --task_name="${TASK_NAME}" \
#     --output_dir="${OUTPUT_DIR}" \
#     --save_dir="${SAVE_DIR}" \
#     --epochs=${epoch} \
#     --dropout_rate=${dropout} \
#     --valid_freq=1 \
#     --train_batch_size=16 \
#     --test_batch_size=32 \
#     --lr=2e-5 \
#     --decay_steps=5000 \
#     --decay_rate=0.9 \
#     --d_block=${d_block} \
#     --fuse_strategy=${fuse_strategy} \
#     --schema="BIO" \
#     --block_att_head_num=1 \
#     --language=en \
#     --drop_null_data \
#     --extra_attention \
#     --data_aug=2 \
#     --loss_ratio=1
