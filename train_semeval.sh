export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2

SAVE_DIR=${2:- ""}
DATASET="res16"
EPOCH=100
OUTPUT_DIR="./output"
DETECT_LOSS="ce";

TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_2aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_0loss_BIO_${DETECT_LOSS}_no_asp_batch_concat_test"

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
    --epochs=${EPOCH} \
    --dropout_rate=0.2 \
    --detect_dropout_rate=0.2 \
    --valid_freq=1 \
    --test_freq=1 \
    --train_batch_size=16 \
    --test_batch_size=32 \
    --aspect_senti_batch_size=-1 \
    --aspect_senti_test_batch_size=-1 \
    --lr=2e-5 \
    --decay_steps=5000 \
    --decay_rate=0.90 \
    --d_block=512 \
    --fuse_strategy="concat" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug=1 \
    --loss_ratio=1 \
    --detect_loss="${DETECT_LOSS}" \
    --data_sample_ratio=0.5 \
    --tau=1 


# DETECT_LOSS="focal";

# TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_2aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.4detect_dropout_0.25loss_BIO_${DETECT_LOSS}_no_asp_batch"

# echo ${TASK_NAME}
# echo ${SAVE_DIR}
# python train.py \
#     --do_train \
#     --do_valid \
#     --do_test \
#     --mask_sb \
#     --cased=0 \
#     --dataset="${DATASET}" \
#     --model_type="end_to_end" \
#     --task_name="${TASK_NAME}" \
#     --output_dir="${OUTPUT_DIR}" \
#     --save_dir="${SAVE_DIR}" \
#     --epochs=${EPOCH} \
#     --dropout_rate=0.2 \
#     --detect_dropout_rate=0.4 \
#     --valid_freq=1 \
#     --test_freq=1 \
#     --train_batch_size=16 \
#     --test_batch_size=32 \
#     --aspect_senti_batch_size=-1 \
#     --aspect_senti_test_batch_size=-1 \
#     --lr=2e-5 \
#     --decay_steps=5000 \
#     --decay_rate=0.90 \
#     --d_block=512 \
#     --fuse_strategy="gate" \
#     --schema="BIO" \
#     --block_att_head_num=1 \
#     --language=en \
#     --bert_size=base \
#     --drop_null_data \
#     --extra_attention \
#     --data_aug=2 \
#     --loss_ratio=1 \
#     --detect_loss="${DETECT_LOSS}" \
#     --loss_ratio=0.25 \
#     --tau=1