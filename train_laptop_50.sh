export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

SAVE_DIR="checkpoint"
DATASET="laptop_acos"
EPOCH=100
OUTPUT_DIR="./output"
DETECT_LOSS="ce";
FUSE_STRATEGY="gate";

NEG_SAMPLE=50

TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_1aug_2e-5lr_0decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}_${NEG_SAMPLE}neg_16_batch"

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
    --valid_freq=3 \
    --test_freq=3 \
    --train_batch_size=16 \
    --test_batch_size=16 \
    --aspect_senti_batch_size=-1 \
    --aspect_senti_test_batch_size=-1 \
    --lr=2e-5 \
    --decay_steps=-1 \
    --decay_rate=0.90 \
    --d_block=512 \
    --fuse_strategy="${FUSE_STRATEGY}" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug=1 \
    --loss_ratio=1 \
    --detect_loss="${DETECT_LOSS}" \
    --neg_sample=${NEG_SAMPLE}