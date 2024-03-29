export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2

SAVE_DIR="checkpoint"
DATASET="res15"
EPOCH=100
OUTPUT_DIR="./output"
DETECT_LOSS="ce"

FUSE_STRATEGY="update"
N_AUG=2
TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_${N_AUG}aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}"
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
    --test_batch_size=64 \
    --aspect_senti_batch_size=-1 \
    --aspect_senti_test_batch_size=-1 \
    --lr=2e-5 \
    --decay_steps=5000 \
    --decay_rate=0.90 \
    --d_block=512 \
    --fuse_strategy="${FUSE_STRATEGY}" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug="${N_AUG}" \
    --loss_ratio=1 \
    --detect_loss="${DETECT_LOSS}" \
    --tau=1

FUSE_STRATEGY="gate"
N_AUG=1
TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_${N_AUG}aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}"
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
    --test_batch_size=64 \
    --aspect_senti_batch_size=-1 \
    --aspect_senti_test_batch_size=-1 \
    --lr=2e-5 \
    --decay_steps=5000 \
    --decay_rate=0.90 \
    --d_block=512 \
    --fuse_strategy="${FUSE_STRATEGY}" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug="${N_AUG}" \
    --loss_ratio=1 \
    --detect_loss="${DETECT_LOSS}" \
    --tau=1

FUSE_STRATEGY="update"
N_AUG=1
TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_${N_AUG}aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}"
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
    --test_batch_size=64 \
    --aspect_senti_batch_size=-1 \
    --aspect_senti_test_batch_size=-1 \
    --lr=2e-5 \
    --decay_steps=5000 \
    --decay_rate=0.90 \
    --d_block=512 \
    --fuse_strategy="${FUSE_STRATEGY}" \
    --schema="BIO" \
    --block_att_head_num=1 \
    --language=en \
    --bert_size=base \
    --drop_null_data \
    --extra_attention \
    --data_aug="${N_AUG}" \
    --loss_ratio=1 \
    --detect_loss="${DETECT_LOSS}" \
    --tau=1

# for IN_ACT in relu gelu
# do
#     TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_2aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}_${OUT_ACT}outact_${IN_ACT}inact"
#     echo ${TASK_NAME}
#     echo ${SAVE_DIR}
#     python train.py \
#         --do_train \
#         --do_valid \
#         --do_test \
#         --mask_sb \
#         --cased=0 \
#         --dataset="${DATASET}" \
#         --model_type="end_to_end" \
#         --task_name="${TASK_NAME}" \
#         --output_dir="${OUTPUT_DIR}" \
#         --save_dir="${SAVE_DIR}" \
#         --epochs=${EPOCH} \
#         --dropout_rate=0.2 \
#         --detect_dropout_rate=0.2 \
#         --valid_freq=1 \
#         --test_freq=1 \
#         --train_batch_size=16 \
#         --test_batch_size=32 \
#         --aspect_senti_batch_size=-1 \
#         --aspect_senti_test_batch_size=-1 \
#         --lr=2e-5 \
#         --decay_steps=5000 \
#         --decay_rate=0.90 \
#         --d_block=512 \
#         --block_output_activation="${OUT_ACT}" \
#         --block_inter_activation="${IN_ACT}" \
#         --fuse_strategy="${FUSE_STRATEGY}" \
#         --schema="BIO" \
#         --block_att_head_num=1 \
#         --language=en \
#         --bert_size=base \
#         --drop_null_data \
#         --extra_attention \
#         --data_aug=2 \
#         --loss_ratio=1 \
#         --detect_loss="${DETECT_LOSS}" \
#         --tau=1 
# done

# OUT_ACT="none"
# for IN_ACT in relu gelu
# do
#     TASK_NAME="ttee_${DATASET}_${EPOCH}epoch_512d_2aug_2e-5lr_5000decaystep_0.90decayrate_0.2dropout_0.2detect_dropout_1loss_BIO_${DETECT_LOSS}_no_asp_batch_${FUSE_STRATEGY}_${OUT_ACT}outact_${IN_ACT}inact_continue"
#     echo ${TASK_NAME}
#     echo ${SAVE_DIR}
#     python train.py \
#         --do_train \
#         --do_valid \
#         --do_test \
#         --mask_sb \
#         --cased=0 \
#         --dataset="${DATASET}" \
#         --model_type="end_to_end" \
#         --task_name="${TASK_NAME}" \
#         --output_dir="${OUTPUT_DIR}" \
#         --save_dir="${SAVE_DIR}" \
#         --epochs=${EPOCH} \
#         --dropout_rate=0.2 \
#         --detect_dropout_rate=0.2 \
#         --valid_freq=1 \
#         --test_freq=1 \
#         --train_batch_size=16 \
#         --test_batch_size=32 \
#         --aspect_senti_batch_size=-1 \
#         --aspect_senti_test_batch_size=-1 \
#         --lr=2e-5 \
#         --decay_steps=5000 \
#         --decay_rate=0.90 \
#         --d_block=512 \
#         --block_inter_activation="${IN_ACT}" \
#         --fuse_strategy="${FUSE_STRATEGY}" \
#         --schema="BIO" \
#         --block_att_head_num=1 \
#         --language=en \
#         --bert_size=base \
#         --drop_null_data \
#         --extra_attention \
#         --data_aug=2 \
#         --loss_ratio=1 \
#         --detect_loss="${DETECT_LOSS}" \
#         --tau=1 
# done 