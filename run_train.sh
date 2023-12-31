export CUDA_VISIBLE_DEVICES=${1:-0}
export TF_CPP_MIN_LOG_LEVEL=2


# 假设我们有一个由空格分隔的列表，每个子列表由冒号分隔
list="1:128:72"

# 使用空格作为外层列表的分隔符，遍历列表
for sublist in $list; 
do
    # 使用内部字段分隔符IFS来分隔子列表的项
    IFS=':' read -r n_aug d_block asp_bs <<< "$sublist"
    echo "num_sentence_merge: ${n_aug}; d_block: ${d_block}; aspect_batch_size: ${asp_bs};"
    TASK_NAME=ttee_laptop_100epoch_${d_block}d_${n_aug}aug_2e-5lr_${decay_steps}decaystep_0.91decayrate_0cased_gate_0.2dropout_dropnull_1loss_BIO
    SAVE_DIR=""
    OUTPUT_DIR="./output"
    echo ${TASK_NAME}
    python train.py \
        --do_train \
        --do_valid \
        --do_test \
        --mask_sb \
        --cased=0 \
        --dataset="laptop_acos" \
        --model_type="end_to_end" \
        --task_name="${TASK_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --save_dir="${SAVE_DIR}" \
        --epochs=100 \
        --dropout_rate=2e-5 \
        --valid_freq=1 \
        --train_batch_size=16 \
        --test_batch_size=32 \
        --aspect_senti_batch_size="${asp_bs}" \
        --aspect_senti_test_batch_size="$(( asp_bs * 2))" \
        --lr="2e-5" \
        --decay_steps=5000 \
        --decay_rate=0.91 \
        --d_block=${d_block} \
        --fuse_strategy="gate" \
        --schema="BIO" \
        --block_att_head_num=1 \
        --language=en \
        --drop_null_data \
        --extra_attention \
        --data_aug="${n_aug}" \
        --loss_ratio=1
done
