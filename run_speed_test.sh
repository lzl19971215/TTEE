export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
for n_aspect in 4 8 16 32 64 121
do
    # python data_preprocessing_for_TAS.py --dataset laptop --output_folder aspect_${n_aspect} --n_aspect_per_sentence ${n_aspect} --sample_test
    python train.py \
        --speed_test \
        --do_train \
        --do_valid \
        --mask_sb \
        --cased=0 \
        --dataset="laptop_acos" \
        --model_type="end_to_end" \
        --task_name="speed_test_aspect_${n_aspect}" \
        --epochs=1 \
        --dropout_rate=2e-5 \
        --valid_freq=1 \
        --train_batch_size=16 \
        --test_batch_size=32 \
        --aspect_senti_batch_size=72 \
        --aspect_senti_test_batch_size=144 \
        --lr="2e-5" \
        --decay_steps=5000 \
        --decay_rate=0.91 \
        --d_block=128 \
        --fuse_strategy="gate" \
        --schema="BIO" \
        --block_att_head_num=1 \
        --language=en \
        --drop_null_data \
        --extra_attention \
        --data_aug=1 \
        --loss_ratio=1 \
        --neg_sample=$(( n_aspect * 3 ))
done