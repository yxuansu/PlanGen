CUDA_VISIBLE_DEVICES=1 python train.py\
    --model_name bert-base-cased\
    --crf_low_rank 64\
    --crf_beam_size 256\
    --train_src_path ../data/train/totto_train_table.txt\
    --train_tgt_path ../data/train/totto_train_content_plan.txt\
    --dev_src_path ../data/dev/totto_dev_table.txt\
    --dev_tgt_path ../data/dev/totto_dev_content_plan.txt\
    --max_src_len 320\
    --max_tgt_len 25\
    --special_token_path ../data/totto_col_header_vocab.txt\
    --min_slot_key_cnt 10\
    --number_of_gpu 1\
    --batch_size_per_gpu 8\
    --gradient_accumulation_steps 16\
    --effective_batch_size 128\
    --total_steps 40000\
    --print_every 100\
    --save_every 500\
    --learning_rate 2e-5\
    --mle_loss_weight 0.5\
    --save_path_prefix ./ckpt/

