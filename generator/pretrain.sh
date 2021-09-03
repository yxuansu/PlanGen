CUDA_VISIBLE_DEVICES=2 python3 pretrain.py\
    --train_table_text_path ../data/train/totto_train_table.txt\
    --train_content_text_path ../data/train/totto_train_content_plan.txt\
    --train_reference_sentence_path ../data/train/totto_train_reference.txt\
    --dev_table_text_path ../data/dev/totto_dev_table.txt\
    --dev_content_text_path ../data/dev/totto_dev_content_plan.txt\
    --dev_reference_sentence_path ../data/dev/totto_dev_reference.txt\
    --dev_reference_path ../data/raw_data/totto_dev_data.jsonl\
    --special_token_path ../data/totto_col_header_vocab.txt\
    --ckpt_path ./ckpt/pretrain/