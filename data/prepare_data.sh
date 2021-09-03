mkdir ./train
mkdir ./dev
mkdir ./test
wget https://storage.googleapis.com/totto-public/totto_data.zip
unzip totto_data.zip
mv totto_data raw_data
rm totto_data.zip
git clone https://github.com/google-research/language.git language_repo
cd language_repo
pip3 install -r language/totto/eval_requirements.txt
python3 -m language.totto.baseline_preprocessing.preprocess_data_main --input_path="../raw_data/totto_dev_data.jsonl" --output_path="../dev/processed_totto_dev_data.jsonl"
cd ..
python3 data_processing.py\
    --special_token_path ./totto_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./dev/processed_totto_dev_data.jsonl\
    --file_head_name ./dev/totto_dev\
    --dataset_mode dev
cd language_repo
python3 -m language.totto.baseline_preprocessing.preprocess_data_main --input_path="../raw_data/unlabeled_totto_test_data.jsonl" --output_path="../test/processed_unlabeled_totto_test_data.jsonl"
cd ..
python3 data_processing.py\
    --special_token_path ./totto_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./test/processed_unlabeled_totto_test_data.jsonl\
    --file_head_name ./test/totto_test\
    --dataset_mode test
cd language_repo
python3 -m language.totto.baseline_preprocessing.preprocess_data_main --input_path="../raw_data/totto_train_data.jsonl" --output_path="../train/processed_totto_train_data.jsonl"
cd ..
python3 data_processing.py\
    --special_token_path ./totto_col_header_vocab.txt\
    --special_token_min_cnt 10\
    --raw_data_path ./train/processed_totto_train_data.jsonl\
    --file_head_name ./train/totto_train\
    --dataset_mode train