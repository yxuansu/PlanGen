## Content Planner
In this repo, we provide a simpler and more robust implementation of our content planner. In the following, we show how to conduct training and inference with our content planner on the ToTTo dataset.

****

## Catalogue:
* <a href='#training'>1. Training of Content Planner</a>
    * <a href='#prepare_data'>1.1. Data Preparation</a>
    * <a href='#train_content_planner'>1.2. Training</a>

****

<span id='training'/>

### 1. Training of Content Planner:

<span id='prepare_data'/>

#### 1.1. Data Preparation:
To prepare the ToTTo dataset, please follow the instruction as described [[here]](https://github.com/yxuansu/PlanGen#2-totto-data-preprocessing). Also, please make sure the environment configuration is properly installed as described [[here]](https://github.com/yxuansu/PlanGen#2-installation).


<span id='train_content_planner'/>

#### 1.2. Training:
To train the content planner, please run the following commands:
```yaml
chmod +x ./train.sh
./train.sh
```
The arguments are as follows:
* `--model_name`: The name of huggingface pre-trained model.
* `--crf_low_rank`: The low rank configuration of the CRF layer. For more details, please refer to the Section 3.1 of our another paper (https://aclanthology.org/2021.eacl-main.18.pdf).
* `--crf_beam_size`: The beam width of the CRF layer. For more details, please refer to the Section 3.1 of our another paper (https://aclanthology.org/2021.eacl-main.18.pdf).


* `--train_path`: The file path of training set.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--margin`: The contrastive margin $\rho$.
* `--max_len`: The maximum length of training samples.
* `--number_of_gpu`: The number of available GPUs.
* `--batch_size_per_gpu`: The batch size for each GPU.
* `--gradient_accumulation_steps`: How many forward computations between two gradient updates.
* `--effective_batch_size`: The overall batch size. It equals to batch_size_per_gpu x gradient_accumulation_steps x number_of_gpu.
* `--total_steps`: The number of total gradient update steps.
* `--print_every`: Have many steps to show the intermediate results.
* `--save_every`: How many steps to save one checkpoint.
* `--learning_rate`: The learning rate.
* `--save_path_prefix`: Where to save the checkpoints.


   
     64\
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

