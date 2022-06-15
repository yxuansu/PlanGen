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
* `--train_src_path`: The path of the training table file.
* `--train_tgt_path`: The path of the training content plan file.
* `--dev_src_path`: The path of the validation table file.
* `--dev_tgt_path`: The path of the validation content plan file.
* `--max_src_len`: The maximum length of the table sequence.
* `--max_tgt_len`: The maximum length of the content plan sequence.
* `--special_token_path`: The file that contains the special tokens (i.e., slot keys) which we would like to add into the tokenizer and model.
* `--min_slot_key_cnt`: The minimum frequency of the special tokens (i.e., slot keys) exist in the training file.
* `--number_of_gpu`: The number of available GPUs.
* `--batch_size_per_gpu`: The batch size for each GPU.
* `--gradient_accumulation_steps`: How many forward computations between two gradient updates.
* `--effective_batch_size`: The overall batch size. It equals to batch_size_per_gpu x gradient_accumulation_steps x number_of_gpu.
* `--total_steps`: The number of total gradient update steps.
* `--print_every`: Have many steps to show the intermediate results.
* `--save_every`: How many steps to save one checkpoint.
* `--learning_rate`: The learning rate.
* `--mle_loss_weight`: The weight of the MLE loss in the training objective.
* `--save_path_prefix`: Where to save the checkpoints.
