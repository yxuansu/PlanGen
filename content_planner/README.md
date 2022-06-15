# Content Planner
In this repo, we provide a simpler and more robust implementation of our content planner. In the following, we show how to conduct training and inference with our content planner on the ToTTo dataset.

****

## Catalogue:
* <a href='#training'>1. Training of Content Planner</a>
    * <a href='#prepare_data'>1.1. Data Preparation</a>
    * <a href='#train_content_planner'>1.2. Training</a>
* <a href='#inference'>2. Inference with Content Planner</a>
    * <a href='#load_model'>2.1. Load Model</a>
    * <a href='#perform_inference'>2.2. Perform Inference</a>
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


****

<span id='inference'/>

### 2. Inference with Content Planner:
In the following, we show how to perform inference with the content planner.

<span id='load_model'/>

#### 2.1. Load Model:
To the load the pre-trained model, please run the following commands:
```python
from utlis import load_special_tokens
from contentplanner import ContentPlanner
# load the list of table slot keys
path = r'../data/totto_col_header_vocab.txt'
min_slot_cnt = 10
special_token_list = load_special_tokens(path, min_slot_cnt)
# create a model instance
model_name, special_token_list = 'bert-base-cased', special_token_list
model = ContentPlanner(model_name, special_token_list=special_token_list)
# load the pre-trained parameters
ckpt_path = r'./ckpt/' # the path specified in the --save_path_prefix argument of the training script
model.load_pretrained_model(ckpt_path)
model.eval()
```

**[Note]** The ckpt_path is the path specified in the `--save_path_prefix` argument of the training script. If you would like to have a quick test on the inference part, we also provide our pre-trained parameters for you to use. 

To download the pre-trained parameters, please run the following commands:
```yaml
chmod +x ./download_ckpt.sh
./download_ckpt.sh
```

<span id='perform_inference'/>

#### 2.2. Perform Inference:
After loading the model, we can then perform inference as:
```python
import torch
# example table
table = r'__page_title__ : List of Governors of South Carolina __EOS__ __#__ : 76 __EOS__ __Governor__ : Daniel Henry Chamberlain __EOS__ __Took_Office__ : December 1 , 1874 __EOS__ __section_title__ : Governors under the Constitution of 1868 __EOS__'

# prepare table id list
table_id_list = model.tokenizer.encode(table, max_length=320, truncation=True, add_special_tokens=False)[:320]
cls_token_id, sep_token_id = model.tokenizer.cls_token_id, model.tokenizer.sep_token_id
table_id_list = [cls_token_id] + table_id_list + [sep_token_id]
src_tensor = torch.LongTensor(table_id_list).view(1,-1)

# prepare selected content plan id list
selected_id_list = [model.targettokenizer.extract_selective_ids(table.strip('\n').strip())]
candidate_set = model.targettokenizer.convert_ids_to_text(selected_id_list[0])
print ('The candidate set is: {}\n'.format(candidate_set))
'''
   The candidate set is: __EOS__ __Governor__ __Took_Office__ __page_title__ __section_title__ __#__
'''
```

The function `extract_selective_ids()` extracts the candidate set of the content plan from the given table. The given table is represented as a list of `key : value` pairs. The final predicted content plan **only** contains the slot keys that exist in the candidate set.

```python
# make prediction
predicted_content_plan = model.selective_decoding(src_tensor, selected_id_list)
print ('The pedicted content plan is: {}'.format(predicted_content_plan))
'''
   The pedicted content plan is: ['__Governor__ __#__ __page_title__ __Took_Office__']
'''
```

After preparing the input ids of the table and the candidate set, we can then use the model to predict the content plan with the function `selective_decoding()`.


