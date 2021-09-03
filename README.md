# Plan-then-Generate: Controlled Data-to-Text Generation via Planning
Authors: Yixuan Su, David Vandyke, Sihui Wang, Yimai Fang, and Nigel Collier

Code for EMNLP 2021 paper [Plan-then-Generate: Controlled Data-to-Text Generation via Planning](https://arxiv.org/abs/2108.13740)

# 1. Environment Setup:
```yaml
chmod +x ./config_setup.sh
./config_setup.sh
```

# 2. ToTTo Data Preprocessing:
## Option (1): Preprocess the ToTTo data from scratch by yourself:
```yaml
```
```yaml
cd data
chmod +x ./prepare_data.sh
./prepare_data.sh
```
This process could take up to 1 hour

## Option (2): Download the our processed data [here](https://drive.google.com/file/d/1YBGwo0atBmaCOhlu0v0yz21ixNALwF8v/view?usp=sharing)
```yaml
unzip data.zip and replace with the empty ./data folder
```
For more details about ToTTo dataset, please refer to the original Google Research [repo](https://github.com/google-research-datasets/ToTTo)

# 3. Content Planner:
Please refer to README.md in ./content_planner

# 4. Sequence Generator:
Please refer to README.md in ./generator

# 5. Citation
If you find our paper and resources useful, please kindly cite our paper:
```yaml
@inproceedings{su2021plangen,
    title={Plan-then-Generate: Controlled Data-to-Text Generation via Planning}, 
     author={Yixuan Su and David Vandyke and Sihui Wang and Yimai Fang and Nigel Collier},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```
