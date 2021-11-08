# Plan-then-Generate: Controlled Data-to-Text Generation via Planning
Authors: Yixuan Su, David Vandyke, Sihui Wang, Yimai Fang, and Nigel Collier

Code for EMNLP 2021 paper [Plan-then-Generate: Controlled Data-to-Text Generation via Planning](https://arxiv.org/abs/2108.13740)

# 1. Environment Setup:
## (1) Hardware Requirement:
The code in this repo is thoroughly tested on our machine with a single Nvida V100 GPU (16GB)
## (2) Installation:
```yaml
chmod +x ./config_setup.sh
./config_setup.sh
```
# 2. ToTTo Data Preprocessing:
## Option (1): Preprocess the ToTTo data from scratch by yourself:
```yaml
cd ./data
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
Please refer to README.md in ./content_planner folder

# 4. Sequence Generator:
Please refer to README.md in ./generator folder

# 5. Citation
If you find our paper and resources useful, please kindly cite our paper:

```bibtex
@inproceedings{su-etal-2021-plan-generate,
    title = "Plan-then-Generate: Controlled Data-to-Text Generation via Planning",
    author = "Su, Yixuan  and
      Vandyke, David  and
      Wang, Sihui  and
      Fang, Yimai  and
      Collier, Nigel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.76",
    pages = "895--909",
    abstract = "Recent developments in neural networks have led to the advance in data-to-text generation. However, the lack of ability of neural models to control the structure of generated output can be limiting in certain real-world applications. In this study, we propose a novel Plan-then-Generate (PlanGen) framework to improve the controllability of neural data-to-text models. Extensive experiments and analyses are conducted on two benchmark datasets, ToTTo and WebNLG. The results show that our model is able to control both the intra-sentence and inter-sentence structure of the generated output. Furthermore, empirical comparisons against previous state-of-the-art methods show that our model improves the generation quality as well as the output diversity as judged by human and automatic evaluations.",
}
```
