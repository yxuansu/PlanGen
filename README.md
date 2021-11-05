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
@article{DBLP:journals/corr/abs-2108-13740,
  author    = {Yixuan Su and
               David Vandyke and
               Sihui Wang and
               Yimai Fang and
               Nigel Collier},
  title     = {Plan-then-Generate: Controlled Data-to-Text Generation via Planning},
  journal   = {CoRR},
  volume    = {abs/2108.13740},
  year      = {2021},
  url       = {https://arxiv.org/abs/2108.13740},
  eprinttype = {arXiv},
  eprint    = {2108.13740},
  timestamp = {Fri, 03 Sep 2021 10:51:17 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2108-13740.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
