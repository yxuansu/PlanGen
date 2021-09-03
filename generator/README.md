# 1. To test whether the environment is installed correctly:
```yaml
chmod +x ./test_pretrain.sh
./test_pretrain.sh
```
# 2. Pre-train the generator:
```yaml
chmod +x ./pretrain.sh
./pretrain.sh
```
# 3. After pre-training, then finetune the generator with the RL objective:
```yaml
chmod +x ./finetune.sh
./finetune.sh
```
# 4. Perform inference with pre-trained checkpoints, you can downloaded the checkpoints [here](https://drive.google.com/file/d/1C0UVXemo4G14tXrxN_tomqpxJlfGUppl/view?usp=sharing)
```yaml
unzip the downloaded ckpt.zip and replace it with the empty ./ckpt folder

chmod +x ./perform_inference.sh
./perform_inference.sh
```
# 5. Controlled Generation:
We provide a jupyter notebook which illustrates how to perform controlled generation (by varying the content plan) with our model. 
## (1) First, download the pre-trained checkpoints as described in the above section. 
## (2) Then, install jupyter notebook in your local machine by
```yaml
pip install jupyter notebook
```
## (3) Last, open the provided notebook (Controlled Generation.ipynb), and have fun!
