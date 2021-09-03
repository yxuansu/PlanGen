# 1. To test whether the environment is installed correctly:
```yaml
chmod +x ./test_pretrain.sh
./test_pretrain.sh
```
# 2. Pretrain the generator:
```yaml
chmod +x ./pretrain.sh
./pretrain.sh
```
# 3. After pretraining, then finetune the generator with the RL objective:
```yaml
chmod +x ./finetune.sh
./finetune.sh
```
# 4. Perform inference with pretrained checkpoints, you can downloaded the checkpoints [here](https://drive.google.com/file/d/1C0UVXemo4G14tXrxN_tomqpxJlfGUppl/view?usp=sharing)
```yaml
chmod +x ./perform_inference.sh
./perform_inference.sh
```
