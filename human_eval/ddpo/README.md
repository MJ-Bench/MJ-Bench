# Finetuning Diffusion Models

1. `ddpo_fintune.py` is used for finetuning diffusion models with real-valued reward models, i.e., the reward model only takes one image (and maybe prompt as well) as input and return the scalar real-valued reward. The reward model should use the following API:

```python
def default_reward_fn(images, prompts, metadata):
    # images: batch_size x C x H x W 
    # do something and return the rewards
    # you can ignore the prompts or metadata and only use the images.
```

2. `dpo_finetune.py` is copied from the diffuser lib, and it currently only supports finetuning using offline dataset, i.e., you should first generate pairwise images given the prompt, and then rank them using the reward model, e.g., LLava. 
