from llava.train.train import train

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
