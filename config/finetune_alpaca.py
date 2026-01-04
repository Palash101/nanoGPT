"""
Fine-tune a GPT model on the Alpaca instruction-tuning dataset.
This config is designed for supervised fine-tuning (SFT) on instruction-following tasks.

This config is set up to fine-tune from a model trained with train_tinystories.py
(checkpoints saved in out-tinystories/).

Usage:
  # Fine-tune from out-tinystories checkpoint (default - will overwrite the checkpoint)
  python train.py config/finetune_alpaca.py
  
  # To preserve your original checkpoint, copy it first:
  # cp -r out-tinystories out-alpaca
  # Then change out_dir below to 'out-alpaca' or use --out_dir=out-alpaca
"""

import time

# Set out_dir to your pre-trained model directory
# The model will be loaded from out_dir/ckpt.pt and saved to the same location
out_dir = 'out-tinystories'  # Change this if you want to save to a different directory
eval_interval = 200
eval_iters = 200
wandb_log = False  # feel free to turn on
wandb_project = 'alpaca'
wandb_run_name = 'ft-alpaca-' + str(time.time())

dataset = 'alpaca'
# Load from the checkpoint in out_dir
init_from = 'resume'  # loads from out_dir/ckpt.pt

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# training hyperparameters for fine-tuning
# Alpaca has ~52k examples, formatted into instruction-following prompts
# With batch_size=1, grad_accum=32, block_size=1024: 32,768 tokens/iter
# Alpaca has ~10-20M tokens total, so 1 epoch ≈ 300-600 iters
batch_size = 1
gradient_accumulation_steps = 32
block_size = 1024  # context length
# NOTE: When resuming from a checkpoint, max_iters needs to be higher than the checkpoint's iter_num
# If your checkpoint is at iter 14000, set this to 14000 + desired_additional_iters
# For ~2000 more iterations of fine-tuning, set to 16000
max_iters = 16000  # will continue from checkpoint's iter_num until this value

# fine-tune at constant learning rate (typical for instruction tuning)
learning_rate = 2e-5  # lower LR for fine-tuning (2e-5 to 3e-5 is common)
decay_lr = False

# model settings (inherited from init_from model, but can override)
dropout = 0.1  # use some dropout for fine-tuning to prevent overfitting

# system settings
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', etc., or 'mps' on macbooks
dtype = 'float16'  # 'float32', 'bfloat16', or 'float16'
compile = False  # set to True if using PyTorch 2.0+ (can speed up training)

