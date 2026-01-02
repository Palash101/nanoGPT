# config for training GPT-2 Medium (350M) on TinyStories dataset
# GPT-2 Medium: n_layer=24, n_head=16, n_embd=1024
# Launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories_medium.py
# Or for single GPU:
# $ python train.py config/train_tinystories_medium.py

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'gpt2-medium-350M'

# dataset
dataset = 'tinystories'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# Adjust gradient_accumulation_steps based on number of GPUs
# For single GPU: gradient_accumulation_steps = 40 (12 * 40 = 480 effective batch size)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8  # adjust based on number of GPUs

# model config - GPT-2 Medium
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0  # for pretraining 0 is good
bias = False  # False is a bit better and faster

# training iterations
# TinyStories has ~2.14M stories, adjust max_iters based on your needs
max_iters = 200000  # adjust based on dataset size and training time
lr_decay_iters = 200000

# eval stuff
eval_interval = 2000
eval_iters = 200
log_interval = 10

# optimizer
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5  # learning_rate / 10

# system
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', etc., or 'mps' on macbooks
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16'. train.py will auto-detect bfloat16 support
compile = True  # use PyTorch 2.0 to compile the model to be faster

# checkpointing
out_dir = 'out-tinystories-medium'
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

