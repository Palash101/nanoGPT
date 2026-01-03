# config for training GPT-2 (124M) on TinyStories dataset
# GPT-2 Small: n_layer=12, n_head=12, n_embd=768
# Launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories.py
# Or for single GPU:
# $ python train.py config/train_tinystories.py

wandb_log = True
wandb_project = 'tinystories'
wandb_run_name = 'gpt2-124M'

# dataset
dataset = 'tinystories'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# Adjust gradient_accumulation_steps based on number of GPUs
# For single GPU: gradient_accumulation_steps = 40 (12 * 40 = 480 effective batch size)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40  # adjusted for single GPU/CPU training

# model config - GPT-2 Small (124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good
bias = False  # False is a bit better and faster

# training iterations
# Calculation: tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
#              = 40 * 12 * 1024 = 491,520 tokens per iteration
# TinyStories has ~1-2B tokens total
# 1 epoch ≈ 1.5B tokens / 491,520 ≈ 3,050 iterations
# Recommended: 3-5 epochs for good convergence
# 3 epochs ≈ 9,150 iterations, 5 epochs ≈ 15,250 iterations
max_iters = 15000  # ~5 epochs (adjust based on your needs: 9000 for 3 epochs, 15000 for 5 epochs)
lr_decay_iters = 15000

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
# T4 GPUs don't support bfloat16 natively, use float16 instead
# float16 will automatically use GradScaler for safe training
dtype = 'float16'  # 'float32', 'bfloat16', or 'float16'. Use float16 for T4 GPUs
compile = True  # use PyTorch 2.0 to compile the model to be faster

# checkpointing
out_dir = 'out-tinystories'
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

