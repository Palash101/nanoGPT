"""
Prepare the Alpaca dataset for supervised fine-tuning.
Downloads from Hugging Face (tatsu-lab/alpaca) and tokenizes using GPT-2 BPE encoding.
Formats instructions in a prompt-response format for instruction tuning.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
num_proc = 8

enc = tiktoken.get_encoding("gpt2")

def format_alpaca_prompt(example):
    """
    Format Alpaca example into instruction-following prompt.
    Format: ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}
    """
    instruction = example['instruction']
    input_text = example['input'] if example['input'] else ""
    output = example['output']
    
    if input_text:
        # Format with input field
        formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        # Format without input field (when input is empty)
        formatted = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return formatted

if __name__ == '__main__':
    # Load Alpaca dataset from Hugging Face
    print("Loading Alpaca dataset from Hugging Face (tatsu-lab/alpaca)...")
    dataset = load_dataset("tatsu-lab/alpaca", num_proc=num_proc)
    
    # Alpaca dataset only has a 'train' split, so we need to create a validation split
    if 'train' in dataset:
        print(f"Train samples: {len(dataset['train']):,}")
        # Create train/val split (90/10)
        split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')  # rename test to val
    else:
        split_dataset = dataset
    
    print(f"Dataset splits: {list(split_dataset.keys())}")
    print(f"Train samples: {len(split_dataset['train']):,}")
    print(f"Val samples: {len(split_dataset['val']):,}")
    
    # Format the examples into instruction-following prompts
    print("Formatting examples into instruction-following prompts...")
    def process(example):
        # Format the prompt
        text = format_alpaca_prompt(example)
        # Tokenize
        ids = enc.encode_ordinary(text)  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # Tokenize the dataset
    print("Tokenizing the dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['instruction', 'input', 'output'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # Concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Use a reasonable number of batches based on dataset size
        total_batches = min(1024, len(dset))
        if total_batches == 0:
            total_batches = 1
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"{split}.bin has {arr_len:,} tokens")

