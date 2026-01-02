"""
Prepare the TinyStories dataset for training.
Downloads from Hugging Face and tokenizes using GPT-2 BPE encoding.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load TinyStories dataset from Hugging Face
    # The dataset already has train and validation splits
    print("Loading TinyStories dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)
    
    # The dataset has 'train' and 'validation' splits
    # Rename 'validation' to 'val' to match the expected naming convention
    if 'validation' in dataset:
        dataset['val'] = dataset.pop('validation')
    
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train']):,}")
    print(f"Val samples: {len(dataset['val']):,}")

    # Define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    print("Tokenizing the dataset...")
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
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

