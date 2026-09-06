import os
import regex as re
import numpy as np
from regex_bpe_tokenizer import Tokenizer

input_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
temp_bin_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/train_tokens.tmp.bin"
output_npy_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/tiny_stories_train_tokens.npy"
dtype = np.uint16  # Use np.int32 if len(tok.vocab) > 65,535
flush_size = 1_000_000

tok = Tokenizer.from_files("vocab.json", "merges.pkl", special_tokens=["<|endoftext|>"])

# Stream lines into the generator, writing binary buffers
buffer = []
with open(input_path, "r", encoding="utf-8") as in_f, open(temp_bin_path, "wb") as out_f:
    # Generator expression passing the file lines to encode_iterable
    for token_id in tok.encode_iterable(in_f):
        buffer.append(token_id)
        if len(buffer) >= flush_size:
            out_f.write(np.array(buffer, dtype=dtype).tobytes())
            buffer.clear()

    if buffer:
        out_f.write(np.array(buffer, dtype=dtype).tobytes())
        buffer.clear()

# Derive token count from file size and wrap into .npy using memmap
total_bytes = os.path.getsize(temp_bin_path)
total_tokens = total_bytes // np.dtype(dtype).itemsize

# Copy chunks from raw binary into a valid .npy memmap file
raw_stream = np.memmap(temp_bin_path, dtype=dtype, mode="r", shape=(total_tokens,))
npy_stream = np.lib.format.open_memmap(
    output_npy_path, 
    mode="w+", 
    dtype=dtype, 
    shape=(total_tokens,)
)

for i in range(0, total_tokens, flush_size):
    npy_stream[i : i + flush_size] = raw_stream[i : i + flush_size]

npy_stream.flush()
del raw_stream
del npy_stream
os.remove(temp_bin_path)

print(f"Successfully created {output_npy_path} with {total_tokens:,} tokens.")


# Zero RAM overhead memory map
tokens = np.load("/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/tiny_stories_train_tokens.npy", mmap_mode="r")

print("Shape:", tokens.shape)        # Should be (total_tokens,)
print("Dtype:", tokens.dtype)        # Should be uint16
print("Sample tokens:", tokens[:10])

# Direct sequence chunking for LM training
context_size = 1024
x = tokens[0 : context_size]
y = tokens[1 : context_size + 1]

# Decode check using the decode method
print("Decoded text sample:", tok.decode(tokens[:30].tolist()))
