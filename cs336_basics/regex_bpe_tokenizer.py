from __future__ import annotations

import sys
from pathlib import Path

# Adds the 'assignment1-basics' directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import regex as re
from itertools import repeat
import multiprocessing as mp
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries
from typing import Iterator
from copy import deepcopy



PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenization(
                    file_path: str, 
                    special_tokens: list[str],
                    begin: int, 
                    terminate: int
                ) -> dict[tuple[bytes, ...], int]:
    
    with open(file_path, 'rb') as f:
        f.seek(begin)
        chunk = f.read(terminate - begin).decode("utf-8", errors="ignore")
        count: dict[tuple[bytes, ...], int] = {}

        # special_token = "<|endoftext|>"
        special_pattern = "|".join([re.escape(token) for token in special_tokens])
        chunk = re.split(special_pattern, chunk)

        for chunk_split in chunk:
            for m in re.finditer(PATTERN, chunk_split):
                bytes_tuple = tuple(bytes([ch]) for ch in m.group(0).encode('utf-8'))
                count[bytes_tuple] = count.get(bytes_tuple, 0) + 1
    return count



def mp_regex(
            file_path: str, 
            special_tokens: list[str],
            start: list[int] , 
            end: list[int], 
            num_workers: int | None = os.cpu_count()
        ) -> dict[tuple[bytes, ...], int]:

    ctx = mp.get_context("fork")
    
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        results: Iterator[dict[tuple[bytes, ...], int]] = executor.map(pre_tokenization, repeat(file_path), repeat(special_tokens), start, end)
        
    pre_token_counts: Counter[tuple[bytes, ...]] = Counter()
    for worker_dict in results:
        pre_token_counts.update(worker_dict)
    return dict(pre_token_counts)






def find_pairs(pre_token_count: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    
    pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    reverse_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for tpl, appear in pre_token_count.items():
        for i in range(len(tpl)-1):
            pairs[(tpl[i], tpl[i+1])] += appear
            reverse_pair[(tpl[i], tpl[i+1])].add(tpl)
    return pairs, reverse_pair




def pair_diff(
            word: tuple[bytes, ...],
            pre_token_count: dict[tuple[bytes, ...], int]
        ) -> dict[tuple[bytes, bytes], int]:
    pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for i in range(len(word)-1):
        pair_dict[(word[i], word[i+1])] += pre_token_count[word]
    return pair_dict

def remove_word_from_pair(
                        word: tuple[bytes, ...], 
                        pair_dict: dict[tuple[bytes, bytes], int], 
                        reverse_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
                        ) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    for p in pair_dict:
        reverse_pair[p].remove(word)
    return reverse_pair

def add_word_to_pair(
                    word: tuple[bytes, ...], 
                    pair_dict: dict[tuple[bytes, bytes], int], 
                    reverse_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
                    ) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    for p in pair_dict:
        reverse_pair[p].add(word)
    return reverse_pair


def global_delta_pair(
                    old_pair_dict: dict[tuple[bytes, bytes], int], 
                    new_pair_dict: dict[tuple[bytes, bytes], int], 
                    global_pair_dict: dict[tuple[bytes, bytes], int],
                    reverse_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
                    ) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    
    delta_pair: dict[tuple[bytes, bytes], int] = {}
    delta_pair.update(old_pair_dict)
    delta_pair.update(new_pair_dict)
    for dp in delta_pair:
        delta_pair[dp] = old_pair_dict.get(dp, 0) - new_pair_dict.get(dp, 0)
        if global_pair_dict.get(dp, 0) - delta_pair.get(dp, 0) == 0:
            del global_pair_dict[dp]
            del reverse_pair[dp]
        elif global_pair_dict.get(dp, 0) - delta_pair.get(dp, 0) > 0:
            global_pair_dict[dp] = global_pair_dict.get(dp, 0) - delta_pair.get(dp, 0)
    return reverse_pair





def merge(
        pre_token_count: dict[tuple[bytes, ...], int], 
        pair: tuple[bytes, bytes], 
        reverse_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
        pairs: dict[tuple[bytes, bytes], int],
        verbose: bool = False
    ) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    
    total_merges: int = 0
    words: set[tuple[bytes, ...]] = deepcopy(reverse_pair[pair])

    for word in words:
        i = 0
        n = len(word)
        end_flag: bool = False
        merge_happen: bool = False
        merged_word: list[bytes] = []
        appear: int = pre_token_count[word]

        old_pair_dict = pair_diff(word, pre_token_count)
        reverse_pair = remove_word_from_pair(word, old_pair_dict, reverse_pair)

        while i < n-1:
            if (pair[0], pair[1]) == (word[i], word[i+1]):
                merged_word.append(pair[0]+pair[1])
                end_flag = True if i == n-2 else False
                merge_happen = True
                i += 2
                total_merges += 1
                continue
            merged_word.append(word[i])
            i += 1

        if not end_flag:
            merged_word.append(word[-1])

        new_tpl: tuple[bytes, ...] = tuple(merged_word)

        del pre_token_count[word]
        pre_token_count[new_tpl] = appear
        
        new_pair_dict = pair_diff(new_tpl, pre_token_count)
        reverse_pair = add_word_to_pair(new_tpl, new_pair_dict, reverse_pair)

        reverse_pair = global_delta_pair(old_pair_dict, new_pair_dict, global_pair_dict=pairs, reverse_pair=reverse_pair)

        if verbose and merge_happen:
            print(f"Merge Successful with {pair=} resulting in new string {new_tpl=}")

    return pre_token_count, reverse_pair






def train(
        input_path: str,
        vocab_size: int, 
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    i2b_vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merge_iters: int = vocab_size - (len(i2b_vocab) + len(special_tokens))        
    
    merge_order: list[tuple[bytes, bytes]] = []

    # Chunking file / finding chunk boundaries
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    start = boundaries[:-1]
    end = boundaries[1:]
    
    # print(f"{start}\n{end}")

    # Parallelilzing pre-tokenization using lazy RegEx (re.finditer)
    pre_token_count: dict[tuple[bytes, ...], int] = mp_regex(
                                                            file_path=input_path, 
                                                            special_tokens=special_tokens, 
                                                            start=start, 
                                                            end=end, 
                                                            num_workers=num_processes
                                                            )
    # print(len(pre_token_count))
    
    pairs, reverse_pair = find_pairs(pre_token_count)

    # Merge Steps
    for _ in range(merge_iters):
        max_pair: tuple[bytes, bytes] = max(pairs, key=lambda k: (pairs[k], k))

        v_idx: int = max(i2b_vocab) + 1
        b_string: bytes = max_pair[0] + max_pair[1]

        merge_order.append(max_pair)

        i2b_vocab[v_idx] = b_string
        # b2i_vocab[b_string] = v_idx
        
        pre_token_count, reverse_pair = merge(pre_token_count, pair=max_pair, reverse_pair=reverse_pair, pairs=pairs)

    # Appending special_tokens to the vocabulary
    for token in special_tokens:
        i2b_vocab[max(i2b_vocab)+1] = token.encode('utf-8')

    return i2b_vocab, merge_order
        


# Example Usage
if __name__ == "__main__":
    import pstats
    from cProfile import Profile

    train_data = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/tests/fixtures/corpus.en"

    with Profile() as prof:
        i2b_vocab, merge_order = train(input_path=train_data, vocab_size=500, special_tokens=["<|endoftext|>"])
    prof.dump_stats("train.prof")

    stats = pstats.Stats("train.prof")
    stats.sort_stats("cumtime").print_stats(15)      # top 15 by cumulative time
    stats.print_callers("find_pairs")                # who calls into find_pairs

    print(f"{'Length of Vocab':<50}: {len(i2b_vocab)}\n")
    # print("-"*54)
    # for key, value in i2b_vocab.items():
    #     print(f"{'Index':<10}: {key:>5} {'|':^5} {'Byte_String':<12}: {repr(value):>15}")

