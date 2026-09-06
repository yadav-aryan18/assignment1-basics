from __future__ import annotations


import torch
import torch.nn as nn
from transformer_lm import TransformerLM
from regex_bpe_tokenizer import Tokenizer

import numpy as np



def TemperatureScaledSoftmax(logits: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
    y, _ = logits.max(dim=dim, keepdim=True)
    logits = logits - y
    logits = torch.exp(logits / temperature)
    sumX_dims = logits.sum(dim=dim, keepdim=True)
    logits = logits / sumX_dims
    return logits

def top_p_sampling(logits: torch.Tensor, top_p: float, dim: int = -1) -> int:
    current_row = logits[-1, :]
    sorted_probs, sorted_indices = torch.sort(current_row, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = (cumulative_probs > top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    sample_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[sample_sorted_idx].item()
    return int(next_token)


def TransformerLMDecoder(
        d_model: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_tokens: int = 256,
        top_p: float | None = None,
        temperature: float = 1.0,
        load_path: str | None = None,
    ) -> None:


        if load_path is None:
            raise ValueError("load_path must be provided")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}  |  Device name: {torch.cuda.get_device_name(device)}")

        model = TransformerLM(
            d_model=d_model,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        ).to(device)

        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model"])

        input_text = input("Enter text: ")

        tokenizer = Tokenizer.from_files("vocab.json", "merges.pkl")

        token_indices = tokenizer.encode(input_text)

        with torch.no_grad():
            while len(token_indices) <= max_tokens:
                token_idx_tensor = torch.tensor(token_indices, device=device)
                logits = model(token_idx_tensor)
                scaled_probs = TemperatureScaledSoftmax(logits, temperature)

                if top_p is not None:
                    next_token = top_p_sampling(scaled_probs, top_p)
                else:
                    # print(f"Dimensions: {scaled_probs.shape}")
                    next_token = int(torch.multinomial(scaled_probs[-1, :], num_samples=1).item())

                if next_token == tokenizer.reverse_vocab[b"<|endoftext|>"]:
                    break
                token_indices.append(next_token)

        output_text = tokenizer.decode(token_indices)

        print("---" * 50)
        print(f"User: {input_text}")
        print(f"LLM: {output_text}")


if __name__ == "__main__":
    load_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/simple_model.pth"

    model = TransformerLMDecoder(
        d_model=512,
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        load_path=load_path,
        temperature=0.5,
        top_p=0.9,
    )
