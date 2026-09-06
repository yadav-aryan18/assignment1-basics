from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from typing import BinaryIO, IO
from transformer_lm import TransformerLM
from transformer_lm_backward import AdamW, cosineAnnealingScheduler, crossEntropyLoss, gradient_clipping





def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    valid_range = np.arange(len(dataset) - context_length)
    starting_indices = np.random.choice(valid_range, batch_size)
    idx = starting_indices[:, None] + np.arange(context_length)

    input_tensor = torch.from_numpy(dataset[idx]).to(device=device)
    target_tensor = torch.from_numpy(dataset[idx + 1]).to(device=device)

    # Change to torch.int32 when using vocab-size > 32k
    input_tensor = input_tensor.to(torch.int32)
    target_tensor = target_tensor.to(torch.int64)

    input_tensor = input_tensor.view(batch_size, context_length)
    target_tensor = target_tensor.view(batch_size, context_length)

    return input_tensor, target_tensor



def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, start_epoch: int, path: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "start_epoch": start_epoch,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {start_epoch} at path: {path}")


def load_checkpoint(path: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["start_epoch"]
    print(f"Checkpoint loaded from path: {path}")
    return model, optimizer, start_epoch


def train(
        d_model: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        dataset_path: str | os.PathLike,
        save_path: str | os.PathLike | BinaryIO | IO[bytes],
        load_path: str | os.PathLike | BinaryIO | IO[bytes] | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        eps: float = 1e-8,
        max_learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-4,
        warmup_steps: int = 10,
        final_steps: int = 100,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Training on device: {device}  |  Device name: {torch.cuda.get_device_name(device)}")
        # Data loading
        dataset = np.load(dataset_path, mmap_mode="r")
        print(f"Dataset loaded from: {dataset_path}")

        # Initializing the model
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
        ).to(device=device)

        print("Model initialized...")

        # Initializing the optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )

        print("Optimizer initialized...")

        # Loading checkpoint (if one exists)
        if load_path is not None:
            model, optimizer, start_epoch = load_checkpoint(load_path, model, optimizer)
            print(f"Checkpoint loaded from: {load_path}")
            print(f"Resuming training from epoch: {start_epoch}")
        else:
            start_epoch = 0



        total_epochs: int = num_epochs

        print(f"Starting training from epoch: {start_epoch}")

        batch_count = 0

        for epoch in range(start_epoch, total_epochs):
            input_batch, targets = data_loading(dataset, batch_size, context_length, device)
            batch_count += 1

            # print(f"Batches loaded: {batch_count}")

            optimizer.zero_grad()

            # print("Gradients cleared")

            predictions = model(input_batch)

            # print("Predictions made")

            loss = crossEntropyLoss(predictions, targets)

            # print("Loss computed")

            loss.backward()

            # print("Gradients computed")

            gradient_clipping(model.parameters(), max_norm=max_grad_norm, eps=eps)

            # print("Gradients clipped")

            optimizer.step()

            # print("Optimizer step taken")

            lr_scheduler = cosineAnnealingScheduler(
                max_learning_rate=max_learning_rate,
                min_learning_rate=min_learning_rate,
                current_step=start_epoch,
                warmup_steps=warmup_steps,
                final_steps=final_steps,
            )

            # print("Learning rate scheduler stepped")

            optimizer.defaults['lr'] = lr_scheduler

            # print("Learning rate updated")

            if epoch % 20 == 0:
                save_checkpoint(model, optimizer, epoch, save_path)

            print(f"Epoch {epoch} loss: {loss.item()}")



if __name__ == "__main__":

    dataset_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/tiny_stories_train_tokens.npy"
    save_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/simple_model.pth"
    load_path = "/media/nightking/WD-SN570/deep_learning/stanford_cs336/assignment1-basics/cs336_basics/data/simple_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        d_model= 512,
        vocab_size = 10000,
        context_length = 256,
        num_layers = 4,
        num_heads = 16,
        d_ff = 1344,
        rope_theta = 10000,
        num_epochs = 1000,
        learning_rate = 1e-2,
        batch_size = 50,
        dataset_path = dataset_path,
        save_path = save_path,
        load_path = load_path,
        betas = (0.9, 0.95),
        weight_decay = 0.01,
        max_grad_norm = 1.0,
        eps = 1e-8,
        max_learning_rate = 1e-2,
        min_learning_rate = 1e-4,
        warmup_steps = 10,
        final_steps = 900,
        device = None,
        dtype = None,
    )
