from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import torch
from torch import nn, optim

Checkpoint = Mapping[str, Any]
CheckpointInput = Union[str, Path, Checkpoint]


def save_checkpoint(state: Checkpoint, filename: str = "my_checkpoint.pth.tar") -> None:
    """
    Persist a checkpoint dictionary to disk.

    Args:
        state: Typically contains at least ``state_dict`` and optionally ``optimizer``.
        filename: Target path for the serialized checkpoint.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"=> Saving checkpoint to {path}")
    torch.save(dict(state), path)


def load_checkpoint(
    checkpoint: CheckpointInput,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> MutableMapping[str, Any]:
    """
    Load model (and optionally optimizer) state from a checkpoint.

    Args:
        checkpoint: Path to the checkpoint file or an in-memory dictionary.
        model: Model receiving the ``state_dict``.
        optimizer: Optional optimizer to restore.
        map_location: Device remapping used by ``torch.load``.

    Returns:
        The loaded checkpoint dictionary for additional metadata consumption.
    """
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location=map_location)
    else:
        payload = dict(checkpoint)

    print("=> Loading checkpoint")
    model.load_state_dict(payload["state_dict"])

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    return payload
