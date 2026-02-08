from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from yavalath.core.board import Board, CellState


@dataclass(frozen=True)
class ActionSpace:
    positions: list[tuple[int, int, int]]
    pos_to_index: dict[tuple[int, int, int], int]

    @property
    def size(self) -> int:
        return len(self.positions)


def build_action_space(radius: int) -> ActionSpace:
    board = Board(radius)
    positions = sorted(board.board.keys())
    return ActionSpace(
        positions=positions, pos_to_index={p: i for i, p in enumerate(positions)}
    )


def legal_action_mask(board: Board, action_space: ActionSpace) -> np.ndarray:
    mask = np.zeros(action_space.size, dtype=bool)
    for pos in board.get_empty_cells():
        idx = action_space.pos_to_index[pos]
        mask[idx] = True
    return mask


def encode_state(board: Board, player: CellState, device: torch.device) -> torch.Tensor:
    array = board.to_numpy(player)
    return torch.from_numpy(array).unsqueeze(0).to(device)


def mask_q_values(q_values: torch.Tensor, legal_mask: np.ndarray) -> torch.Tensor:
    mask = torch.from_numpy(legal_mask).to(q_values.device)
    masked = q_values.clone()
    masked[:, ~mask] = -1e9
    return masked


def select_greedy_action(
    model: nn.Module, state: torch.Tensor, legal_mask: np.ndarray
) -> int:
    with torch.no_grad():
        q_values = model(state)
        q_values = mask_q_values(q_values, legal_mask)
        return int(torch.argmax(q_values, dim=1).item())


class DQN(nn.Module):
    def __init__(self, board_size: int, action_dim: int):
        super().__init__()
        input_dim = 2 * board_size * board_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def iter_legal_indices(legal_mask: np.ndarray) -> Iterable[int]:
    return np.flatnonzero(legal_mask)
