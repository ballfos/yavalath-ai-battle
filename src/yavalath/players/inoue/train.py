from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from yavalath.core.board import Board, CellState, PutResult
from yavalath.players.inoue.dqn import (
    DQN,
    ActionSpace,
    build_action_space,
    encode_state,
    iter_legal_indices,
    legal_action_mask,
    mask_q_values,
)


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_legal_masks: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ):
        self.buffer.append((state, action, reward, next_state, done, next_legal_mask))

    def sample(self, batch_size: int) -> ReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return ReplayBatch(
            states=np.stack(states),
            actions=np.array(actions, dtype=np.int64),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.stack(next_states),
            dones=np.array(dones, dtype=np.float32),
            next_legal_masks=np.stack(masks),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(
    model: nn.Module,
    state: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        legal_indices = list(iter_legal_indices(legal_mask))
        return int(random.choice(legal_indices))

    with torch.no_grad():
        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = model(state_t)
        q_values = mask_q_values(q_values, legal_mask)
        return int(torch.argmax(q_values, dim=1).item())


def compute_td_loss(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    gamma: float,
    device: torch.device,
    zero_sum: bool,
    max_grad_norm: float | None,
) -> float:
    states_t = torch.from_numpy(batch.states).to(device)
    actions_t = torch.from_numpy(batch.actions).to(device).unsqueeze(1)
    rewards_t = torch.from_numpy(batch.rewards).to(device)
    next_states_t = torch.from_numpy(batch.next_states).to(device)
    dones_t = torch.from_numpy(batch.dones).to(device)
    next_masks_t = torch.from_numpy(batch.next_legal_masks).to(device)

    q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states_t)
        next_q[~next_masks_t] = -1e9
        max_next_q = next_q.max(dim=1).values
        if zero_sum:
            max_next_q = -max_next_q
        targets = rewards_t + gamma * max_next_q * (1.0 - dones_t)

    loss = nn.SmoothL1Loss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())


def train_dqn(
    radius: int,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    target_update: int,
    buffer_capacity: int,
    min_buffer_size: int,
    learning_rate: float,
    model_path: Path,
    seed: int | None,
    reward_win: float,
    reward_lose: float,
    reward_draw: float,
    reward_step: float,
    reward_opponent_win: float,
    zero_sum: bool,
    reward_scale: float,
    max_grad_norm: float | None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_space: ActionSpace = build_action_space(radius)
    board_size = 2 * radius + 1

    policy_net = DQN(board_size=board_size, action_dim=action_space.size).to(device)
    target_net = DQN(board_size=board_size, action_dim=action_space.size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    global_step = 0
    pbar = tqdm(range(1, num_episodes + 1), desc="Training")

    for episode in pbar:
        board = Board(radius)
        turn = CellState.PLAYER1
        done = False
        episode_loss = 0.0
        updates = 0

        while not done:
            state = board.to_numpy(turn)
            legal_mask = legal_action_mask(board, action_space)
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay**episode))

            action_idx = select_action(policy_net, state, legal_mask, epsilon, device)
            position = action_space.positions[action_idx]

            result = board.put(position, turn)

            reward = reward_step
            if result == PutResult.WIN:
                reward = reward_win
                done = True
            elif result == PutResult.LOSE:
                reward = reward_lose
                done = True
            elif not board.get_empty_cells():
                reward = reward_draw
                done = True
            else:
                opponent = (
                    CellState.PLAYER2
                    if turn == CellState.PLAYER1
                    else CellState.PLAYER1
                )
                if _opponent_can_win_next(board, opponent):
                    reward = reward_opponent_win

            reward *= reward_scale

            next_turn = (
                CellState.PLAYER2 if turn == CellState.PLAYER1 else CellState.PLAYER1
            )
            next_state = board.to_numpy(next_turn)
            next_legal_mask = legal_action_mask(board, action_space)

            replay_buffer.push(
                state, action_idx, reward, next_state, done, next_legal_mask
            )

            if len(replay_buffer) >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_td_loss(
                    policy_net,
                    target_net,
                    optimizer,
                    batch,
                    gamma,
                    device,
                    zero_sum,
                    max_grad_norm,
                )
                episode_loss += loss
                updates += 1

            if done:
                break

            turn = next_turn
            global_step += 1

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = episode_loss / updates if updates > 0 else 0.0
        pbar.set_postfix(
            epsilon=f"{epsilon:.3f}",
            loss=f"{avg_loss:.4f}",
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), model_path)
    print(f"Saved model to: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN for Yavalath")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--target-update", type=int, default=20)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--model-path", type=Path, default=Path(__file__).with_name("model.pt")
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reward-win", type=float, default=200.0)
    parser.add_argument("--reward-lose", type=float, default=-100.0)
    parser.add_argument("--reward-draw", type=float, default=1.0)
    parser.add_argument("--reward-step", type=float, default=-30.0)
    parser.add_argument("--reward-opponent-win", type=float, default=-100.0)
    parser.add_argument("--zero-sum", action="store_true", default=True)
    parser.add_argument("--no-zero-sum", dest="zero_sum", action="store_false")
    parser.add_argument("--reward-scale", type=float, default=0.01)

    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_dqn(
        radius=args.radius,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer_size,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        seed=args.seed,
        reward_win=args.reward_win,
        reward_lose=args.reward_lose,
        reward_draw=args.reward_draw,
        reward_step=args.reward_step,
        reward_opponent_win=args.reward_opponent_win,
        zero_sum=args.zero_sum,
        reward_scale=args.reward_scale,
        max_grad_norm=args.max_grad_norm,
    )


def _opponent_can_win_next(board: Board, opponent: CellState) -> bool:
    for pos in board.get_empty_cells():
        result = board.put(pos, opponent)
        board.pick(pos)
        if result == PutResult.WIN:
            return True
    return False


if __name__ == "__main__":
    main()
