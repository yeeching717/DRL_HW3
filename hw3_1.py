import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Gridworld import Gridworld


action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def encode_state(game):
    state = game.board.render_np().astype(np.float32)
    return state.reshape(1, -1)


def build_torch_model(input_dim, hidden1=150, hidden2=100):
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 4),
    )


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state1 = torch.cat([b[0] for b in batch])
        action = torch.tensor([b[1] for b in batch])
        reward = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        state2 = torch.cat([b[3] for b in batch])
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return state1, action, reward, state2, done

    def __len__(self):
        return len(self.buffer)


def moving_average(values, window=50):
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.array(values, dtype=np.float32), kernel, mode='valid')


def plot_training_stats(losses, episode_returns, wins, window=50, title_prefix="", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if losses:
        ax.plot(losses, label="Loss")

    if episode_returns:
        avg_returns = moving_average(episode_returns, window=window)
        ax.plot(avg_returns, label=f"Avg Return (window={window})")

    if wins:
        win_rate = moving_average(wins, window=window)
        ax.plot(win_rate, label=f"Win Rate (window={window})")

    ax.set_title(f"{title_prefix}Training Metrics")
    ax.set_xlabel("Steps / Episodes")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# HW3-1: Naive DQN for static mode

def train_naive_dqn_static(epochs=1000, gamma=0.9, epsilon=1.0, plot=False, window=50, return_stats=False,
                           save_path=None, size=4, obstacle_count=0):
    sample_game = Gridworld(size=size, mode='static', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    model = build_torch_model(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    episode_returns = []
    wins = []
    for _ in range(epochs):
        game = Gridworld(size=size, mode='static', obstacle_count=obstacle_count)
        state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 10.0).astype(np.float32))
        status = 1
        episode_return = 0
        while status == 1:
            qval = model(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(torch.argmax(qval).item())
            game.makeMove(action_set[action_idx])
            next_state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 10.0).astype(np.float32))
            reward = game.reward()
            episode_return += reward
            with torch.no_grad():
                max_next_q = torch.max(model(next_state))
            target = reward + gamma * max_next_q if reward == -1 else reward
            target = torch.tensor([target], dtype=torch.float32)
            pred = qval.squeeze()[action_idx]
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            state = next_state
            if reward != -1:
                wins.append(1 if reward > 0 else 0)
                episode_returns.append(episode_return)
                status = 0
        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)
    if plot:
        plot_training_stats(
            losses,
            episode_returns,
            wins,
            window=window,
            title_prefix="Naive DQN (static) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return model, losses, stats
    return model, losses


# HW3-1: DQN with experience replay for random mode

def train_replay_dqn_random(epochs=5000, gamma=0.9, epsilon=0.3,
                            batch_size=200, mem_size=1000, max_moves=50,
                            plot=False, window=50, return_stats=False,
                            save_path=None, size=4, obstacle_count=0):
    sample_game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    model = build_torch_model(input_dim)


def train_naive_dqn_static_budget(
    total_steps=1500,
    gamma=0.9,
    epsilon=1.0,
    max_moves=20,
    plot=False,
    window=50,
    return_stats=False,
    save_path=None,
    size=4,
    obstacle_count=0,
    hidden1=64,
    hidden2=64,
):
    sample_game = Gridworld(size=size, mode='static', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    model = build_torch_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    episode_returns = []
    wins = []
    steps = 0
    epsilon_decay = (epsilon - 0.1) / max(total_steps, 1)

    while steps < total_steps:
        game = Gridworld(size=size, mode='static', obstacle_count=obstacle_count)
        state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 10.0).astype(np.float32))
        episode_return = 0
        for _ in range(max_moves):
            if steps >= total_steps:
                break
            qval = model(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(torch.argmax(qval).item())
            game.makeMove(action_set[action_idx])
            next_state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 10.0).astype(np.float32))
            reward = game.reward()
            episode_return += reward
            with torch.no_grad():
                max_next_q = torch.max(model(next_state))
            target = reward + gamma * max_next_q if reward == -1 else reward
            target = torch.tensor([target], dtype=torch.float32)
            pred = qval.squeeze()[action_idx]
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            state = next_state
            steps += 1
            epsilon = max(0.1, epsilon - epsilon_decay)
            if reward != -1:
                wins.append(1 if reward > 0 else 0)
                episode_returns.append(episode_return)
                break

    if plot:
        plot_training_stats(
            losses,
            episode_returns,
            wins,
            window=window,
            title_prefix="Naive DQN (static) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return model, losses, stats
    return model, losses


def train_replay_dqn_random_budget(
    total_steps=2000,
    gamma=0.9,
    epsilon=0.3,
    batch_size=32,
    mem_size=200,
    max_moves=20,
    plot=False,
    window=50,
    return_stats=False,
    save_path=None,
    size=4,
    obstacle_count=0,
    hidden1=64,
    hidden2=64,
):
    sample_game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    model = build_torch_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer(mem_size)
    losses = []
    episode_returns = []
    wins = []
    steps = 0

    while steps < total_steps:
        game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
        state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 100.0).astype(np.float32))
        episode_return = 0
        for _ in range(max_moves):
            if steps >= total_steps:
                break
            qval = model(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(torch.argmax(qval).item())
            game.makeMove(action_set[action_idx])
            next_state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 100.0).astype(np.float32))
            reward = game.reward()
            episode_return += reward
            done = reward > 0
            replay.push(state, action_idx, reward, next_state, done)
            state = next_state
            if len(replay) > batch_size:
                s1, a, r, s2, d = replay.sample(batch_size)
                q1 = model(s1)
                with torch.no_grad():
                    q2 = model(s2)
                target = r + gamma * ((1 - d) * torch.max(q2, dim=1)[0])
                pred = q1.gather(dim=1, index=a.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(pred, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            steps += 1
            if reward != -1:
                wins.append(1 if reward > 0 else 0)
                episode_returns.append(episode_return)
                break

    if plot:
        plot_training_stats(
            losses,
            episode_returns,
            wins,
            window=window,
            title_prefix="Replay DQN (random) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return model, losses, stats
    return model, losses
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer(mem_size)
    losses = []
    episode_returns = []
    wins = []
    for _ in range(epochs):
        game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
        state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 100.0).astype(np.float32))
        status = 1
        mov = 0
        episode_return = 0
        while status == 1:
            mov += 1
            qval = model(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(torch.argmax(qval).item())
            game.makeMove(action_set[action_idx])
            next_state = torch.from_numpy((encode_state(game) + np.random.rand(1, input_dim) / 100.0).astype(np.float32))
            reward = game.reward()
            episode_return += reward
            done = reward > 0
            replay.push(state, action_idx, reward, next_state, done)
            state = next_state
            if len(replay) > batch_size:
                s1, a, r, s2, d = replay.sample(batch_size)
                q1 = model(s1)
                with torch.no_grad():
                    q2 = model(s2)
                target = r + gamma * ((1 - d) * torch.max(q2, dim=1)[0])
                pred = q1.gather(dim=1, index=a.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(pred, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if reward != -1 or mov > max_moves:
                wins.append(1 if reward > 0 else 0)
                episode_returns.append(episode_return)
                status = 0
                mov = 0
    if plot:
        plot_training_stats(
            losses,
            episode_returns,
            wins,
            window=window,
            title_prefix="Replay DQN (random) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return model, losses, stats
    return model, losses
