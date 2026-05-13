import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from Gridworld import Gridworld


action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def encode_state(game):
    state = game.board.render_np().astype(np.float32)
    return state.reshape(1, -1)


class KerasReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s1 = np.vstack([b[0] for b in batch])
        a = np.array([b[1] for b in batch], dtype=np.int32)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.vstack([b[3] for b in batch])
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return s1, a, r, s2, d

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


def build_keras_model(input_dim, hidden1=150, hidden2=100):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden1, activation='relu'),
        tf.keras.layers.Dense(hidden2, activation='relu'),
        tf.keras.layers.Dense(4),
    ])


# HW3-3: Keras DQN for random mode (with training tips)

def train_keras_dqn_random(epochs=5000, gamma=0.9, epsilon=0.3,
                            batch_size=200, mem_size=1000, max_moves=50,
                            sync_freq=500, plot=False, window=50, return_stats=False,
                            save_path=None, size=4, obstacle_count=0):
    sample_game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    policy_net = build_keras_model(input_dim)
    target_net = build_keras_model(input_dim)
    target_net.set_weights(policy_net.get_weights())
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=2000,
        decay_rate=0.95,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()
    replay = KerasReplayBuffer(mem_size)
    losses = []
    episode_returns = []
    wins = []
    step = 0
    for _ in range(epochs):
        game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
        state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
        status = 1
        mov = 0
        episode_return = 0
        while status == 1:
            step += 1
            mov += 1
            qval = policy_net(state, training=False).numpy()
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(np.argmax(qval))
            game.makeMove(action_set[action_idx])
            next_state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
            reward = game.reward()
            episode_return += reward
            done = reward > 0
            replay.push(state, action_idx, reward, next_state, done)
            state = next_state
            if len(replay) > batch_size:
                s1, a, r, s2, d = replay.sample(batch_size)
                with tf.GradientTape() as tape:
                    q1 = policy_net(s1, training=True)
                    q2 = target_net(s2, training=False)
                    target = r + gamma * ((1 - d) * tf.reduce_max(q2, axis=1))
                    idx = tf.stack([tf.range(batch_size), a], axis=1)
                    pred = tf.gather_nd(q1, idx)
                    loss = loss_fn(target, pred)
                grads = tape.gradient(loss, policy_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
                losses.append(float(loss.numpy()))
                if step % sync_freq == 0:
                    target_net.set_weights(policy_net.get_weights())
            if reward != -1 or mov > max_moves:
                wins.append(1 if reward > 0 else 0)
                episode_returns.append(episode_return)
                status = 0
                mov = 0
        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)
    if plot:
        plot_training_stats(
            losses,
            episode_returns,
            wins,
            window=window,
            title_prefix="Keras DQN (random) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return policy_net, losses, stats
    return policy_net, losses


def train_keras_dqn_random_budget(
    total_steps=2000,
    gamma=0.9,
    epsilon=0.3,
    batch_size=32,
    mem_size=200,
    max_moves=20,
    sync_freq=200,
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
    policy_net = build_keras_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    target_net = build_keras_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    target_net.set_weights(policy_net.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()
    replay = KerasReplayBuffer(mem_size)
    losses = []
    episode_returns = []
    wins = []
    steps = 0

    while steps < total_steps:
        game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
        state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
        episode_return = 0
        for _ in range(max_moves):
            if steps >= total_steps:
                break
            qval = policy_net(state, training=False).numpy()
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(np.argmax(qval))
            game.makeMove(action_set[action_idx])
            next_state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
            reward = game.reward()
            episode_return += reward
            done = reward > 0
            replay.push(state, action_idx, reward, next_state, done)
            state = next_state
            if len(replay) > batch_size:
                s1, a, r, s2, d = replay.sample(batch_size)
                with tf.GradientTape() as tape:
                    q1 = policy_net(s1, training=True)
                    q2 = target_net(s2, training=False)
                    target = r + gamma * ((1 - d) * tf.reduce_max(q2, axis=1))
                    idx = tf.stack([tf.range(batch_size), a], axis=1)
                    pred = tf.gather_nd(q1, idx)
                    loss = loss_fn(target, pred)
                grads = tape.gradient(loss, policy_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
                losses.append(float(loss.numpy()))
                if steps % sync_freq == 0:
                    target_net.set_weights(policy_net.get_weights())
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
            title_prefix="Keras DQN (random) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return policy_net, losses, stats
    return policy_net, losses
