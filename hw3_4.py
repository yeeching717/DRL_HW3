import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

from Gridworld import Gridworld


action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def encode_state(game):
    state = game.board.render_np().astype(np.float32)
    return state.reshape(1, -1)


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


class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, sigma_init=0.5, activation=None):
        super().__init__()
        self.units = units
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        mu_range = 1.0 / np.sqrt(input_dim)
        self.w_mu = self.add_weight(
            name="w_mu",
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            name="w_sigma",
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init / np.sqrt(input_dim)),
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name="b_mu",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name="b_sigma",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_init / np.sqrt(input_dim)),
            trainable=True,
        )

    def call(self, inputs, training=False):
        if training:
            eps_in = tf.random.normal((inputs.shape[-1],))
            eps_out = tf.random.normal((self.units,))
            eps_w = tf.tensordot(
                tf.sign(eps_in) * tf.sqrt(tf.abs(eps_in)),
                tf.sign(eps_out) * tf.sqrt(tf.abs(eps_out)),
                axes=0,
            )
            eps_b = tf.sign(eps_out) * tf.sqrt(tf.abs(eps_out))
            w = self.w_mu + self.w_sigma * eps_w
            b = self.b_mu + self.b_sigma * eps_b
        else:
            w = self.w_mu
            b = self.b_mu
        out = tf.matmul(inputs, w) + b
        if self.activation is not None:
            out = self.activation(out)
        return out


class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, priority=None):
        if priority is None:
            priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, frame_idx):
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


def get_n_step_info(buffer, gamma):
    reward = 0.0
    next_state = buffer[-1][3]
    done = False
    for i, (_, _, r, ns, d) in enumerate(buffer):
        reward += (gamma ** i) * r
        next_state = ns
        if d:
            done = True
            break
    return reward, next_state, done


def build_dueling_noisy_model(input_dim, hidden1=150, hidden2=100):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden1, activation='relu')(inputs)
    x = tf.keras.layers.Dense(hidden2, activation='relu')(x)

    v = NoisyDense(hidden2, activation='relu')(x)
    v = NoisyDense(1)(v)

    a = NoisyDense(hidden2, activation='relu')(x)
    a = NoisyDense(4)(a)

    a_mean = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
    q = tf.keras.layers.Add()([v, tf.keras.layers.Subtract()([a, a_mean])])
    return tf.keras.Model(inputs=inputs, outputs=q)


# HW3-4: Simplified Rainbow DQN for random mode (Keras)

def train_rainbow_dqn_random(
    epochs=5000,
    gamma=0.9,
    n_step=3,
    batch_size=200,
    mem_size=2000,
    max_moves=50,
    sync_freq=500,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_frames=100000,
    epsilon=0.05,
    plot=False,
    window=50,
    return_stats=False,
    save_path=None,
    size=4,
    obstacle_count=0,
):
    sample_game = Gridworld(size=size, mode='random', obstacle_count=obstacle_count)
    input_dim = encode_state(sample_game).shape[1]
    policy_net = build_dueling_noisy_model(input_dim)
    target_net = build_dueling_noisy_model(input_dim)
    target_net.set_weights(policy_net.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber(reduction="none")
    replay = PrioritizedReplay(mem_size, alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames)

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
        n_step_buffer = deque(maxlen=n_step)

        while status == 1:
            step += 1
            mov += 1

            qval = policy_net(state, training=True).numpy()
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(np.argmax(qval))

            game.makeMove(action_set[action_idx])
            next_state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
            reward = game.reward()
            episode_return += reward
            done = reward > 0

            n_step_buffer.append((state, action_idx, reward, next_state, done))
            if len(n_step_buffer) >= n_step:
                r_n, ns_n, d_n = get_n_step_info(n_step_buffer, gamma)
                s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
                replay.push(s0, a0, r_n, ns_n, d_n)

            state = next_state

            if done or mov > max_moves:
                while len(n_step_buffer) > 1:
                    n_step_buffer.popleft()
                    r_n, ns_n, d_n = get_n_step_info(n_step_buffer, gamma)
                    s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
                    replay.push(s0, a0, r_n, ns_n, d_n)
                n_step_buffer.clear()

            if len(replay) > batch_size:
                batch, indices, weights = replay.sample(batch_size, frame_idx=step)
                s1 = np.vstack([b[0] for b in batch])
                a = np.array([b[1] for b in batch], dtype=np.int32)
                r = np.array([b[2] for b in batch], dtype=np.float32)
                s2 = np.vstack([b[3] for b in batch])
                d = np.array([b[4] for b in batch], dtype=np.float32)

                with tf.GradientTape() as tape:
                    q1 = policy_net(s1, training=True)
                    next_actions = tf.cast(tf.argmax(policy_net(s2, training=True), axis=1), tf.int32)
                    q2 = target_net(s2, training=False)
                    idx2 = tf.stack([tf.range(batch_size), next_actions], axis=1)
                    target_q = tf.gather_nd(q2, idx2)
                    target = r + (gamma ** n_step) * ((1 - d) * target_q)
                    idx1 = tf.stack([tf.range(batch_size), a], axis=1)
                    pred = tf.gather_nd(q1, idx1)
                    td_error = target - pred
                    per_loss = loss_fn(target, pred)
                    per_loss = per_loss * weights
                    loss = tf.reduce_mean(per_loss)

                grads = tape.gradient(loss, policy_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
                losses.append(float(loss.numpy()))

                new_priorities = np.abs(td_error.numpy()) + 1e-6
                replay.update_priorities(indices, new_priorities)

                if step % sync_freq == 0:
                    target_net.set_weights(policy_net.get_weights())

            if done or mov > max_moves:
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
            title_prefix="Rainbow DQN (random) - ",
            save_path=save_path,
        )

    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return policy_net, losses, stats
    return policy_net, losses


def train_rainbow_dqn_random_budget(
    total_steps=2000,
    gamma=0.9,
    batch_size=32,
    mem_size=200,
    max_moves=20,
    sync_freq=200,
    epsilon=0.05,
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
    policy_net = build_dueling_noisy_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    target_net = build_dueling_noisy_model(input_dim, hidden1=hidden1, hidden2=hidden2)
    target_net.set_weights(policy_net.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()
    replay = deque(maxlen=mem_size)
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
            qval = policy_net(state, training=True).numpy()
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = int(np.argmax(qval))
            game.makeMove(action_set[action_idx])
            next_state = encode_state(game) + np.random.rand(1, input_dim) / 100.0
            reward = game.reward()
            episode_return += reward
            done = reward > 0
            replay.append((state, action_idx, reward, next_state, done))
            state = next_state

            if len(replay) > batch_size:
                batch = random.sample(replay, batch_size)
                s1 = np.vstack([b[0] for b in batch])
                a = np.array([b[1] for b in batch], dtype=np.int32)
                r = np.array([b[2] for b in batch], dtype=np.float32)
                s2 = np.vstack([b[3] for b in batch])
                d = np.array([b[4] for b in batch], dtype=np.float32)

                with tf.GradientTape() as tape:
                    q1 = policy_net(s1, training=True)
                    next_actions = tf.cast(tf.argmax(policy_net(s2, training=True), axis=1), tf.int32)
                    q2 = target_net(s2, training=False)
                    idx2 = tf.stack([tf.range(batch_size), next_actions], axis=1)
                    target_q = tf.gather_nd(q2, idx2)
                    target = r + gamma * ((1 - d) * target_q)
                    idx1 = tf.stack([tf.range(batch_size), a], axis=1)
                    pred = tf.gather_nd(q1, idx1)
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
            title_prefix="Rainbow DQN (random) - ",
            save_path=save_path,
        )
    stats = {"episode_returns": episode_returns, "wins": wins}
    if return_stats:
        return policy_net, losses, stats
    return policy_net, losses
