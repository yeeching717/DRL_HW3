import os
import numpy as np
import matplotlib
import torch

matplotlib.use("Agg")

from Gridworld import Gridworld
from hw3_1 import train_naive_dqn_static
from hw3_2 import train_double_dqn_player
from hw3_3 import train_keras_dqn_random
from hw3_4 import train_rainbow_dqn_random


action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def encode_state(game):
    state = game.board.render_np().astype(np.float32)
    return state.reshape(1, -1)


def evaluate_torch(model, mode, episodes=200, max_moves=50, size=4, obstacle_count=0):
    wins = 0
    model.eval()
    for _ in range(episodes):
        game = Gridworld(size=size, mode=mode, obstacle_count=obstacle_count)
        state = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            state_t = np.array(state, dtype=np.float32)
            qval = model(torch.from_numpy(state_t))
            action_idx = int(np.argmax(qval.detach().numpy()))
            game.makeMove(action_set[action_idx])
            state = encode_state(game)
            reward = game.reward()
            if reward != -1 or mov > max_moves:
                if reward > 0:
                    wins += 1
                status = 0
    return wins / float(episodes)


def evaluate_keras(model, mode, episodes=200, max_moves=50, size=4, obstacle_count=0):
    wins = 0
    for _ in range(episodes):
        game = Gridworld(size=size, mode=mode, obstacle_count=obstacle_count)
        state = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model(state, training=False).numpy()
            action_idx = int(np.argmax(qval))
            game.makeMove(action_set[action_idx])
            state = encode_state(game)
            reward = game.reward()
            if reward != -1 or mov > max_moves:
                if reward > 0:
                    wins += 1
                status = 0
    return wins / float(episodes)


if __name__ == "__main__":
    os.makedirs("site/assets", exist_ok=True)

    model1, _ = train_naive_dqn_static(
        epochs=200,
        plot=True,
        save_path="site/assets/hw3_1_train.png",
    )
    win1 = evaluate_torch(model1, mode="static", episodes=100)

    model2, _ = train_double_dqn_player(
        epochs=200,
        plot=True,
        save_path="site/assets/hw3_2_train.png",
    )
    win2 = evaluate_torch(model2, mode="player", episodes=100)

    model3, _ = train_keras_dqn_random(
        epochs=200,
        plot=True,
        save_path="site/assets/hw3_3_train.png",
    )
    win3 = evaluate_keras(model3, mode="random", episodes=100)

    model4, _ = train_rainbow_dqn_random(
        epochs=200,
        plot=True,
        save_path="site/assets/hw3_4_train.png",
    )

    print("Win rates:")
    print(f"HW3-1 (Naive DQN, static): {win1:.2%}")
    print(f"HW3-2 (Double DQN, player): {win2:.2%}")
    print(f"HW3-3 (Keras DQN, random): {win3:.2%}")
    win4 = evaluate_keras(model4, mode="random", episodes=100)
    print(f"HW3-4 (Rainbow DQN, random): {win4:.2%}")
