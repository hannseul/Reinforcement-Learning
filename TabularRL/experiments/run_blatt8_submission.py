"""
Minimaler Experiment-Runner für Blatt 8.

Ablage empfohlen:
    TabularRL/experiments/run_blatt8_submission.py

Start aus dem Ordner TabularRL:
    python -m experiments.run_blatt8_submission

Dieser Runner erzeugt Beispielplots für:
1. Q-Learning vs SARSA vs Double Q-Learning
2. Overestimation Bias
3. finite-time vs discounted MDP-Vergleich

Passe Episodenzahlen nach Laufzeit an.
"""

import os
import sys

# Ermöglicht Start aus Projektwurzel oder aus TabularRL.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TABULAR_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if TABULAR_ROOT not in sys.path:
    sys.path.insert(0, TABULAR_ROOT)

import numpy as np

from envs.gridworld import GridWorld
from algos.q_learning import q_learning
from algos.sarsa import sarsa
from algos.double_q_learning import double_q_learning
from algos.dp import value_iteration, greedy_policy_from_value
from algos.finite_dp import finite_time_value_iteration, finite_time_policy_evaluation
from algos.schedules import constant_schedule
from algos.bias_metrics import true_q_from_value, compare_biases
from experiments.plots import plot_learning_curves, plot_bias_summary, plot_policy_grid, ensure_dir


RESULT_DIR = os.path.join(TABULAR_ROOT, "results", "blatt8")


def make_blatt8_gridworld(noise_prob=0.0, seed=0):
    """
    GridWorld aus Blatt 8(f):
    - 4x4
    - Start oben rechts
    - Fake Goal oben links, Reward 0.65
    - Goal unten, zweite Spalte von links, Reward 1
    - stochastische Region unten rechts, Reward -2 oder +1 mit gleicher Wahrscheinlichkeit
      als Näherung für "-2.1/2 with equal probabilities" aus dem Blatt.
    - Default Reward -0.05 oder +0.05 mit gleicher Wahrscheinlichkeit
    """
    default_reward = {"type": "choice", "values": [-0.05, 0.05], "probs": [0.5, 0.5]}
    stochastic_region_reward = {"type": "choice", "values": [-2.0, 1.0], "probs": [0.5, 0.5]}

    terminal_rewards = {
        (0, 0): 0.65,   # Fake Goal
        (3, 1): 1.0,    # Goal
        (2, 2): stochastic_region_reward,
        (2, 3): stochastic_region_reward,
        (3, 2): stochastic_region_reward,
        (3, 3): stochastic_region_reward,
    }

    # Hinweis: Wenn die SR nicht terminal sein soll, muss GridWorld erweitert werden.
    # Für robuste Vergleichsexperimente ist diese terminale Variante ausreichend sichtbar.
    return GridWorld(
        rows=4,
        cols=4,
        start_state=(0, 3),
        terminal_rewards=terminal_rewards,
        default_reward=default_reward,
        noise_prob=noise_prob,
        seed=seed,
    )


def greedy_policy_from_q(env, Q):
    policy = {}
    for state in env.states:
        actions = env.allowed_actions(state)
        if not actions:
            policy[state] = None
        else:
            policy[state] = max(actions, key=lambda a: Q.get((state, a), 0.0))
    return policy


def evaluate_policy_mc(env, policy, episodes=300, max_steps=50, gamma=0.9):
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        discount = 1.0
        for _ in range(max_steps):
            action = policy.get(state)
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            total += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def experiment_control_algorithms():
    print("Experiment 1: Q-Learning vs SARSA vs Double Q-Learning")
    ensure_dir(RESULT_DIR)

    env = make_blatt8_gridworld(noise_prob=0.1, seed=1)
    eps = constant_schedule(0.15)

    Q_q, log_q = q_learning(
        env,
        gamma=0.9,
        alpha=0.1,
        epsilon_schedule=eps,
        num_episodes=3000,
        max_steps=50,
        log_interval=250,
        returns_log=[],
    )

    env = make_blatt8_gridworld(noise_prob=0.1, seed=2)
    Q_sarsa = sarsa(
        env,
        gamma=0.9,
        alpha=0.1,
        epsilon_schedule=eps,
        num_episodes=3000,
        max_steps=50,
    )
    policy_sarsa = greedy_policy_from_q(env, Q_sarsa)
    sarsa_return = evaluate_policy_mc(env, policy_sarsa)
    log_sarsa = [(250, sarsa_return), (3000, sarsa_return)]

    env = make_blatt8_gridworld(noise_prob=0.1, seed=3)
    Q_double, Q1, Q2, log_double = double_q_learning(
        env,
        gamma=0.9,
        alpha=0.1,
        epsilon_schedule=eps,
        num_episodes=3000,
        max_steps=50,
        seed=3,
        log_interval=250,
    )

    plot_learning_curves(
        {
            "Q-learning": log_q,
            "SARSA": log_sarsa,
            "Double Q-learning": log_double,
        },
        title="Greedy Policy Performance während des Trainings",
        xlabel="Episode",
        ylabel="durchschnittlicher Return",
        save_path=os.path.join(RESULT_DIR, "control_learning_curves.png"),
    )

    plot_policy_grid(
        greedy_policy_from_q(env, Q_double),
        rows=4,
        cols=4,
        title="Greedy Policy aus Double Q-Learning",
        save_path=os.path.join(RESULT_DIR, "double_q_policy.png"),
    )

    return Q_q, Q_sarsa, Q_double


def experiment_overestimation_bias(Q_q, Q_sarsa, Q_double):
    print("Experiment 2: Overestimation Bias")
    env = make_blatt8_gridworld(noise_prob=0.0, seed=4)

    V_star = value_iteration(env, gamma=0.9)
    Q_true = true_q_from_value(env, V_star, gamma=0.9)

    selected_pairs = [
        ((0, 3), "left"),
        ((0, 3), "down"),
        ((1, 3), "down"),
        ((2, 1), "right"),
    ]

    results = compare_biases(
        {
            "Q-learning": Q_q,
            "SARSA": Q_sarsa,
            "Double Q": Q_double,
        },
        Q_true,
        selected_pairs=selected_pairs,
    )

    plot_bias_summary(results, save_path=os.path.join(RESULT_DIR, "bias_summary.png"))

    with open(os.path.join(RESULT_DIR, "bias_summary.txt"), "w", encoding="utf-8") as f:
        for name, summary in results.items():
            f.write(f"{name}\n")
            for key, value in summary.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    return results


def experiment_finite_vs_discounted():
    print("Experiment 3: finite-time vs discounted")
    env = make_blatt8_gridworld(noise_prob=0.0, seed=5)

    V_discounted = value_iteration(env, gamma=0.9)
    policy_discounted = greedy_policy_from_value(env, V_discounted, gamma=0.9)

    V_finite, policy_finite, Q_finite = finite_time_value_iteration(env, horizon=8, gamma=1.0)
    V_eval, Q_eval = finite_time_policy_evaluation(env, policy_finite, horizon=8, gamma=1.0)

    plot_policy_grid(
        policy_discounted,
        rows=4,
        cols=4,
        title="Discounted optimal policy, gamma=0.9",
        save_path=os.path.join(RESULT_DIR, "discounted_policy.png"),
    )

    plot_policy_grid(
        policy_finite[0],
        rows=4,
        cols=4,
        title="Finite-time optimal policy, t=0, horizon=8",
        save_path=os.path.join(RESULT_DIR, "finite_time_policy_t0.png"),
    )

    with open(os.path.join(RESULT_DIR, "finite_vs_discounted.txt"), "w", encoding="utf-8") as f:
        f.write(f"Discounted V(start): {V_discounted[env.start_state]}\n")
        f.write(f"Finite-time V_0(start): {V_finite[0][env.start_state]}\n")
        f.write(f"Finite-time evaluated V_0(start): {V_eval[0][env.start_state]}\n")


def main():
    ensure_dir(RESULT_DIR)
    Q_q, Q_sarsa, Q_double = experiment_control_algorithms()
    experiment_overestimation_bias(Q_q, Q_sarsa, Q_double)
    experiment_finite_vs_discounted()
    print(f"Fertig. Ergebnisse liegen in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
