"""
Test- und Kontrollskript fuer die Abgabe von Blatt 8, Aufgaben a-d.

Ablage empfohlen:
    TabularRL/experiments/test_blatt8_submission.py

Start aus dem Ordner TabularRL:
    python -m experiments.test_blatt8_submission

Schneller Testlauf:
    python -m experiments.test_blatt8_submission --quick

Abgabe-naeherer Lauf:
    python -m experiments.test_blatt8_submission --full

Das Skript prueft:
    a) Konvergenz bekannter Dynamik vs. sample-based Verfahren
    b) finite-time vs. discounted MDPs
    c) Backpropagation, Robust RL, Overestimation Bias
    d) Q-Learning Parameter: alpha / epsilon / schedule

Es speichert Ergebnisse unter:
    TabularRL/results/blatt8_test/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TABULAR_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if TABULAR_ROOT not in sys.path:
    sys.path.insert(0, TABULAR_ROOT)

from envs.gridworld import GridWorld
from algos.dp import value_iteration, policy_evaluation, greedy_policy_from_value
from algos.q_learning import q_learning
from algos.sarsa import sarsa
try:
    from algos.actor_critic import actor_critic, greedy_policy_from_theta
except Exception as exc:
    actor_critic = None
    greedy_policy_from_theta = None
    ACTOR_CRITIC_IMPORT_ERROR = exc
else:
    ACTOR_CRITIC_IMPORT_ERROR = None
from algos.schedules import constant_schedule

try:
    from algos.double_q_learning import double_q_learning
except Exception as exc:  # pragma: no cover
    double_q_learning = None
    DOUBLE_Q_IMPORT_ERROR = exc
else:
    DOUBLE_Q_IMPORT_ERROR = None

try:
    from algos.finite_dp import finite_time_value_iteration, finite_time_policy_evaluation
except Exception as exc:  # pragma: no cover
    finite_time_value_iteration = None
    finite_time_policy_evaluation = None
    FINITE_DP_IMPORT_ERROR = exc
else:
    FINITE_DP_IMPORT_ERROR = None

try:
    from algos.bias_metrics import compare_biases, true_q_from_value
except Exception:
    try:
        from algos.bias_metrics import compare_biases, true_q_from_value
    except Exception as exc:  # pragma: no cover
        compare_biases = None
        true_q_from_value = None
        BIAS_IMPORT_ERROR = exc
    else:
        BIAS_IMPORT_ERROR = None
else:
    BIAS_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


State = Tuple[int, int]
Action = str
QTable = Dict[Tuple[Any, Any], float]
Policy = Dict[Any, Optional[Any]]


@dataclass(frozen=True)
class RunConfig:
    name: str
    n_runs: int
    num_episodes: int
    eval_episodes: int
    max_steps: int
    log_interval: int


QUICK = RunConfig(
    name="quick",
    n_runs=3,
    num_episodes=800,
    eval_episodes=100,
    max_steps=50,
    log_interval=200,
)

FULL = RunConfig(
    name="full",
    n_runs=50,
    num_episodes=20_000,
    eval_episodes=2_000,
    max_steps=100,
    log_interval=500,
)

GAMMA = 0.9
RESULT_DIR = os.path.join(TABULAR_ROOT, "results", "blatt8_test")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(curves: Dict[str, Sequence[Tuple[float, float]]], title: str, xlabel: str, ylabel: str, path: str) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(8, 5))
    for name, data in curves.items():
        if not data:
            continue
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bars(values: Dict[str, float], title: str, ylabel: str, path: str) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(path))
    names = list(values.keys())
    ys = [values[n] for n in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, ys)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_policy(policy: Policy, rows: int, cols: int, title: str, path: str) -> None:
    if plt is None:
        return
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→", None: "T"}
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_title(title)
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            ax.text(c + 0.5, r + 0.5, arrows.get(policy.get(state), str(policy.get(state))), ha="center", va="center", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def make_submission_grid(noise_prob: float = 0.0, seed: int = 0) -> GridWorld:
    """4x4 GridWorld aus Blatt 8(f), auch fuer a-d als Testumgebung geeignet."""
    default_reward = {"type": "choice", "values": [-0.05, 0.05], "probs": [0.5, 0.5]}
    stochastic_region_reward = {"type": "choice", "values": [-2.0, 1.0], "probs": [0.5, 0.5]}
    terminal_rewards = {
        (0, 0): 0.65,  # Fake Goal
        (3, 1): 1.0,   # echtes Goal
        (2, 2): stochastic_region_reward,
        (2, 3): stochastic_region_reward,
        (3, 2): stochastic_region_reward,
        (3, 3): stochastic_region_reward,
    }
    return GridWorld(
        rows=4,
        cols=4,
        start_state=(0, 3),
        terminal_rewards=terminal_rewards,
        default_reward=default_reward,
        noise_prob=noise_prob,
        seed=seed,
    )


def make_backprop_grid(seed: int = 0) -> GridWorld:
    return GridWorld(
        rows=4,
        cols=4,
        start_state=(0, 0),
        terminal_rewards={(3, 3): 10.0},
        default_reward=0.0,
        seed=seed,
    )

def make_grid_b(noise_prob: float = 0.0, seed: int = 0) -> GridWorld:
    return GridWorld(
        rows=5,
        cols=5,
        start_state=(0, 0),
        terminal_rewards={(0, 2): 2.0, (4, 4): 10.0},
        default_reward=-0.1,
        seed=seed,
    )


def make_cliff_grid(wind_prob: float = 0.1, seed: int = 0) -> GridWorld:
    """Windy Cliff Walk nach Blatt 7/robust RL: Start links oben, Goal rechts oben, Cliff dazwischen."""
    terminal_rewards = {(0, c): -100.0 for c in range(1, 11)}
    terminal_rewards[(0, 11)] = 100.0
    return GridWorld(
        rows=4,
        cols=12,
        start_state=(0, 0),
        terminal_rewards=terminal_rewards,
        default_reward=-1.0,
        wind_prob=wind_prob,
        wind_direction="up",
        seed=seed,
    )


def _expected_reward(env: Any, state: Any, action: Any, next_state: Any) -> float:
    try:
        return float(env.expected_reward(state, action, next_state))
    except TypeError:
        pass
    try:
        return float(env.expected_reward(state, action))
    except TypeError:
        pass
    return float(env.expected_reward(next_state))


def greedy_policy_from_q(env: Any, Q: QTable) -> Policy:
    policy: Policy = {}
    for state in env.states:
        actions = env.allowed_actions(state)
        if not actions:
            policy[state] = None
        else:
            policy[state] = max(actions, key=lambda a: Q.get((state, a), 0.0))
    return policy


def evaluate_policy_mc(env: Any, policy: Policy, episodes: int, max_steps: int, gamma: float = GAMMA) -> float:
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
            total += discount * float(reward)
            discount *= gamma
            state = next_state
            if done:
                break
        returns.append(total)
    return float(np.mean(returns)) if returns else 0.0


def q_to_true_bias(env: Any, Q_est: QTable, V_true: Dict[Any, float], gamma: float = GAMMA) -> Dict[str, float]:
    Q_true: QTable = {}
    for state in env.states:
        for action in env.allowed_actions(state):
            value = 0.0
            for next_state, prob in env.get_transition_probabilities(state, action).items():
                reward = _expected_reward(env, state, action, next_state)
                value += prob * (reward + gamma * V_true[next_state])
            Q_true[(state, action)] = value
    biases = [Q_est.get(k, 0.0) - v for k, v in Q_true.items()]
    if not biases:
        return {"summed_total_bias": 0.0, "summed_squared_total_bias": 0.0, "mean_bias": 0.0, "max_positive_bias": 0.0}
    arr = np.array(biases, dtype=float)
    return {
        "summed_total_bias": float(np.sum(arr)),
        "summed_squared_total_bias": float(np.sum(arr ** 2)),
        "mean_bias": float(np.mean(arr)),
        "max_positive_bias": float(np.max(arr)),
    }


def value_iteration_trace(env: Any, gamma: float, iterations: int) -> Tuple[Dict[Any, float], List[Dict[Any, float]], List[Tuple[int, float]]]:
    V = {state: 0.0 for state in env.states}
    trace = [{s: V[s] for s in env.states}]
    deltas: List[Tuple[int, float]] = []
    for it in range(1, iterations + 1):
        delta = 0.0
        new_V = dict(V)
        for state in env.states:
            if state in env.terminal_states:
                continue
            values = []
            for action in env.allowed_actions(state):
                q = 0.0
                for next_state, prob in env.get_transition_probabilities(state, action).items():
                    reward = _expected_reward(env, state, action, next_state)
                    q += prob * (reward + gamma * V[next_state])
                values.append(q)
            if values:
                new_V[state] = max(values)
                delta = max(delta, abs(new_V[state] - V[state]))
        V = new_V
        trace.append({s: V[s] for s in env.states})
        deltas.append((it, delta))
    return V, trace, deltas


def aggregate_logs(logs: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    by_x: Dict[int, List[float]] = {}
    for log in logs:
        for x, y in log:
            by_x.setdefault(int(x), []).append(float(y))
    return [(x, float(np.mean(vals))) for x, vals in sorted(by_x.items())]


def run_q_learning_many(make_env: Callable[[int], Any], config: RunConfig, alpha: float, eps_fn: Callable[[int], float]) -> Tuple[QTable, List[Tuple[int, float]], List[float]]:
    logs = []
    final_returns = []
    last_Q: QTable = {}
    for seed in range(config.n_runs):
        env = make_env(seed)
        returns_log: List[Tuple[int, float]] = []
        Q, log = q_learning(
            env,
            gamma=GAMMA,
            alpha=alpha,
            epsilon_schedule=eps_fn,
            num_episodes=config.num_episodes,
            max_steps=config.max_steps,
            log_interval=config.log_interval,
            returns_log=returns_log,
        )
        last_Q = Q
        logs.append(log)
        policy = greedy_policy_from_q(env, Q)
        final_returns.append(evaluate_policy_mc(env, policy, config.eval_episodes, config.max_steps))
    return last_Q, aggregate_logs(logs), final_returns


def run_sarsa_many(make_env: Callable[[int], Any], config: RunConfig, alpha: float, eps_fn: Callable[[int], float]) -> Tuple[QTable, float]:
    final_returns = []
    last_Q: QTable = {}
    for seed in range(config.n_runs):
        env = make_env(10_000 + seed)
        Q = sarsa(env, gamma=GAMMA, alpha=alpha, epsilon_schedule=eps_fn, num_episodes=config.num_episodes, max_steps=config.max_steps)
        last_Q = Q
        final_returns.append(evaluate_policy_mc(env, greedy_policy_from_q(env, Q), config.eval_episodes, config.max_steps))
    return last_Q, float(np.mean(final_returns))


def run_double_q_many(make_env: Callable[[int], Any], config: RunConfig, alpha: float, eps_fn: Callable[[int], float]) -> Tuple[QTable, List[Tuple[int, float]], List[float]]:
    if double_q_learning is None:
        raise RuntimeError(f"double_q_learning konnte nicht importiert werden: {DOUBLE_Q_IMPORT_ERROR}")
    logs = []
    final_returns = []
    last_Q: QTable = {}
    for seed in range(config.n_runs):
        env = make_env(20_000 + seed)
        Q_sum, _, _, log = double_q_learning(
            env,
            gamma=GAMMA,
            alpha=alpha,
            epsilon_schedule=eps_fn,
            num_episodes=config.num_episodes,
            max_steps=config.max_steps,
            seed=seed,
            log_interval=config.log_interval,
        )
        last_Q = Q_sum
        logs.append(log)
        final_returns.append(evaluate_policy_mc(env, greedy_policy_from_q(env, Q_sum), config.eval_episodes, config.max_steps))
    return last_Q, aggregate_logs(logs), final_returns


def experiment_a(config: RunConfig) -> Dict[str, Any]:
    print("[a] Konvergenzraten: bekannte Dynamik vs. sample-based")
    env = make_submission_grid(noise_prob=0.0, seed=0)
    V_star = value_iteration(env, gamma=GAMMA)
    _, _, vi_deltas = value_iteration_trace(env, gamma=GAMMA, iterations=40)

    fixed_policy = greedy_policy_from_value(env, V_star, gamma=GAMMA)
    V_pi = policy_evaluation(env, fixed_policy, gamma=GAMMA)
    start_value_exact = float(V_pi[env.start_state])

    Q_q, q_log, q_returns = run_q_learning_many(lambda seed: make_submission_grid(noise_prob=0.0, seed=seed), config, alpha=0.1, eps_fn=constant_schedule(0.15))

    rows = [{"iteration": i, "bellman_delta": d} for i, d in vi_deltas]
    save_csv(rows, os.path.join(RESULT_DIR, "a_value_iteration_deltas.csv"))
    save_csv([{"episode": x, "avg_return": y} for x, y in q_log], os.path.join(RESULT_DIR, "a_q_learning_curve.csv"))
    plot_lines({"Value Iteration Delta": vi_deltas}, "a) Value Iteration Konvergenz", "Iteration", "Bellman-Delta", os.path.join(RESULT_DIR, "a_value_iteration_delta.png"))
    plot_lines({"Q-Learning Return": q_log}, "a) Sample-based Learning Curve", "Episode", "Return", os.path.join(RESULT_DIR, "a_q_learning_curve.png"))

    return {
        "exact_start_value": start_value_exact,
        "q_learning_final_return_mean": float(np.mean(q_returns)),
        "q_learning_final_return_std": float(np.std(q_returns)),
        "last_vi_delta": vi_deltas[-1][1] if vi_deltas else None,
    }


def experiment_b(config: RunConfig) -> Dict[str, Any]:
    print("[b] finite-time vs. discounted MDPs")
    if finite_time_value_iteration is None or finite_time_policy_evaluation is None:
        raise RuntimeError(f"finite_dp konnte nicht importiert werden: {FINITE_DP_IMPORT_ERROR}")

    env = make_grid_b(noise_prob=0.0, seed=1)
    V_discounted = value_iteration(env, gamma=GAMMA)
    policy_discounted = greedy_policy_from_value(env, V_discounted, gamma=GAMMA)

    rows = []
    policies = {}
    for horizon in [4, 8, 16]:
        V_ft, policy_ft, _ = finite_time_value_iteration(env, horizon=horizon, gamma=1.0)
        V_eval, _ = finite_time_policy_evaluation(env, policy_ft, horizon=horizon, gamma=1.0)
        rows.append({
            "horizon": horizon,
            "finite_time_start_value": float(V_ft[0][env.start_state]),
            "finite_time_eval_start_value": float(V_eval[0][env.start_state]),
            "first_action": str(policy_ft[0].get(env.start_state)),
        })
        policies[horizon] = {str(k): v for k, v in policy_ft[0].items()}
        plot_policy(policy_ft[0], 5, 5, f"b) Finite-time Policy, H={horizon}", os.path.join(RESULT_DIR, f"b_finite_policy_H{horizon}.png"))

    plot_policy(policy_discounted, 5, 5, "b) Discounted Policy, gamma=0.9", os.path.join(RESULT_DIR, "b_discounted_policy.png"))
    save_csv(rows, os.path.join(RESULT_DIR, "b_finite_vs_discounted.csv"))

    return {
        "discounted_start_value": float(V_discounted[env.start_state]),
        "discounted_first_action": str(policy_discounted[env.start_state]),
        "finite_time_rows": rows,
    }


def experiment_c(config: RunConfig) -> Dict[str, Any]:
    print("[c] Backpropagation, Robust RL, Overestimation Bias")

    # Backpropagation: Werte in GridWorld verbreiten sich vom Goal aus rueckwaerts.
    bp_env = make_backprop_grid(seed=2)
    _, bp_trace, _ = value_iteration_trace(bp_env, gamma=GAMMA, iterations=8)
    bp_states = [(3, 2), (2, 2), (1, 2), (0, 0)]
    bp_rows = []
    for it, V in enumerate(bp_trace):
        row = {"iteration": it}
        for state in bp_states:
            row[str(state)] = float(V[state])
        bp_rows.append(row)
    save_csv(bp_rows, os.path.join(RESULT_DIR, "c_backpropagation_values.csv"))
    plot_lines(
        {str(state): [(row["iteration"], row[str(state)]) for row in bp_rows] for state in bp_states},
        "c) Backpropagation der Values",
        "Iteration",
        "V(s)",
        os.path.join(RESULT_DIR, "c_backpropagation_values.png"),
    )

    # Robust RL: windy cliff, Q-Learning vs SARSA.
    eps = constant_schedule(0.15)
    Q_cliff, q_cliff_log, q_cliff_returns = run_q_learning_many(lambda seed: make_cliff_grid(wind_prob=0.15, seed=seed), config, alpha=0.1, eps_fn=eps)
    Q_sarsa, sarsa_return = run_sarsa_many(lambda seed: make_cliff_grid(wind_prob=0.15, seed=seed), config, alpha=0.1, eps_fn=eps)
    plot_policy(greedy_policy_from_q(make_cliff_grid(0.15, 123), Q_cliff), 4, 12, "c) Cliff Q-Learning Policy", os.path.join(RESULT_DIR, "c_cliff_q_policy.png"))
    plot_policy(greedy_policy_from_q(make_cliff_grid(0.15, 124), Q_sarsa), 4, 12, "c) Cliff SARSA Policy", os.path.join(RESULT_DIR, "c_cliff_sarsa_policy.png"))

    # Overestimation Bias: Q-Learning vs Double Q auf stochastischer GridWorld.
    env_true = make_submission_grid(noise_prob=0.0, seed=3)
    V_true = value_iteration(env_true, gamma=GAMMA)
    Q_q, _, _ = run_q_learning_many(lambda seed: make_submission_grid(noise_prob=0.0, seed=seed), config, alpha=0.1, eps_fn=constant_schedule(0.2))
    Q_double, _, _ = run_double_q_many(lambda seed: make_submission_grid(noise_prob=0.0, seed=seed), config, alpha=0.1, eps_fn=constant_schedule(0.2))

    if compare_biases is not None and true_q_from_value is not None:
        Q_true = true_q_from_value(env_true, V_true, gamma=GAMMA)
        bias_results = compare_biases({"Q-Learning": Q_q, "Double Q-Learning": Q_double}, Q_true)
    else:
        bias_results = {
            "Q-Learning": q_to_true_bias(env_true, Q_q, V_true, gamma=GAMMA),
            "Double Q-Learning": q_to_true_bias(env_true, Q_double, V_true, gamma=GAMMA),
        }
    save_json(bias_results, os.path.join(RESULT_DIR, "c_bias_summary.json"))
    plot_bars(
        {name: float(summary["summed_total_bias"]) for name, summary in bias_results.items()},
        "c) Summed Total Bias",
        "Bias",
        os.path.join(RESULT_DIR, "c_bias_total.png"),
    )
    plot_bars(
        {name: float(summary["summed_squared_total_bias"]) for name, summary in bias_results.items()},
        "c) Summed Squared Total Bias",
        "Squared Bias",
        os.path.join(RESULT_DIR, "c_bias_squared.png"),
    )

    return {
        "backpropagation_rows": bp_rows,
        "cliff_q_return_mean": float(np.mean(q_cliff_returns)),
        "cliff_sarsa_return_mean": float(sarsa_return),
        "bias_results": bias_results,
    }


def decreasing_eps(initial: float = 1.0, min_eps: float = 0.05, rate: float = 0.001) -> Callable[[int], float]:
    def schedule(t: int) -> float:
        return max(min_eps, initial / (1.0 + rate * t))
    return schedule


def experiment_d(config: RunConfig) -> Dict[str, Any]:
    print("[d] Q-Learning Parameter: alpha, epsilon, schedule")
    parameter_sets = [
        ("eps=0.00 alpha=0.10 greedy", 0.10, constant_schedule(0.00)),
        ("eps=0.01 alpha=0.10", 0.10, constant_schedule(0.01)),
        ("eps=0.05 alpha=0.10", 0.10, constant_schedule(0.05)),
        ("eps=0.15 alpha=0.10", 0.10, constant_schedule(0.15)),
        ("eps=0.40 alpha=0.10", 0.10, constant_schedule(0.40)),
        ("eps=0.15 alpha=0.03", 0.03, constant_schedule(0.15)),
        ("eps=0.15 alpha=0.05", 0.05, constant_schedule(0.15)),
        ("eps=0.15 alpha=0.30", 0.30, constant_schedule(0.15)),
        ("eps=0.15 alpha=0.50", 0.50, constant_schedule(0.15)),
        ("eps decreasing slow alpha=0.10", 0.10, decreasing_eps(1.0, 0.05, 0.0005)),
        ("eps decreasing medium alpha=0.10", 0.10, decreasing_eps(1.0, 0.05, 0.002)),
        ("eps decreasing fast alpha=0.10", 0.10, decreasing_eps(1.0, 0.01, 0.01)),
    ]

    all_rows = {}

    for noise_prob in [0.0, 0.1, 0.3]:
        summary_rows = []
        curves = {}

        for name, alpha, eps_fn in parameter_sets:
            _, log, final_returns = run_q_learning_many(
                lambda seed, p=noise_prob: make_submission_grid(noise_prob=p, seed=seed),
                config,
                alpha=alpha,
                eps_fn=eps_fn,
            )

            curves[name] = log
            summary_rows.append({
                "noise_prob": noise_prob,
                "config": name,
                "mean_final_return": float(np.mean(final_returns)),
                "std_final_return": float(np.std(final_returns)),
                "n_runs": config.n_runs,
                "num_episodes": config.num_episodes,
            })

        suffix = str(noise_prob).replace(".", "_")
        save_csv(summary_rows, os.path.join(RESULT_DIR, f"d_parameter_sweep_noise_{suffix}.csv"))
        plot_lines(
            curves,
            f"d) Q-Learning Parameter Sweep, noise={noise_prob}",
            "Episode",
            "Return",
            os.path.join(RESULT_DIR, f"d_parameter_sweep_noise_{suffix}.png"),
        )

        all_rows[f"noise={noise_prob}"] = summary_rows

    return {"rows_by_noise": all_rows}

def run_actor_critic_many(
    make_env: Callable[[int], Any],
    config: RunConfig,
    alpha_v: float = 0.1,
    alpha_theta: float = 0.01,
) -> Tuple[Policy, List[Tuple[int, float]], List[float]]:
    if actor_critic is None or greedy_policy_from_theta is None:
        raise RuntimeError(f"actor_critic konnte nicht importiert werden: {ACTOR_CRITIC_IMPORT_ERROR}")

    logs = []
    final_returns = []
    last_policy: Policy = {}

    for seed in range(config.n_runs):
        env = make_env(30_000 + seed)

        theta, V, log = actor_critic(
            env,
            gamma=GAMMA,
            alpha_v=alpha_v,
            alpha_theta=alpha_theta,
            num_episodes=config.num_episodes,
            max_steps=config.max_steps,
            log_interval=config.log_interval,
        )

        policy = greedy_policy_from_theta(env, theta)
        last_policy = policy
        logs.append(log)
        final_returns.append(
            evaluate_policy_mc(env, policy, config.eval_episodes, config.max_steps)
        )

    return last_policy, aggregate_logs(logs), final_returns

def experiment_e_actor_critic(config: RunConfig) -> Dict[str, Any]:
    print("[e] Actor-Critic vs. Q-Learning vs. SARSA")

    make_env = lambda seed: make_submission_grid(noise_prob=0.1, seed=seed)

    Q_q, q_log, q_returns = run_q_learning_many(
        make_env,
        config,
        alpha=0.10,
        eps_fn=constant_schedule(0.15),
    )

    Q_sarsa, sarsa_return = run_sarsa_many(
        make_env,
        config,
        alpha=0.10,
        eps_fn=constant_schedule(0.15),
    )

    ac_policy, ac_log, ac_returns = run_actor_critic_many(
        make_env,
        config,
        alpha_v=0.10,
        alpha_theta=0.01,
    )

    curves = {
        "Q-Learning": q_log,
        "Actor-Critic": ac_log,
    }

    plot_lines(
        curves,
        "e) Actor-Critic vs. Q-Learning",
        "Episode",
        "Return",
        os.path.join(RESULT_DIR, "e_actor_critic_vs_q_learning_curve.png"),
    )

    final_values = {
        "Q-Learning": float(np.mean(q_returns)),
        "SARSA": float(sarsa_return),
        "Actor-Critic": float(np.mean(ac_returns)),
    }

    plot_bars(
        final_values,
        "e) Final Policy Evaluation",
        "Average Return",
        os.path.join(RESULT_DIR, "e_actor_critic_vs_q_learning_sarsa.png"),
    )

    plot_policy(
        greedy_policy_from_q(make_env(123), Q_q),
        4,
        4,
        "e) Q-Learning Policy",
        os.path.join(RESULT_DIR, "e_q_learning_policy.png"),
    )

    plot_policy(
        greedy_policy_from_q(make_env(124), Q_sarsa),
        4,
        4,
        "e) SARSA Policy",
        os.path.join(RESULT_DIR, "e_sarsa_policy.png"),
    )

    plot_policy(
        ac_policy,
        4,
        4,
        "e) Actor-Critic Policy",
        os.path.join(RESULT_DIR, "e_actor_critic_policy.png"),
    )

    save_csv(
        [
            {
                "algorithm": name,
                "mean_final_return": value,
                "n_runs": config.n_runs,
                "num_episodes": config.num_episodes,
            }
            for name, value in final_values.items()
        ],
        os.path.join(RESULT_DIR, "e_actor_critic_summary.csv"),
    )

    return {
        "final_values": final_values,
        "actor_critic_mean": float(np.mean(ac_returns)),
        "actor_critic_std": float(np.std(ac_returns)),
    }

def check_required_modules() -> Dict[str, Any]:
    checks = {
        "double_q_learning": DOUBLE_Q_IMPORT_ERROR is None,
        "finite_dp": FINITE_DP_IMPORT_ERROR is None,
        "bias_metrics": BIAS_IMPORT_ERROR is None,
        "matplotlib_available": plt is not None,
        "actor_critic": ACTOR_CRITIC_IMPORT_ERROR is None,
    }
    errors = {
        "double_q_learning_error": repr(DOUBLE_Q_IMPORT_ERROR) if DOUBLE_Q_IMPORT_ERROR else None,
        "finite_dp_error": repr(FINITE_DP_IMPORT_ERROR) if FINITE_DP_IMPORT_ERROR else None,
        "bias_metrics_error": repr(BIAS_IMPORT_ERROR) if BIAS_IMPORT_ERROR else None,
        "actor_critic_error": repr(ACTOR_CRITIC_IMPORT_ERROR) if ACTOR_CRITIC_IMPORT_ERROR else None,
    }
    return {"checks": checks, "errors": errors}



'''
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="kurzer Funktionstest")
    parser.add_argument("--full", action="store_true", help="abgabe-naeherer Lauf")
    args = parser.parse_args()
    config = FULL if args.full else QUICK

    ensure_dir(RESULT_DIR)
    report: Dict[str, Any] = {"config": config.__dict__, "module_checks": check_required_modules()}
    save_json(report["module_checks"], os.path.join(RESULT_DIR, "module_checks.json"))

    failures: List[str] = []
    for label, fn in [
        ("a", experiment_a),
        ("b", experiment_b),
        ("c", experiment_c),
        ("d", experiment_d),
        ("e_actor_critic", experiment_e_actor_critic),
    ]:
        try:
            report[label] = fn(config)
            print(f"[{label}] OK")
        except Exception as exc:
            failures.append(label)
            report[label] = {"error": repr(exc)}
            print(f"[{label}] FEHLER: {exc}")

    report["failures"] = failures
    save_json(report, os.path.join(RESULT_DIR, "blatt8_test_report.json"))

    print("\nFertig.")
    print(f"Modus: {config.name}")
    print(f"Ergebnisse: {RESULT_DIR}")
    if failures:
        print(f"Fehlgeschlagene Teile: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="kurzer Funktionstest")
    parser.add_argument("--full", action="store_true", help="abgabe-naeherer Lauf")
    args = parser.parse_args()
    config = FULL if args.full else QUICK

    ensure_dir(RESULT_DIR)
    report: Dict[str, Any] = {"config": config.__dict__, "module_checks": check_required_modules()}
    save_json(report["module_checks"], os.path.join(RESULT_DIR, "module_checks.json"))

    failures: List[str] = []
    for label, fn in [
        ("b", experiment_b),
    ]:
        try:
            report[label] = fn(config)
            print(f"[{label}] OK")
        except Exception as exc:
            failures.append(label)
            report[label] = {"error": repr(exc)}
            print(f"[{label}] FEHLER: {exc}")

    report["failures"] = failures
    save_json(report, os.path.join(RESULT_DIR, "blatt8_test_report.json"))

    print("\nFertig.")
    print(f"Modus: {config.name}")
    print(f"Ergebnisse: {RESULT_DIR}")
    if failures:
        print(f"Fehlgeschlagene Teile: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()