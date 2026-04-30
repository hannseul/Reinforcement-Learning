import numpy as np
from algos.schedules import constant_schedule


def initialize_q(env, value=0.0):
    """Initialisiert Q[(state, action)] für alle erlaubten State-Action-Paare."""
    Q = {}
    for state in env.states:
        for action in env.allowed_actions(state):
            Q[(state, action)] = float(value)
    return Q


def epsilon_greedy_action(env, Q, state, epsilon, rng=None):
    """Wählt epsilon-greedy bezüglich einer Q-Funktion."""
    rng = np.random.default_rng() if rng is None else rng
    actions = env.allowed_actions(state)
    if not actions:
        return None

    if rng.random() < epsilon:
        return rng.choice(actions)

    values = np.array([Q.get((state, a), 0.0) for a in actions], dtype=float)
    max_value = np.max(values)
    best_actions = [a for a, v in zip(actions, values) if v == max_value]
    return rng.choice(best_actions)


def greedy_action_from_sum(env, Q1, Q2, state, rng=None):
    """Greedy-Aktion bezüglich Q1 + Q2; zufälliges Tie-Breaking."""
    rng = np.random.default_rng() if rng is None else rng
    actions = env.allowed_actions(state)
    if not actions:
        return None

    values = np.array([Q1.get((state, a), 0.0) + Q2.get((state, a), 0.0) for a in actions])
    max_value = np.max(values)
    best_actions = [a for a, v in zip(actions, values) if v == max_value]
    return rng.choice(best_actions)


def double_q_learning(
    env,
    gamma=0.9,
    alpha=0.1,
    epsilon_schedule=None,
    num_episodes=5000,
    max_steps=100,
    seed=None,
    log_interval=None,
):
    """
    Double Q-Learning.

    Idee:
    - Es werden zwei Q-Funktionen Q1 und Q2 gelernt.
    - Für das Update wird zufällig eine der beiden aktualisiert.
    - Die Aktionsauswahl im Target kommt aus der aktualisierten Q-Funktion,
      die Bewertung aber aus der anderen Q-Funktion.
    - Dadurch wird der Max-Operator entkoppelt und Overestimation Bias reduziert.

    Rückgabe:
        Q_sum: Q1 + Q2, geeignet für greedy policy
        Q1, Q2: einzelne Schätzer
        log: Liste mit Evaluationswerten, falls log_interval gesetzt ist
    """
    rng = np.random.default_rng(seed)

    if epsilon_schedule is None:
        epsilon_schedule = constant_schedule(0.1)

    Q1 = initialize_q(env)
    Q2 = initialize_q(env)
    log = []

    for episode in range(num_episodes):
        epsilon = epsilon_schedule(episode)
        state = env.reset()

        for _ in range(max_steps):
            if state in env.terminal_states:
                break

            # Verhalten: epsilon-greedy bezüglich Q1 + Q2
            Q_sum_for_action = {
                (s, a): Q1.get((s, a), 0.0) + Q2.get((s, a), 0.0)
                for (s, a) in set(Q1.keys()).union(Q2.keys())
            }
            action = epsilon_greedy_action(env, Q_sum_for_action, state, epsilon, rng=rng)
            if action is None:
                break

            next_state, reward, done, _ = env.step(action)

            if rng.random() < 0.5:
                # Update Q1, Bewertung mit Q2
                next_actions = env.allowed_actions(next_state)
                if done or not next_actions:
                    td_target = reward
                else:
                    a_star = max(next_actions, key=lambda a: Q1.get((next_state, a), 0.0))
                    td_target = reward + gamma * Q2.get((next_state, a_star), 0.0)

                Q1[(state, action)] += alpha * (td_target - Q1[(state, action)])
            else:
                # Update Q2, Bewertung mit Q1
                next_actions = env.allowed_actions(next_state)
                if done or not next_actions:
                    td_target = reward
                else:
                    a_star = max(next_actions, key=lambda a: Q2.get((next_state, a), 0.0))
                    td_target = reward + gamma * Q1.get((next_state, a_star), 0.0)

                Q2[(state, action)] += alpha * (td_target - Q2[(state, action)])

            state = next_state
            if done:
                break

        if log_interval is not None and (episode + 1) % log_interval == 0:
            Q_sum = combine_double_q(Q1, Q2)
            avg_return = evaluate_greedy_policy(env, Q_sum, episodes=100, max_steps=max_steps)
            log.append((episode + 1, avg_return))

    Q_sum = combine_double_q(Q1, Q2)
    return Q_sum, Q1, Q2, log


def combine_double_q(Q1, Q2):
    """Kombiniert Q1 und Q2 durch Addition."""
    keys = set(Q1.keys()).union(Q2.keys())
    return {key: Q1.get(key, 0.0) + Q2.get(key, 0.0) for key in keys}


def evaluate_greedy_policy(env, Q, episodes=100, max_steps=200):
    """Monte-Carlo-Evaluation der greedy Policy aus Q."""
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            actions = env.allowed_actions(state)
            if state in env.terminal_states or not actions:
                break
            action = max(actions, key=lambda a: Q.get((state, a), 0.0))
            next_state, reward, done, _ = env.step(action)
            total += reward
            state = next_state
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))
