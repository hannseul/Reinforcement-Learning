import numpy as np


def _expected_reward(env, state, action, next_state=None):
    """
    Kompatibilitäts-Helfer für verschiedene Environment-Interfaces.

    Unterstützt:
    - env.expected_reward(state, action, next_state)
    - env.expected_reward(state, action)
    - env.expected_reward(next_state)  # aktueller GridWorld-Code
    """
    try:
        return env.expected_reward(state, action, next_state)
    except TypeError:
        pass

    try:
        return env.expected_reward(state, action)
    except TypeError:
        pass

    if next_state is None:
        return env.expected_reward(state)
    return env.expected_reward(next_state)


def finite_time_value_iteration(env, horizon, gamma=1.0):
    """
    Optimale finite-time Value Iteration per Rückwärtsinduktion.

    V[t][s] ist der optimale Wert im Zustand s bei verbleibenden Entscheidungen t..horizon-1.
    policy[t][s] ist die optimale Aktion zu diesem Zeitpunkt.
    """
    V = [{state: 0.0 for state in env.states} for _ in range(horizon + 1)]
    policy = [{state: None for state in env.states} for _ in range(horizon)]
    Q = [{} for _ in range(horizon)]

    for t in reversed(range(horizon)):
        for state in env.states:
            if state in env.terminal_states:
                V[t][state] = 0.0
                policy[t][state] = None
                continue

            actions = env.allowed_actions(state)
            if not actions:
                V[t][state] = 0.0
                policy[t][state] = None
                continue

            action_values = {}
            for action in actions:
                value = 0.0
                transitions = env.get_transition_probabilities(state, action)
                for next_state, prob in transitions.items():
                    reward = _expected_reward(env, state, action, next_state)
                    value += prob * (reward + gamma * V[t + 1][next_state])
                action_values[action] = value
                Q[t][(state, action)] = value

            best_action = max(action_values, key=action_values.get)
            V[t][state] = action_values[best_action]
            policy[t][state] = best_action

    return V, policy, Q


def finite_time_policy_evaluation(env, policy, horizon, gamma=1.0):
    """
    Finite-time Policy Evaluation per Rückwärtsinduktion.

    policy kann sein:
    - dict: state -> action, stationäre deterministische Policy
    - list[dict]: policy[t][state] -> action, zeitabhängige deterministische Policy
    - callable: policy(t, state) -> action oder action-prob dict
    - dict mit state -> dict(action -> prob), stationär stochastisch
    """
    V = [{state: 0.0 for state in env.states} for _ in range(horizon + 1)]
    Q = [{} for _ in range(horizon)]

    for t in reversed(range(horizon)):
        for state in env.states:
            if state in env.terminal_states:
                V[t][state] = 0.0
                continue

            action_probs = _policy_action_probs(env, policy, t, state)
            value = 0.0

            for action, action_prob in action_probs.items():
                q_value = 0.0
                transitions = env.get_transition_probabilities(state, action)
                for next_state, prob in transitions.items():
                    reward = _expected_reward(env, state, action, next_state)
                    q_value += prob * (reward + gamma * V[t + 1][next_state])

                Q[t][(state, action)] = q_value
                value += action_prob * q_value

            V[t][state] = value

    return V, Q


def _policy_action_probs(env, policy, t, state):
    actions = env.allowed_actions(state)
    if not actions:
        return {}

    raw = None
    if callable(policy):
        raw = policy(t, state)
    elif isinstance(policy, list):
        raw = policy[t].get(state)
    elif isinstance(policy, dict):
        raw = policy.get(state)
    else:
        raise TypeError("policy must be callable, dict, or list[dict]")

    if raw is None:
        return {}

    if isinstance(raw, dict):
        return {a: float(p) for a, p in raw.items() if p > 0}

    return {raw: 1.0}


def greedy_policy_at_time(policy, t):
    """Hilfsfunktion: extrahiert policy[t], falls zeitabhängig."""
    return policy[t]
