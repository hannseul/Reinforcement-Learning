import numpy as np


def true_q_from_value(env, V, gamma=0.9):
    """
    Berechnet Q_true aus einer bekannten Value-Funktion V und bekannten Transitionen.
    Typischer Einsatz: V kommt aus Value Iteration.
    """
    Q_true = {}
    for state in env.states:
        for action in env.allowed_actions(state):
            q = 0.0
            transitions = env.get_transition_probabilities(state, action)
            for next_state, prob in transitions.items():
                reward = _expected_reward(env, state, action, next_state)
                q += prob * (reward + gamma * V[next_state])
            Q_true[(state, action)] = q
    return Q_true


def bias_table(Q_est, Q_true):
    """Gibt pro State-Action-Paar den Bias Q_est - Q_true zurück."""
    keys = sorted(set(Q_true.keys()).intersection(Q_est.keys()), key=str)
    return {key: Q_est[key] - Q_true[key] for key in keys}


def summed_total_bias(Q_est, Q_true):
    """Summe der Bias-Werte über alle State-Action-Paare."""
    b = bias_table(Q_est, Q_true)
    return float(sum(b.values()))


def summed_squared_total_bias(Q_est, Q_true):
    """Summe der quadrierten Bias-Werte über alle State-Action-Paare."""
    b = bias_table(Q_est, Q_true)
    return float(sum(value ** 2 for value in b.values()))


def selected_bias(Q_est, Q_true, selected_pairs):
    """Bias und quadrierter Bias für ausgewählte State-Action-Paare."""
    rows = []
    for pair in selected_pairs:
        if pair not in Q_est or pair not in Q_true:
            continue
        bias = Q_est[pair] - Q_true[pair]
        rows.append({
            "state": pair[0],
            "action": pair[1],
            "q_est": float(Q_est[pair]),
            "q_true": float(Q_true[pair]),
            "bias": float(bias),
            "squared_bias": float(bias ** 2),
        })
    return rows


def summarize_bias(Q_est, Q_true, selected_pairs=None):
    """Kompakte Zusammenfassung für Overestimation-Bias-Experimente."""
    b = bias_table(Q_est, Q_true)
    values = np.array(list(b.values()), dtype=float) if b else np.array([])

    summary = {
        "n_pairs": int(len(values)),
        "summed_total_bias": summed_total_bias(Q_est, Q_true),
        "summed_squared_total_bias": summed_squared_total_bias(Q_est, Q_true),
        "mean_bias": float(np.mean(values)) if len(values) else 0.0,
        "max_positive_bias": float(np.max(values)) if len(values) else 0.0,
        "max_negative_bias": float(np.min(values)) if len(values) else 0.0,
    }

    if selected_pairs is not None:
        summary["selected"] = selected_bias(Q_est, Q_true, selected_pairs)

    return summary


def compare_biases(named_q_estimates, Q_true, selected_pairs=None):
    """
    Vergleicht mehrere Algorithmen.

    named_q_estimates: dict, z.B. {"Q-learning": Q_q, "Double Q": Q_double}
    """
    return {
        name: summarize_bias(Q_est, Q_true, selected_pairs=selected_pairs)
        for name, Q_est in named_q_estimates.items()
    }


def _expected_reward(env, state, action, next_state=None):
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
