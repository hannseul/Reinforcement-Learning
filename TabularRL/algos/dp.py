import numpy as np


def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=10_000):
    V = {state: 0.0 for state in env.states}

    for iteration in range(max_iterations):
        delta = 0.0

        for state in env.states:
            if state in env.terminal_states:
                continue

            old_value = V[state]

            action_values = []

            for action in env.allowed_actions(state):
                q_value = 0.0

                transitions = env.get_transition_probabilities(state, action)

                for next_state, prob in transitions.items():
                    reward = env.expected_reward(next_state)
                    q_value += prob * (reward + gamma * V[next_state])

                action_values.append(q_value)

            V[state] = max(action_values)
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    return V

def greedy_policy_from_value(env, V, gamma=0.9):
    policy = {}

    for state in env.states:
        if state in env.terminal_states:
            policy[state] = None
            continue

        best_action = None
        best_value = -float("inf")

        for action in env.allowed_actions(state):
            q_value = 0.0
            transitions = env.get_transition_probabilities(state, action)

            for next_state, prob in transitions.items():
                reward = env.expected_reward(next_state)
                q_value += prob * (reward + gamma * V[next_state])

            if q_value > best_value:
                best_value = q_value
                best_action = action

        policy[state] = best_action

    return policy

def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    V = {state: 0.0 for state in env.states}

    while True:
        delta = 0.0

        for state in env.states:
            if state in env.terminal_states:
                continue

            action = policy[state]

            new_value = 0.0
            transitions = env.get_transition_probabilities(state, action)

            for next_state, prob in transitions.items():
                reward = env.expected_reward(next_state)
                new_value += prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(V[state] - new_value))
            V[state] = new_value

        if delta < theta:
            break

    return V

def policy_iteration(env, gamma=0.9, theta=1e-6):
    # Schritt 0: initial policy (z.B. random/first action)
    policy = {}

    for state in env.states:
        if state in env.terminal_states:
            policy[state] = None
        else:
            policy[state] = env.allowed_actions(state)[0]

    while True:
        # Schritt 1: Policy Evaluation
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)

        policy_stable = True

        # Schritt 2: Policy Improvement
        for state in env.states:
            if state in env.terminal_states:
                continue

            old_action = policy[state]

            best_action = None
            best_value = -float("inf")

            for action in env.allowed_actions(state):
                q_value = 0.0
                transitions = env.get_transition_probabilities(state, action)

                for next_state, prob in transitions.items():
                    reward = env.expected_reward(next_state)
                    q_value += prob * (reward + gamma * V[next_state])

                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            policy[state] = best_action

            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, V