import numpy as np
from algos.schedules import constant_schedule


def epsilon_greedy_action(env, Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(env.allowed_actions(state))

    q_values = {a: Q[(state, a)] for a in env.allowed_actions(state)}
    return max(q_values, key=q_values.get)


def sarsa(
    env,
    gamma=0.9,
    alpha=0.1,
    epsilon_schedule=None,
    num_episodes=5000,
    max_steps=100,
):
    
    if epsilon_schedule is None:
        epsilon_schedule = constant_schedule(0.1)

    Q = {}

    # initialisieren
    for state in env.states:
        for action in env.actions:
            Q[(state, action)] = 0.0

    for episode in range(num_episodes):
        epsilon = epsilon_schedule(episode)
        
        state = env.reset()

        if state in env.terminal_states:
            continue

        action = epsilon_greedy_action(env, Q, state, epsilon)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)

            if next_state in env.terminal_states:
                td_target = reward
                Q[(state, action)] += alpha * (td_target - Q[(state, action)])
                break

            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            # SARSA Update
            td_target = reward + gamma * Q[(next_state, next_action)]
            td_error = td_target - Q[(state, action)]

            Q[(state, action)] += alpha * td_error

            state = next_state
            action = next_action

            if done:
                break

    return Q