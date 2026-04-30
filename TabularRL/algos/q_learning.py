import numpy as np
from algos.schedules import constant_schedule

def epsilon_greedy_action(env, Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(env.allowed_actions(state))

    # greedy
    q_values = {a: Q[(state, a)] for a in env.allowed_actions(state)}
    return max(q_values, key=q_values.get)


def q_learning(
    env,
    gamma=0.9,
    alpha=0.1,
    epsilon_schedule=None,
    num_episodes=5000,
    max_steps=100,
    log_interval=500,
    returns_log=[],
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

        for _ in range(max_steps):
            if state in env.terminal_states:
                break

            action = epsilon_greedy_action(env, Q, state, epsilon)

            next_state, reward, done, _ = env.step(action)

            # max over next actions
            next_actions = env.allowed_actions(next_state)
            if next_actions:
                max_q_next = max(Q[(next_state, a)] for a in next_actions)
            else:
                max_q_next = 0.0

            # Q update
            td_target = reward + gamma * max_q_next
            td_error = td_target - Q[(state, action)]

            Q[(state, action)] += alpha * td_error

            state = next_state

            if done:
                break
        
        if log_interval is not None and (episode + 1) % log_interval == 0:
            avg_return = evaluate_current_policy(env, Q)
            print(f"Episode {episode+1}: Avg Return = {round(avg_return,2)}")
            returns_log.append((episode + 1, avg_return))

    return Q, returns_log

def evaluate_current_policy(env, Q, episodes=100, max_steps=200):
    total_returns = []

    for _ in range(episodes):
        state = env.reset()
        episode_return = 0

        for _ in range(max_steps):
            if state in env.terminal_states:
                break

            # greedy policy!
            action = max(
                env.allowed_actions(state),
                key=lambda a: Q[(state, a)]
            )

            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            state = next_state

            if done:
                break

        total_returns.append(episode_return)

    return sum(total_returns) / len(total_returns)
