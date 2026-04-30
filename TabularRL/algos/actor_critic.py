import numpy as np


def softmax_preferences(preferences):
    values = np.array(list(preferences.values()), dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    probs = exp_values / np.sum(exp_values)

    return {
        action: prob
        for action, prob in zip(preferences.keys(), probs)
    }


def sample_action(action_probs, rng):
    actions = list(action_probs.keys())
    probs = list(action_probs.values())
    return rng.choice(actions, p=probs)


def actor_critic(
    env,
    gamma=0.9,
    alpha_v=0.1,
    alpha_theta=0.01,
    num_episodes=5000,
    max_steps=100,
    log_interval=None,
):
    V = {state: 0.0 for state in env.states}

    theta = {}
    for state in env.states:
        theta[state] = {}
        for action in env.allowed_actions(state):
            theta[state][action] = 0.0

    returns_log = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            if state in env.terminal_states:
                break

            action_probs = softmax_preferences(theta[state])
            action = sample_action(action_probs, env.rng)

            next_state, reward, done, _ = env.step(action)
            episode_return += reward

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            delta = td_target - V[state]

            # Critic update
            V[state] += alpha_v * delta

            # Actor update: grad log pi(a|s)
            for a in theta[state]:
                if a == action:
                    grad_log = 1.0 - action_probs[a]
                else:
                    grad_log = -action_probs[a]

                theta[state][a] += alpha_theta * delta * grad_log

            state = next_state

            if done:
                break

        if log_interval is not None and (episode + 1) % log_interval == 0:
            returns_log.append((episode + 1, episode_return))
            print(f"Episode {episode + 1}: Return = {round(episode_return, 2)}")

    return theta, V, returns_log


def greedy_policy_from_theta(env, theta):
    policy = {}

    for state in env.states:
        if state in env.terminal_states:
            policy[state] = None
        else:
            policy[state] = max(theta[state], key=theta[state].get)

    return policy