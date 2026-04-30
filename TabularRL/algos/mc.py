import numpy as np


def generate_episode(env, policy, max_steps=100):
    episode = []

    #random start state
    state = tuple(env.rng.choice(env.states))
    env.current_state = state

    for _ in range(max_steps):
        action = policy[state]

        if action is None:
            break

        next_state, reward, done, _ = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if done:
            break

    return episode


def mc_policy_evaluation(env, policy, gamma=0.9, num_episodes=1000):
    returns = {state: [] for state in env.states}
    V = {state: 0.0 for state in env.states}

    for _ in range(num_episodes):
        episode = generate_episode(env, policy)

        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G

            # First-visit MC
            if state not in visited:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited.add(state)

    return V

