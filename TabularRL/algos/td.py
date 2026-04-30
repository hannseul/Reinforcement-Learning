import numpy as np
import random


def td0_policy_evaluation(env, policy, gamma=0.9, alpha=0.1, num_episodes=1000, max_steps=100):
    V = {state: 0.0 for state in env.states}

    for _ in range(num_episodes):
        state = random.choice(env.states)
        env.current_state = state

        for _ in range(max_steps):
            action = policy[state]

            if action is None:
                break

            next_state, reward, done, _ = env.step(action)

            # TD Update
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]

            V[state] += alpha * td_error

            state = next_state

            if done:
                break

    return V