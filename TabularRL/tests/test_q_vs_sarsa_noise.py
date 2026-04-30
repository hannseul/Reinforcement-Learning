from envs.gridworld import GridWorld
from algos.q_learning import q_learning
from algos.sarsa import sarsa
from algos.schedules import linear_decay_schedule


def extract_policy(env, Q):
    policy = {}

    for state in env.states:
        if state in env.terminal_states:
            policy[state] = None
        else:
            policy[state] = max(
                env.allowed_actions(state),
                key=lambda a: Q[(state, a)]
            )

    return policy


def evaluate_policy(env, policy, episodes=500, max_steps=100):
    total_returns = []

    for _ in range(episodes):
        state = env.reset()
        episode_return = 0

        for _ in range(max_steps):
            action = policy[state]

            if action is None:
                break

            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            state = next_state

            if done:
                break

        total_returns.append(episode_return)

    return sum(total_returns) / len(total_returns)


def print_policy(env, policy, title):
    print(title)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            action = policy[(r, c)]
            row.append(action if action is not None else "T")
        print(row)
    print()


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    noise_prob=0.2,
    seed=42,
)

epsilon_schedule = linear_decay_schedule(
    start=1.0,
    end=0.05,
    decay_steps=3000,
)

Q_q = q_learning(
    env,
    num_episodes=8000,
    alpha=0.1,
    gamma=0.9,
    epsilon_schedule=epsilon_schedule,
)

policy_q = extract_policy(env, Q_q)

Q_s = sarsa(
    env,
    num_episodes=8000,
    alpha=0.1,
    gamma=0.9,
    epsilon_schedule=epsilon_schedule,
)

policy_s = extract_policy(env, Q_s)

print_policy(env, policy_q, "Q-Learning Policy:")
print_policy(env, policy_s, "SARSA Policy:")

q_score = evaluate_policy(env, policy_q)
sarsa_score = evaluate_policy(env, policy_s)

print("Average return Q-Learning:", round(q_score, 2))
print("Average return SARSA:", round(sarsa_score, 2))