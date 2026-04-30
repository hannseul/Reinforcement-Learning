from envs.gridworld import GridWorld
from algos.q_learning import q_learning
from algos.sarsa import sarsa
from algos.schedules import linear_decay_schedule


def make_cliff_env(seed=42, noise_prob=0.0, wind_prob=0.0):
    rows = 4
    cols = 12

    start_state = (0, 0)
    goal_state = (0, 11)

    terminal_rewards = {
        goal_state: {"type": "constant", "value": 100}
    }

    # Cliff states: top row between start and goal
    for c in range(1, 11):
        terminal_rewards[(0, c)] = {"type": "constant", "value": -150}

    return GridWorld(
        rows=rows,
        cols=cols,
        start_state=start_state,
        terminal_rewards=terminal_rewards,
        default_reward=-1,
        noise_prob=noise_prob,
        wind_prob=wind_prob,
        wind_direction="up",
        seed=seed,
    )


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


def evaluate_policy(env, policy, episodes=500, max_steps=200):
    returns = []

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

        returns.append(episode_return)

    return sum(returns) / len(returns)


def print_policy(env, policy, title):
    arrows = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
        None: "T",
    }

    print(title)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            state = (r, c)

            if state == env.start_state:
                row.append("S")
            elif state in env.terminal_rewards and env.terminal_rewards[state]["value"] == -100:
                row.append("C")
            elif state in env.terminal_rewards and env.terminal_rewards[state]["value"] == 100:
                row.append("G")
            else:
                row.append(arrows[policy[state]])

        print(" ".join(row))
    print()


env = make_cliff_env(
    seed=42,
    noise_prob=0.2,
    wind_prob=0.17,
)

epsilon_schedule = linear_decay_schedule(
    start=1.0,
    end=0.05,
    decay_steps=5000,
)

Q_q = q_learning(
    env,
    num_episodes=15000,
    alpha=0.1,
    gamma=0.9,
    epsilon_schedule=epsilon_schedule,
    max_steps=200,
)

policy_q = extract_policy(env, Q_q)

Q_s = sarsa(
    env,
    num_episodes=15000,
    alpha=0.1,
    gamma=0.9,
    epsilon_schedule=epsilon_schedule,
    max_steps=200,
)

policy_s = extract_policy(env, Q_s)

print_policy(env, policy_q, "Q-Learning Policy:")
print_policy(env, policy_s, "SARSA Policy:")

q_score = evaluate_policy(env, policy_q)
s_score = evaluate_policy(env, policy_s)

print("Average return Q-Learning:", round(q_score, 2))
print("Average return SARSA:", round(s_score, 2))
