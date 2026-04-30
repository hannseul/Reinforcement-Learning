from envs.gridworld import GridWorld
from algos.q_learning import q_learning
from algos.sarsa import sarsa
from algos.actor_critic import actor_critic, greedy_policy_from_theta
from algos.schedules import linear_decay_schedule


def extract_policy_from_q(env, Q):
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
    returns = []

    for _ in range(episodes):
        state = env.reset()
        episode_return = 0.0

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
    print(title)

    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            state = (r, c)
            action = policy[state]
            row.append(action if action is not None else "T")
        print(row)

    print()


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    noise_prob=0.1,
    seed=42,
)

epsilon_schedule = linear_decay_schedule(
    start=1.0,
    end=0.05,
    decay_steps=3000,
)

Q_q, _ = q_learning(
    env,
    gamma=0.9,
    alpha=0.1,
    epsilon_schedule=epsilon_schedule,
    num_episodes=8000,
    max_steps=100,
    log_interval=None,
)

policy_q = extract_policy_from_q(env, Q_q)

Q_s = sarsa(
    env,
    gamma=0.9,
    alpha=0.1,
    epsilon_schedule=epsilon_schedule,
    num_episodes=8000,
    max_steps=100,
)

policy_s = extract_policy_from_q(env, Q_s)

theta_ac, V_ac, _ = actor_critic(
    env,
    gamma=0.9,
    alpha_v=0.1,
    alpha_theta=0.01,
    num_episodes=8000,
    max_steps=100,
    log_interval=None,
)

policy_ac = greedy_policy_from_theta(env, theta_ac)

print_policy(env, policy_q, "Q-Learning Policy:")
print_policy(env, policy_s, "SARSA Policy:")
print_policy(env, policy_ac, "Actor-Critic Policy:")

q_return = evaluate_policy(env, policy_q)
sarsa_return = evaluate_policy(env, policy_s)
ac_return = evaluate_policy(env, policy_ac)

print("Average return Q-Learning:", round(q_return, 2))
print("Average return SARSA:", round(sarsa_return, 2))
print("Average return Actor-Critic:", round(ac_return, 2))