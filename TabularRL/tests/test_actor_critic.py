from envs.gridworld import GridWorld
from algos.actor_critic import actor_critic, greedy_policy_from_theta


def print_policy(env, policy):
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            state = (r, c)
            action = policy[state]
            row.append(action if action is not None else "T")
        print(row)


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    seed=42,
)

theta, V, log = actor_critic(
    env,
    gamma=0.9,
    alpha_v=0.1,
    alpha_theta=0.01,
    num_episodes=10000,
    max_steps=100,
    log_interval=1000,
)

policy = greedy_policy_from_theta(env, theta)

print("\nActor-Critic Policy:")
print_policy(env, policy)

print("\nActor-Critic Values:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        row.append(round(V[(r, c)], 2))
    print(row)
    