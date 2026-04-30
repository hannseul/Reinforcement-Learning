from envs.gridworld import GridWorld
from algos.q_learning import q_learning
from algos.schedules import linear_decay_schedule


env = GridWorld(
    rows=4,
    cols=12,
    start_state=(0, 0),
    terminal_rewards={(0, 11): {"type": "constant", "value": 100}},
    default_reward=-1,
    noise_prob=0.1,
    seed=42,
)

# Cliff states
for c in range(1, 11):
    env.terminal_rewards[(0, c)] = {"type": "constant", "value": -100}
    env.terminal_states.add((0, c))


epsilon_schedule = linear_decay_schedule(
    start=1.0,
    end=0.05,
    decay_steps=5000,
)

Q, log = q_learning(
    env,
    num_episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon_schedule=epsilon_schedule,
    max_steps=200,
    log_interval=500,
)

print("\nLogged returns:")
for episode, avg_return in log:
    print(f"Episode {episode}: {round(avg_return, 2)}")


print("\nFinal greedy policy:")
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
            best_action = max(
                env.allowed_actions(state),
                key=lambda a: Q[(state, a)]
            )
            row.append(best_action)

    print(row)
    