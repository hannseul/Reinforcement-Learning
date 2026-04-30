from envs.gridworld import GridWorld
from algos.td import td0_policy_evaluation


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    seed=42,
)

# gleiche Policy wie vorher (Richtung Ziel)
policy = {}

for state in env.states:
    if state in env.terminal_states:
        policy[state] = None
    else:
        policy[state] = "down" if state[0] < 3 else "right"


V_td = td0_policy_evaluation(env, policy, num_episodes=5000, alpha=0.1)

print("TD(0) Values:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        row.append(round(V_td[(r, c)], 2))
    print(row)
    