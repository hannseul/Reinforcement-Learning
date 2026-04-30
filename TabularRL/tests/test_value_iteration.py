from envs.gridworld import GridWorld
from algos.dp import value_iteration, greedy_policy_from_value
from algos.dp import policy_evaluation
from algos.dp import policy_iteration


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
)

V = value_iteration(env, gamma=0.9)
policy_pi, V_pi = policy_iteration(env, gamma=0.9)

print("\nPolicy Iteration - Values:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        row.append(round(V_pi[(r, c)], 2))
    print(row)

print("\nPolicy Iteration - Policy:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        action = policy_pi[(r, c)]
        row.append(action if action is not None else "T")
    print(row)