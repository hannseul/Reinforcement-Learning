from envs.gridworld import GridWorld
from algos.sarsa import sarsa


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    seed=42,
)

Q = sarsa(env, num_episodes=5000, epsilon=0.1)

# Policy extrahieren
policy = {}

for state in env.states:
    if state in env.terminal_states:
        policy[state] = None
    else:
        best_action = max(
            env.allowed_actions(state),
            key=lambda a: Q[(state, a)]
        )
        policy[state] = best_action

print("SARSA Policy:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        a = policy[(r, c)]
        row.append(a if a else "T")
    print(row)
    