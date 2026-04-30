from envs.gridworld import GridWorld
from algos.mc import mc_policy_evaluation
import random


env = GridWorld(
    rows=4,
    cols=4,
    start_state=(0, 0),
    terminal_rewards={(3, 3): {"type": "constant", "value": 10}},
    default_reward=-1,
    seed=42,
)

# einfache Policy: immer Richtung Ziel
policy = {}
num_episodes = 5000

for state in env.states:
    if state in env.terminal_states:
        policy[state] = None
    else:
        policy[state] = random.choice(env.actions)
        


V_mc = mc_policy_evaluation(env, policy, num_episodes=2000)

print("Monte Carlo Values:")
for r in range(env.rows):
    row = []
    for c in range(env.cols):
        row.append(round(V_mc[(r, c)], 2))
    print(row)