from envs.gridworld import GridWorld

print("Transition probabilities:")

env = GridWorld(
    rows=4,
    cols=4,
    start_state=(1, 1),
    terminal_rewards={
        (3, 3): {"type": "constant", "value": 10}
    },
    default_reward={"type": "choice", "values": [-1, 0], "probs": [0.5, 0.5]},
    slippery_prob=0.2,
    wind_prob=0.3,
    wind_direction="up",
    noise_prob=0.1,
    seed=22,
)

probs = env.get_transition_probabilities((1, 1), "right")

print(probs)
print("Sum:", sum(probs.values()))
print("Expected rewards:")
print(env.expected_reward((0, 0)))
print(env.expected_reward((3, 3)))