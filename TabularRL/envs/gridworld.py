import numpy as np


class GridWorld:
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(
        self,
        rows=4,
        cols=4,
        start_state=(0, 0),
        terminal_rewards=None,
        default_reward=0.0,
        wall_mode="stay",
        seed=None,
        wind_prob=0.0,
        wind_direction="up",
        slippery_prob=0.0,
        noise_prob=0.0,
        noise_probs=None,
    ):
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.default_reward = default_reward
        self.wall_mode = wall_mode
        self.rng = np.random.default_rng(seed)

        self.terminal_rewards = terminal_rewards or {}
        self.terminal_states = set(self.terminal_rewards.keys())

        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        self.actions = self.ACTIONS.copy()

        self.current_state = self.start_state

        self.wind_prob = wind_prob
        self.wind_direction = wind_direction

        self.slippery_prob = slippery_prob
        self.noise_prob = noise_prob
        self.noise_probs = noise_probs

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def allowed_actions(self, state=None):
        state = self.current_state if state is None else state

        if state in self.terminal_states:
            return []

        if self.wall_mode == "stay":
            return self.actions.copy()

        allowed = []
        for action in self.actions:
            next_state = self._deterministic_next_state(state, action)
            if next_state != state:
                allowed.append(action)

        return allowed
    
    def _slippery_action(self, action):
        return self.rng.choice(self._adjacent_actions(action))

    def _random_noisy_action(self):
        if self.noise_probs is None:
            return self.rng.choice(self.actions)

        return self.rng.choice(
            self.actions,
            p=self.noise_probs
        )
    
    def _adjacent_actions(self, action):
        adjacent_actions = {
            "up": ["left", "right"],
            "down": ["left", "right"],
            "left": ["up", "down"],
            "right": ["up", "down"],
        }
        return adjacent_actions[action]
    
    def step(self, action):
        if action not in self.allowed_actions(self.current_state):
            raise ValueError(f"Action {action} is not allowed in state {self.current_state}")

        actual_action = action

        #slippery
        if self.rng.random() < self.slippery_prob:
            actual_action = self._slippery_action(action)

        #wind
        if self.rng.random() < self.wind_prob:
            actual_action = self.wind_direction

        #random noise
        if self.rng.random() < self.noise_prob:
            actual_action = self._random_noisy_action()

        next_state = self._deterministic_next_state(
            self.current_state,
            actual_action
        )

        reward = self._reward(next_state)
        done = next_state in self.terminal_states

        self.current_state = self.start_state if done else next_state

        return next_state, reward, done, {}

    def get_transition_probabilities(self, state, action):
        """
        Returns a dict:
        {
            next_state: probability
        }
        """
        if state in self.terminal_states:
            return {state: 1.0}

        probs = {}

        def add_transition(actual_action, prob):
            next_state = self._deterministic_next_state(state, actual_action)
            probs[next_state] = probs.get(next_state, 0.0) + prob

        # Base action probability
        base_prob = 1.0 - self.slippery_prob - self.wind_prob - self.noise_prob

        if base_prob < -1e-12:
            raise ValueError("slippery_prob + wind_prob + noise_prob must be <= 1")

        if base_prob > 0:
            add_transition(action, base_prob)

        # Slippery transitions
        if self.slippery_prob > 0:
            adjacent = self._adjacent_actions(action)
            slip_each = self.slippery_prob / len(adjacent)
            for slip_action in adjacent:
                add_transition(slip_action, slip_each)

        # Wind transition
        if self.wind_prob > 0:
            add_transition(self.wind_direction, self.wind_prob)

        # Random noise transitions
        if self.noise_prob > 0:
            if self.noise_probs is None:
                noise_each = self.noise_prob / len(self.actions)
                for noise_action in self.actions:
                    add_transition(noise_action, noise_each)
            else:
                for noise_action, p in zip(self.actions, self.noise_probs):
                    add_transition(noise_action, self.noise_prob * p)

        return probs

    def expected_reward(self, state):
        reward_config = self.terminal_rewards.get(state, self.default_reward)
        return self._expected_reward_config(reward_config)

    def _expected_reward_config(self, reward_config):
        if isinstance(reward_config, (int, float)):
            return reward_config

        if isinstance(reward_config, dict):
            reward_type = reward_config.get("type")

            if reward_type == "constant":
                return reward_config["value"]

            if reward_type == "choice":
                values = reward_config["values"]
                probs = reward_config["probs"]
                return sum(v * p for v, p in zip(values, probs))

            if reward_type == "normal":
                return reward_config["mean"]

        raise ValueError(f"Unknown reward config: {reward_config}")

    def _deterministic_next_state(self, state, action):
        r, c = state

        if action == "up":
            candidate = (r - 1, c)
        elif action == "down":
            candidate = (r + 1, c)
        elif action == "left":
            candidate = (r, c - 1)
        elif action == "right":
            candidate = (r, c + 1)
        else:
            raise ValueError(f"Unknown action: {action}")

        if self._inside_grid(candidate):
            return candidate

        return state

    def _inside_grid(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _reward(self, state):
        reward_config = self.terminal_rewards.get(state, self.default_reward)
        return self._sample_reward(reward_config)

    def _sample_reward(self, reward_config):
        if isinstance(reward_config, (int, float)):
            return reward_config

        if isinstance(reward_config, dict):
            reward_type = reward_config.get("type")

            if reward_type == "constant":
                return reward_config["value"]

            if reward_type == "choice":
                values = reward_config["values"]
                probs = reward_config["probs"]
                return self.rng.choice(values, p=probs)

            if reward_type == "normal":
                mean = reward_config["mean"]
                std = reward_config["std"]
                return self.rng.normal(mean, std)

        raise ValueError(f"Unknown reward config: {reward_config}")

    def render(self):
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                state = (r, c)
                if state == self.current_state:
                    row.append("A")
                elif state == self.start_state:
                    row.append("S")
                elif state in self.terminal_rewards:
                    row.append("T")
                else:
                    row.append(".")
            print(" ".join(row))
        print()