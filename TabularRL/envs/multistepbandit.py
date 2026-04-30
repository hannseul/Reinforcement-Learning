"""Multi-step bandit environment for tabular RL experiments.

The environment is intentionally small and explicit.  It exposes the same
interface as GridWorld in this repository:

    reset() -> state
    step(action) -> next_state, reward, done, info
    allowed_actions(state) -> list[str]
    get_transition_probabilities(state, action) -> dict[state, prob]

States are represented as strings/tuples:
    "s0"              start state
    (branch, step)     branch state, step starts at 1
    "terminal"         absorbing terminal state

At the start state the agent chooses a branch.  On each branch it then chooses
one action per sequential step.  Transitions are deterministic along the branch;
rewards can be deterministic, normal, binomial/bernoulli, or discrete choices.
"""

from __future__ import annotations

import numpy as np


class MultiStepBandit:
    def __init__(
        self,
        num_branches=3,
        steps_per_branch=2,
        actions_per_step=1,
        start_reward=0.0,
        default_reward=0.0,
        reward_configs=None,
        seed=None,
        auto_reset_on_done=True,
    ):
        """
        Parameters
        ----------
        num_branches : int
            Number of branches available in the start state.
        steps_per_branch : int | list[int]
            Number of sequential states on each branch.
        actions_per_step : int | dict | callable
            Number of actions available at branch states.  If an int is given,
            every branch state has this many actions.  If a dict is given, keys
            are states (branch, step).  If callable, it is called as f(branch, step).
        start_reward : reward config
            Reward obtained when leaving the start state.
        default_reward : reward config
            Reward used when no custom reward is specified for a state-action pair.
        reward_configs : dict | None
            Optional mapping (state, action) -> reward config.  A reward config can be
            a number or one of:
                {"type": "constant", "value": x}
                {"type": "normal", "mean": mu, "std": sigma}
                {"type": "choice", "values": [...], "probs": [...]}
                {"type": "bernoulli", "p": p}          # rewards 0/1
                {"type": "binomial", "n": n, "p": p}
        seed : int | None
            Random seed.
        auto_reset_on_done : bool
            If True, current_state is reset to start after a terminal transition.
            This matches the GridWorld class in this repository.
        """
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")

        self.num_branches = int(num_branches)
        self.start_state = "s0"
        self.terminal_state = "terminal"
        self.terminal_states = {self.terminal_state}
        self.default_reward = default_reward
        self.start_reward = start_reward
        self.reward_configs = reward_configs or {}
        self.rng = np.random.default_rng(seed)
        self.auto_reset_on_done = auto_reset_on_done

        if isinstance(steps_per_branch, int):
            if steps_per_branch <= 0:
                raise ValueError("steps_per_branch must be positive")
            self.steps_per_branch = [int(steps_per_branch)] * self.num_branches
        else:
            if len(steps_per_branch) != self.num_branches:
                raise ValueError("steps_per_branch must have length num_branches")
            self.steps_per_branch = [int(x) for x in steps_per_branch]
            if any(x <= 0 for x in self.steps_per_branch):
                raise ValueError("all branch lengths must be positive")

        self.states = [self.start_state]
        for branch, length in enumerate(self.steps_per_branch):
            for step in range(1, length + 1):
                self.states.append((branch, step))
        self.states.append(self.terminal_state)

        self._actions_by_state = {}
        self._actions_by_state[self.start_state] = [f"branch_{b}" for b in range(self.num_branches)]
        self._actions_by_state[self.terminal_state] = []

        for branch, length in enumerate(self.steps_per_branch):
            for step in range(1, length + 1):
                state = (branch, step)
                n_actions = self._resolve_num_actions(actions_per_step, state)
                self._actions_by_state[state] = [f"a_{i}" for i in range(n_actions)]

        # Algorithms in this repository initialise Q over env.actions for every state.
        # Therefore env.actions is the union of all possible action labels.
        all_actions = []
        for actions in self._actions_by_state.values():
            for action in actions:
                if action not in all_actions:
                    all_actions.append(action)
        self.actions = all_actions

        self.current_state = self.start_state

    def _resolve_num_actions(self, actions_per_step, state):
        branch, step = state
        if isinstance(actions_per_step, int):
            n_actions = actions_per_step
        elif isinstance(actions_per_step, dict):
            n_actions = actions_per_step.get(state, actions_per_step.get(branch, 1))
        elif callable(actions_per_step):
            n_actions = actions_per_step(branch, step)
        else:
            raise TypeError("actions_per_step must be int, dict, or callable")

        if n_actions <= 0:
            raise ValueError(f"state {state} must have at least one action")
        return int(n_actions)

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def allowed_actions(self, state=None):
        state = self.current_state if state is None else state
        return list(self._actions_by_state.get(state, []))

    def step(self, action):
        state = self.current_state
        if action not in self.allowed_actions(state):
            raise ValueError(f"Action {action} is not allowed in state {state}")

        next_state = self._next_state(state, action)
        reward = self._reward(state, action)
        done = next_state in self.terminal_states

        self.current_state = self.start_state if (done and self.auto_reset_on_done) else next_state
        return next_state, reward, done, {}

    def _next_state(self, state, action):
        if state == self.start_state:
            branch = int(action.split("_")[1])
            return (branch, 1)

        if state in self.terminal_states:
            return self.terminal_state

        branch, step = state
        if step < self.steps_per_branch[branch]:
            return (branch, step + 1)
        return self.terminal_state

    def get_transition_probabilities(self, state, action):
        if state in self.terminal_states:
            return {self.terminal_state: 1.0}
        if action not in self.allowed_actions(state):
            raise ValueError(f"Action {action} is not allowed in state {state}")
        return {self._next_state(state, action): 1.0}

    def expected_reward(self, state, action=None, next_state=None):
        """Expected reward.

        Preferred usage is expected_reward(state, action).  If only one argument is
        supplied, this returns the default expected state reward for compatibility
        with older GridWorld-style code.  For action-dependent rewards, use two args.
        """
        if action is None:
            return self._expected_reward_config(self.default_reward)
        return self._expected_reward_config(self._reward_config(state, action))

    def expected_reward_for_transition(self, state, action, next_state=None):
        return self.expected_reward(state, action, next_state)

    def _reward(self, state, action):
        return self._sample_reward(self._reward_config(state, action))

    def _reward_config(self, state, action):
        if state == self.start_state:
            return self.reward_configs.get((state, action), self.start_reward)
        return self.reward_configs.get((state, action), self.default_reward)

    def _expected_reward_config(self, reward_config):
        if isinstance(reward_config, (int, float, np.integer, np.floating)):
            return float(reward_config)

        if isinstance(reward_config, dict):
            reward_type = reward_config.get("type")
            if reward_type == "constant":
                return float(reward_config["value"])
            if reward_type == "normal":
                return float(reward_config["mean"])
            if reward_type == "choice":
                values = reward_config["values"]
                probs = reward_config["probs"]
                return float(sum(v * p for v, p in zip(values, probs)))
            if reward_type == "bernoulli":
                return float(reward_config["p"])
            if reward_type == "binomial":
                return float(reward_config["n"] * reward_config["p"])

        raise ValueError(f"Unknown reward config: {reward_config}")

    def _sample_reward(self, reward_config):
        if isinstance(reward_config, (int, float, np.integer, np.floating)):
            return float(reward_config)

        if isinstance(reward_config, dict):
            reward_type = reward_config.get("type")
            if reward_type == "constant":
                return float(reward_config["value"])
            if reward_type == "normal":
                return float(self.rng.normal(reward_config["mean"], reward_config["std"]))
            if reward_type == "choice":
                return float(self.rng.choice(reward_config["values"], p=reward_config["probs"]))
            if reward_type == "bernoulli":
                return float(self.rng.binomial(1, reward_config["p"]))
            if reward_type == "binomial":
                return float(self.rng.binomial(reward_config["n"], reward_config["p"]))

        raise ValueError(f"Unknown reward config: {reward_config}")

    @classmethod
    def two_step_gaussian(
        cls,
        num_branches=10,
        actions_per_step=1,
        good_branch=0,
        good_mean=0.1,
        bad_mean=-0.1,
        std=1.0,
        seed=None,
    ):
        """Factory for the Blatt-7/8 multi-step bandit examples.

        Two sequential branch states.  Rewards are zero except at the second step.
        In each branch, action a_0 at the second step gives a Gaussian reward.
        One branch has mean good_mean, all other branches have mean bad_mean.
        """
        reward_configs = {}
        for branch in range(num_branches):
            mean = good_mean if branch == good_branch else bad_mean
            reward_configs[((branch, 2), "a_0")] = {"type": "normal", "mean": mean, "std": std}

        return cls(
            num_branches=num_branches,
            steps_per_branch=2,
            actions_per_step=actions_per_step,
            start_reward=0.0,
            default_reward=0.0,
            reward_configs=reward_configs,
            seed=seed,
        )

    def render(self):
        print(f"current_state: {self.current_state}")
        print(f"allowed_actions: {self.allowed_actions(self.current_state)}")
