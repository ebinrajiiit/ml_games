"""
Case study (Lecture 24): defining an RL environment.

Campus Courier: a delivery robot starts at the hostel and must reach the library.
Row 2 is a rain puddle with a dry gap only on the far left.

    col:  0   1   2   3   4
    row 0 .   .   L   .   .        L = library (goal, episode ends)
    row 1 .   .   .   .   .
    row 2 .   ~   ~   ~   ~        ~ = puddle (robot can enter, but it costs)
    row 3 .   .   .   .   .
    row 4 .   .   H   .   .        H = hostel (start)

Run:  pip install gymnasium numpy
      python campus_courier_env.py
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CampusCourier(gym.Env):
    """State: the robot's cell, encoded as one integer 0..24 (row * 5 + col).
    Actions: 0 up, 1 right, 2 down, 3 left. Bumping into the edge leaves it where it is.
    Reward: chosen by the designer. That is the point of the exercise."""

    metadata = {"render_modes": ["ansi"]}
    ROWS, COLS = 5, 5
    START, GOAL = (4, 2), (0, 2)
    PUDDLE = {(2, 1), (2, 2), (2, 3), (2, 4)}
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, step_reward=-1.0, puddle_reward=-10.0, goal_reward=10.0, max_steps=30):
        self.observation_space = spaces.Discrete(self.ROWS * self.COLS)
        self.action_space = spaces.Discrete(4)
        self.step_reward, self.puddle_reward, self.goal_reward = step_reward, puddle_reward, goal_reward
        self.max_steps = max_steps

    def _obs(self):
        return self.pos[0] * self.COLS + self.pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.t = self.START, 0
        return self._obs(), {}

    def step(self, action):
        dr, dc = self.MOVES[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            self.pos = (r, c)
        self.t += 1
        terminated = self.pos == self.GOAL                 # reached the library: a true end
        truncated = self.t >= self.max_steps               # out of time: an artificial cut-off
        if terminated:
            reward = self.goal_reward
        elif self.pos in self.PUDDLE:
            reward = self.puddle_reward
        else:
            reward = self.step_reward
        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        rows = []
        for r in range(self.ROWS):
            rows.append(" ".join("R" if (r, c) == self.pos else "L" if (r, c) == self.GOAL
                                 else "~" if (r, c) in self.PUDDLE else "." for c in range(self.COLS)))
        return "\n".join(rows)


def discounted_return(rewards, gamma):
    return sum(gamma ** k * r for k, r in enumerate(rewards))


if __name__ == "__main__":
    env = CampusCourier()
    rng = np.random.default_rng(0)
    returns, reached = [], 0
    for episode in range(1000):                            # a robot that moves at random
        obs, _ = env.reset(seed=episode)
        rewards, done = [], False
        while not done:
            obs, r, terminated, truncated, _ = env.step(int(rng.integers(4)))
            rewards.append(r)
            done = terminated or truncated
        returns.append(discounted_return(rewards, 0.95))
        reached += terminated
    print(f"Random robot: reached the library in {reached / 10:.1f}% of 1000 episodes,"
          f" average discounted return {np.mean(returns):.2f} (gamma = 0.95)")

    dry_path = [3, 3, 0, 0, 0, 1, 1, 0]                    # left, left, up x3, right x2, up
    obs, _ = env.reset()
    rewards = [env.step(a)[1] for a in dry_path]
    print("Dry route, 8 moves:", rewards, f"return {discounted_return(rewards, 0.95):.2f}")
    obs, _ = env.reset()
    rewards = [env.step(a)[1] for a in [0, 0, 0, 0]]
    print("Straight through the puddle, 4 moves:", rewards, f"return {discounted_return(rewards, 0.95):.2f}")
