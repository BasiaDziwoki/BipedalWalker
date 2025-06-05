import numpy as np
from config import GENOME_LENGTH, rng


class Individual:
    def __init__(self, genome=None):
        self.genome = genome if genome is not None else rng.uniform(-1, 1, GENOME_LENGTH).astype(np.float32)
        self.fitness = None

    def evaluate(self, env):
        observation, _ = env.reset(seed=int(rng.integers(2**10)))
        total_reward = 0.0
        step = 0
        done = False
        truncated = False

        weights = self.genome.reshape((4, 24))

        while not (done or truncated) and step < 1000:
            observation = np.nan_to_num(observation)
            action = np.tanh(np.dot(weights, observation))
            action = np.clip(action, -1, 1)
            observation, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

        self.fitness = total_reward
        return total_reward

    def act(self, observation):
        weights = self.genome.reshape((4, 24))
        observation = np.nan_to_num(observation)
        action = np.tanh(np.dot(weights, observation))
        action = np.clip(action, -1, 1)
        return action