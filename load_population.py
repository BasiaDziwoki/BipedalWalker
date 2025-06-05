import gymnasium as gym
from functions import load_generation, run_individual_in_env



generation_nr, population = load_generation(gen_number=173)
print(f"Wczytano generację nr {generation_nr} z {len(population)} osobnikami.")

best_individual = max(population, key=lambda ind: ind.fitness)

env = gym.make("BipedalWalker-v3", render_mode="human")

run_individual_in_env(env, best_individual)
env.close()
