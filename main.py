import gymnasium as gym
from functions import evolve



def main():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    evolve(env)
    env.close()

    env_render = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env_render.close()



if __name__ == "__main__":
    main()